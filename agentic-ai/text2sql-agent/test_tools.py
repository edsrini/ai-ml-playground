"""Deterministic tests for the tools: no LLM, no network. Run with: pytest"""
import pytest
from sqlalchemy import create_engine

import database
from tools import execute_sql, get_table_schema, list_tables


@pytest.fixture
def cfg(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    database.create_all(engine)
    return {"configurable": {"db_engine": engine}}


def run_sql(cfg, query):
    return execute_sql.invoke({"query": query}, config=cfg)


def test_list_tables(cfg):
    assert list_tables.invoke({}, config=cfg) == ["issues", "orders", "payments", "products", "returns", "sales", "users"]


def test_schema_has_all_columns(cfg):
    columns = [c["name"] for c in get_table_schema.invoke({"table_name": "sales"}, config=cfg)]
    assert columns == ["id", "transaction_date", "model", "price", "quantity", "customer_id"]


def test_unknown_table_returns_error(cfg):
    assert "error" in get_table_schema.invoke({"table_name": "nope"}, config=cfg)[0]


def test_select_and_row_cap(cfg):
    assert run_sql(cfg, "SELECT COUNT(*) AS n FROM sales") == [{"n": database.N_SALES}]
    assert len(run_sql(cfg, "SELECT * FROM sales")) == 50


@pytest.mark.parametrize("query", [
    "DELETE FROM sales", "DROP TABLE sales", "SELECT 1; DROP TABLE sales",
    "WITH x AS (SELECT 1) DELETE FROM sales", "UPDATE sales SET price = 0",
])
def test_writes_are_blocked_and_data_survives(cfg, query):
    assert "error" in run_sql(cfg, query)[0]
    assert run_sql(cfg, "SELECT COUNT(*) AS n FROM sales") == [{"n": database.N_SALES}]


def test_bad_sql_returns_error_not_exception(cfg):
    assert "no such column" in run_sql(cfg, "SELECT nope FROM sales")[0]["error"]


def test_schema_shows_how_tables_join(cfg):
    columns = get_table_schema.invoke({"table_name": "orders"}, config=cfg)
    links = {c["name"]: c["references"] for c in columns if "references" in c}
    assert links == {"sale_id": "sales.id", "customer_id": "users.id"}


def test_row_counts(cfg):
    counts = {t: run_sql(cfg, f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] for t in
              ["users", "products", "sales", "orders", "payments", "returns", "issues"]}
    assert counts == {"users": database.N_USERS, "products": 5, "sales": database.N_SALES, "orders": database.N_SALES,
                      "payments": database.N_SALES, "returns": database.N_RETURNED, "issues": database.N_ISSUES}


@pytest.mark.parametrize("query", [
    "SELECT COUNT(*) AS n FROM orders WHERE sale_id NOT IN (SELECT id FROM sales)",
    "SELECT COUNT(*) AS n FROM orders WHERE customer_id NOT IN (SELECT id FROM users)",
    "SELECT COUNT(*) AS n FROM payments WHERE order_id NOT IN (SELECT id FROM orders)",
    "SELECT COUNT(*) AS n FROM returns WHERE order_id NOT IN (SELECT id FROM orders)",
    "SELECT COUNT(*) AS n FROM issues WHERE order_id IS NOT NULL AND order_id NOT IN (SELECT id FROM orders)",
    "SELECT COUNT(*) AS n FROM issues WHERE customer_id NOT IN (SELECT id FROM users)",
])
def test_no_orphan_rows(cfg, query):
    assert run_sql(cfg, query) == [{"n": 0}]


def test_returned_orders_match_returns_and_refunds(cfg):
    assert run_sql(cfg, "SELECT COUNT(*) AS n FROM orders WHERE status = 'returned'") == [{"n": database.N_RETURNED}]
    assert run_sql(cfg, "SELECT COUNT(*) AS n FROM returns r JOIN orders o ON o.id = r.order_id "
                        "WHERE o.status = 'returned'") == [{"n": database.N_RETURNED}]
    assert run_sql(cfg, "SELECT COUNT(*) AS n FROM payments WHERE status = 'refunded'") == [{"n": database.N_RETURNED}]


def test_known_answer_for_top_revenue_model(cfg):
    # the data is seeded, so this answer is fixed: the number to compare the agent's reply with
    top = run_sql(cfg, "SELECT model, SUM(price * quantity) AS revenue FROM sales GROUP BY model "
                       "ORDER BY revenue DESC LIMIT 1")
    assert top == [{"model": "Laptop", "revenue": 673200.0}]


def test_an_old_database_with_only_sales_is_upgraded(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    database.create_sales_table(engine)  # what the earlier version of this project created
    database.create_all(engine)
    cfg = {"configurable": {"db_engine": engine}}
    assert "issues" in list_tables.invoke({}, config=cfg)
    assert run_sql(cfg, "SELECT COUNT(*) AS n FROM sales") == [{"n": database.N_SALES}]
