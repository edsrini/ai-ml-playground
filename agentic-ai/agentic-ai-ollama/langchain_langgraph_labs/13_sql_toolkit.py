"""Database toolkit for a text-to-SQL agent: three tools (list tables, get schema, run SQL) over a SQLite sales table.

The tools never hardcode a connection. They read the engine from the run config, so the same tools work on any
database. This file only builds and tests the tools; an agent would call them through bind_tools (see lab 12).
"""
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

DB_PATH = Path(__file__).parent / "sales.db"
MAX_ROWS = 50  # an agent does not need more than this in one answer


# Build the demo database once: 100 synthetic sales rows
def create_sales_db():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.begin() as conn:  # begin() commits when the block ends
        conn.execute(text(
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, transaction_date TEXT, model TEXT, "
            "price REAL, quantity INTEGER, customer_id INTEGER)"))
        random.seed(42)  # same rows every time
        models = {"Laptop": 1200.0, "Phone": 800.0, "Tablet": 450.0, "Monitor": 300.0, "Headset": 90.0}
        for i in range(1, 101):
            model = random.choice(list(models))
            sale_date = date(2026, 1, 1) + timedelta(days=random.randint(0, 270))
            conn.execute(
                text("INSERT INTO sales VALUES (:id, :d, :m, :p, :q, :c)"),
                {"id": i, "d": sale_date.isoformat(), "m": model, "p": models[model],
                 "q": random.randint(1, 5), "c": random.randint(1, 20)})
    return engine


# Tool 1: which tables exist? The engine arrives through the run config, not as an argument the model fills in
@tool
def list_tables(config: RunnableConfig) -> List[str]:
    """List the names of all tables in the database."""
    inspector = inspect(config["configurable"]["db_engine"])
    return inspector.get_table_names()


# Tool 2: what columns does one table have?
@tool
def get_table_schema(table_name: str, config: RunnableConfig) -> List[Dict[str, str]]:
    """Return the column names and types of one table."""
    inspector = inspect(config["configurable"]["db_engine"])
    if table_name not in inspector.get_table_names():  # return the problem as data so the model can correct itself
        return [{"error": f"No table named '{table_name}'. Call list_tables first."}]
    columns = inspector.get_columns(table_name)
    return [{"name": col["name"], "type": str(col["type"])} for col in columns]  # str() so the type is plain text


# Tool 3: run a read-only query. Plain code, not the model, decides what is allowed.
@tool
def execute_sql(query: str, config: RunnableConfig) -> List[Dict]:
    """Run a read-only SQL SELECT query and return the rows as a list of dicts."""
    statement = query.strip().rstrip(";")
    if not statement.lower().startswith("select") or ";" in statement:
        return [{"error": "Only a single SELECT statement is allowed."}]
    try:
        with config["configurable"]["db_engine"].connect() as conn:
            result = conn.execute(text(statement))
            return [dict(row._mapping) for row in result.fetchmany(MAX_ROWS)]
    except SQLAlchemyError as error:  # bad column or table name: hand the message back so the model can fix its SQL
        return [{"error": str(error.orig) if getattr(error, "orig", None) else str(error)}]


def demo_inspector(db_engine):
    print("=== Inspector ===\n")
    inspector = inspect(db_engine)  # reads schema information, without writing SQL
    print("tables: ", inspector.get_table_names())
    print("columns:", [col["name"] for col in inspector.get_columns("sales")])
    with db_engine.connect() as conn:  # SQLAlchemy 2.x: db_engine.execute() is gone, use a connection
        df = pd.read_sql(text("SELECT * FROM sales LIMIT 10"), conn)
    print(df.to_string(index=False))


def demo_tools(db_engine):
    print("\n=== Tools ===\n")
    # Tools find the engine in config["configurable"]; swap the engine and the same tools work on another database
    config = {"configurable": {"db_engine": db_engine}}

    print("list_tables:     ", list_tables.invoke({}, config=config))
    print("get_table_schema:", get_table_schema.invoke({"table_name": "sales"}, config=config))
    rows = execute_sql.invoke({"query": "SELECT * FROM sales LIMIT 3"}, config=config)
    print("execute_sql:     ", rows)
    print("row count:       ", execute_sql.invoke({"query": "SELECT COUNT(*) AS n FROM sales"}, config=config))
    print("blocked write:   ", execute_sql.invoke({"query": "DELETE FROM sales"}, config=config))

    # The model only sees these (config is hidden from it): name, description and the arguments it must supply
    for t in (list_tables, get_table_schema, execute_sql):
        print(f"schema for the model -> {t.name}: {t.args}")


if __name__ == "__main__":
    if not DB_PATH.exists():
        create_sales_db()
    engine = create_engine(f"sqlite:///{DB_PATH}")
    demo_inspector(engine)
    demo_tools(engine)
