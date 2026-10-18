"""Three database tools an LLM can call. None of them hardcodes a connection: the SQLAlchemy engine is read from
the run config (config["configurable"]["db_engine"]), so the same tools work on any database.

Plain code decides what is allowed. The model only chooses which tool to call and writes the SQL text.
"""
import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError


load_dotenv()  # reads .env into environment variables
MAX_ROWS = int(os.getenv("MAX_ROWS"))  # most rows one answer may return


@tool
def list_tables(config: RunnableConfig) -> List[str]:
    """List the names of all tables in the database."""
    return inspect(config["configurable"]["db_engine"]).get_table_names()


@tool
def get_table_schema(table_name: str, config: RunnableConfig) -> List[Dict[str, str]]:
    """Return the column names and types of one table. A column that points at another table also lists
    "references" (for example orders.customer_id references users.id), which tells you how to JOIN."""
    inspector = inspect(config["configurable"]["db_engine"])
    if table_name not in inspector.get_table_names():  # errors come back as data so the model can correct itself
        return [{"error": f"No table named '{table_name}'. Call list_tables first."}]
    links = {fk["constrained_columns"][0]: f"{fk['referred_table']}.{fk['referred_columns'][0]}"
             for fk in inspector.get_foreign_keys(table_name)}
    columns = []
    for col in inspector.get_columns(table_name):
        entry = {"name": col["name"], "type": str(col["type"])}
        if col["name"] in links:
            entry["references"] = links[col["name"]]
        columns.append(entry)
    return columns


@tool
def execute_sql(query: str, config: RunnableConfig) -> List[Dict]:
    """Run one read-only SQL SELECT query and return the rows as a list of dicts."""
    statement = query.strip().rstrip(";")
    if not statement.lower().startswith("select") or ";" in statement:
        return [{"error": "Only a single SELECT statement is allowed."}]
    try:
        with config["configurable"]["db_engine"].connect() as conn:
            rows = conn.execute(text(statement)).fetchmany(MAX_ROWS)
            return [dict(row._mapping) for row in rows]
    except SQLAlchemyError as error:  # e.g. "no such column": hand the message back so the model can fix its SQL
        return [{"error": str(getattr(error, "orig", None) or error)}]


TOOLS = [list_tables, get_table_schema, execute_sql]
