# Text2SQL Agent

A local-model agent that answers questions about a database in plain English. It looks at the tables, writes a SQL `SELECT`, runs it, and explains the rows. Runs on Ollama (default `qwen3:4b`), no API keys.

```
question -> list_tables -> get_table_schema -> execute_sql -> answer
```

| File | What it does |
|---|---|
| `tools.py` | Three `@tool`s: `list_tables`, `get_table_schema`, `execute_sql`. The database engine arrives through the run config, so the tools work on any database |
| `agent.py` | `create_agent` with the tools and a short system prompt; the engine is passed in `config` and never shown to the model |
| `database.py` | Demo online-store database, 7 tables of synthetic data (created on first run; an older `sales.db` is upgraded in place) |
| `test_tools.py` | 21 deterministic tests: schema and joins, row cap, blocked writes, errors as data, no orphan rows |

## The database

| Table | Rows | What it holds |
|---|---|---|
| `users` | 200 | customers: name, email, country, signup date |
| `products` | 5 | the models sold, with category and list price |
| `sales` | 1,000 | line items: date, model, price, quantity, customer |
| `orders` | 1,000 | one per sale: status (delivered, shipped, processing, cancelled, returned) and shipping method |
| `payments` | 1,000 | one per order: method, amount, status (paid, failed, refunded) |
| `returns` | 100 | returned orders: reason, status, refund amount |
| `issues` | 250 | support tickets: category, priority, status, open and resolved dates, optionally tied to an order |

`orders.id` equals the sale it fulfils, and `get_table_schema` reports each foreign key (for example `orders.customer_id -> users.id`) so the model knows how to `JOIN`. All rows are seeded, so every run is identical. Sizes are the `N_*` constants at the top of `database.py`; after changing them run `python database.py` to rebuild (it drops and recreates only these demo tables).

Questions to try: *How many high-priority issues are still open? Which customer spent the most? How much money was refunded? Which return reason is most common? How many orders were cancelled?*

## Run

```bash
pip install -r requirements.txt
cp .env.example .env     # model, Ollama URL, database file and row limit live here (read with load_dotenv)
ollama pull qwen3:4b
python agent.py "Which model earned the most revenue?"
pytest
```

Example run (earlier, on the original 100-row table, checked against a direct SQL query): the agent calls `list_tables`, `get_table_schema`, then `SELECT model, SUM(price * quantity) ... LIMIT 1` and answers **Laptop**. On the current 1,000-row data a direct query gives Laptop at 673,200. A run takes about a minute on a local 4B model.

## Design notes

- **The model only writes language.** Plain code decides what is allowed: `execute_sql` accepts a single `SELECT` and nothing else, and returns at most 50 rows.
- **Errors are returned as data**, not raised: `{"error": "no such column: nope"}`. The model can read that and fix its query instead of the run crashing.
- **Small local models are shaky at SQL.** The system prompt forces the schema lookup first, and `temperature=0` keeps runs repeatable. Run `pytest` for the safety checks and compare agent answers with a direct query before trusting them.
- Synthetic data only. Point `DB_PATH` at your own SQLite file, or pass a different engine in `config`, to query something else (use a read-only database user for anything real).
