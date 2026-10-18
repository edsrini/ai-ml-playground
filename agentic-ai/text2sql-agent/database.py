"""The demo database: a small online-store schema with synthetic data (no real people or payments).

sales      1,000 line items                              users      200 customers
orders     one fulfilment record per sale                payments   one payment per order
returns    100 returned orders                           issues     250 customer-support tickets
products   the 5 models that are sold

Everything is seeded, so every run produces the same rows and agent answers can be checked against direct SQL.
"""
import os
import random
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

load_dotenv()  # reads .env into environment variables
DB_PATH = Path(__file__).parent / os.getenv("DB_PATH")  # relative to this folder; an absolute path also works

# How much demo data to create. Change these and run `python database.py` to rebuild with a different size.
N_USERS = 200
N_SALES = 1000
N_RETURNED = N_SALES // 10  # orders that were sent back
N_CANCELLED = N_SALES // 20
N_ISSUES = N_SALES // 4  # support tickets
N_ACCOUNT_ISSUES = 20  # tickets about signing in, with no order attached

MODELS = {"Laptop": 1200.0, "Phone": 800.0, "Tablet": 450.0, "Monitor": 300.0, "Headset": 90.0}
CATEGORIES = {"Laptop": "Computers", "Phone": "Phones", "Tablet": "Tablets", "Monitor": "Accessories", "Headset": "Accessories"}

# Created in this order, so every table exists before the ones that point at it
STORE_DDL = [
    "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, country TEXT, signup_date TEXT)",
    "CREATE TABLE products (id INTEGER PRIMARY KEY, model TEXT, category TEXT, list_price REAL)",
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, sale_id INTEGER REFERENCES sales(id), "
    "customer_id INTEGER REFERENCES users(id), order_date TEXT, status TEXT, shipping_method TEXT)",
    "CREATE TABLE payments (id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(id), method TEXT, "
    "amount REAL, status TEXT, paid_at TEXT)",
    "CREATE TABLE returns (id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(id), reason TEXT, "
    "requested_date TEXT, status TEXT, refund_amount REAL)",
    "CREATE TABLE issues (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES users(id), "
    "order_id INTEGER REFERENCES orders(id), category TEXT, priority TEXT, status TEXT, "
    "opened_date TEXT, resolved_date TEXT, description TEXT)",
]

FIRST_NAMES = ["Alex", "Sam", "Maya", "Omar", "Priya", "Liam", "Noor", "Chen", "Sara", "Diego",
               "Hana", "Ravi", "Zoe", "Ahmed", "Lena", "Marco", "Aisha", "Kofi", "Ivy", "Tariq"]
LAST_NAMES = ["Stone", "Patel", "Haddad", "Kim", "Silva", "Novak", "Reed", "Khan", "Moreau", "Cole"]
ISSUE_TEXT = {
    "delivery_delay": "Order has not arrived within the promised window.",
    "billing": "Charged twice for the same order.",
    "defective_product": "Item stopped working shortly after delivery.",
    "refund_status": "Refund has not shown up after the return was approved.",
    "account_access": "Cannot sign in to the account.",
}


def get_engine():
    """Return an engine for the database, creating whatever tables are missing (an older sales.db is upgraded)."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    create_all(engine)
    return engine


def create_all(engine):
    tables = set(inspect(engine).get_table_names())
    if "sales" not in tables:
        create_sales_table(engine)
    if "users" not in tables:
        create_store_tables(engine)


def create_sales_table(engine):
    random.seed(42)  # same rows every time, so answers are repeatable
    rows = []
    for i in range(1, N_SALES + 1):
        model = random.choice(list(MODELS))
        sale_date = date(2026, 1, 1) + timedelta(days=random.randint(0, 270))
        rows.append({"id": i, "d": sale_date.isoformat(), "m": model, "p": MODELS[model],
                     "q": random.randint(1, 5), "c": random.randint(1, N_USERS)})
    with engine.begin() as conn:  # begin() commits when the block ends
        conn.execute(text(
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, transaction_date TEXT, model TEXT, "
            "price REAL, quantity INTEGER, customer_id INTEGER)"))
        conn.execute(text("INSERT INTO sales VALUES (:id, :d, :m, :p, :q, :c)"), rows)  # one batch, not 1,000 calls


def rebuild(engine):
    """Drop the demo tables and create them again (demo data only: never point this at a real database)."""
    with engine.begin() as conn:
        for table in ["issues", "returns", "payments", "orders", "products", "users", "sales"]:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    create_all(engine)


def create_store_tables(engine):
    """Build users, products, orders, payments, returns and issues from the rows already in `sales`."""
    rng = random.Random(7)  # its own generator, so the original sales rows never change
    with engine.begin() as conn:
        for ddl in STORE_DDL:
            conn.execute(text(ddl))
        sales = conn.execute(text(
            "SELECT id, transaction_date, model, price, quantity, customer_id FROM sales ORDER BY id")).fetchall()

        users = [{"i": uid, "n": f"{FIRST_NAMES[(uid - 1) % 20]} {LAST_NAMES[((uid - 1) // 20) % 10]}",  # 200 unique names
                  "e": f"user{uid}@example.com", "c": rng.choice(["US", "UAE", "UK", "India", "Canada"]),
                  "s": (date(2025, 1, 1) + timedelta(days=rng.randint(0, 360))).isoformat()}
                 for uid in range(1, N_USERS + 1)]
        conn.execute(text("INSERT INTO users VALUES (:i, :n, :e, :c, :s)"), users)
        conn.execute(text("INSERT INTO products VALUES (:i, :m, :c, :p)"),
                     [{"i": pid, "m": model, "c": CATEGORIES[model], "p": price}
                      for pid, (model, price) in enumerate(MODELS.items(), start=1)])

        returned = set(rng.sample([s.id for s in sales], N_RETURNED))  # these orders were sent back
        cancelled = set(rng.sample([s.id for s in sales if s.id not in returned], N_CANCELLED))
        order_date = {s.id: date.fromisoformat(s.transaction_date) for s in sales}
        amount = {s.id: s.price * s.quantity for s in sales}

        orders, payments = [], []
        for s in sales:  # one order and one payment per sale; the order id equals the sale id
            status = ("returned" if s.id in returned else "cancelled" if s.id in cancelled
                      else rng.choices(["delivered", "shipped", "processing"], [85, 10, 5])[0])
            orders.append({"i": s.id, "c": s.customer_id, "d": s.transaction_date, "s": status,
                           "m": rng.choice(["standard", "express"])})
            pay_status = "refunded" if s.id in returned else "failed" if s.id in cancelled else "paid"
            payments.append({"i": s.id, "m": rng.choice(["card", "paypal", "bank_transfer"]), "a": amount[s.id],
                             "s": pay_status, "d": s.transaction_date})
        conn.execute(text("INSERT INTO orders VALUES (:i, :i, :c, :d, :s, :m)"), orders)
        conn.execute(text("INSERT INTO payments VALUES (:i, :i, :m, :a, :s, :d)"), payments)

        returns = []
        for rid, oid in enumerate(sorted(returned), start=1):
            status = rng.choice(["refunded", "approved", "pending"])
            returns.append({"i": rid, "o": oid, "r": rng.choice(["defective", "wrong_item", "changed_mind", "late_delivery"]),
                            "d": (order_date[oid] + timedelta(days=rng.randint(5, 20))).isoformat(), "s": status,
                            "a": 0.0 if status == "pending" else amount[oid]})
        conn.execute(text("INSERT INTO returns VALUES (:i, :o, :r, :d, :s, :a)"), returns)

        customer_of = {s.id: s.customer_id for s in sales}
        returned_ids = sorted(returned)
        n_after_returns = int(len(returned_ids) * 0.8)  # most returned orders also got a ticket
        n_ordinary = N_ISSUES - N_ACCOUNT_ISSUES
        issues = []
        for iid in range(1, N_ISSUES + 1):
            if iid <= n_after_returns:  # the first tickets follow returned orders
                oid, category = returned_ids[iid - 1], rng.choice(["defective_product", "refund_status"])
            elif iid <= n_ordinary:  # the next ones follow ordinary orders
                oid = rng.choice([s.id for s in sales])
                category = rng.choice(["delivery_delay", "billing", "defective_product"])
            else:  # the last ones are account problems with no order
                oid, category = None, "account_access"
            customer = customer_of[oid] if oid else rng.randint(1, N_USERS)
            opened = (order_date[oid] if oid else date(2026, 5, 1)) + timedelta(days=rng.randint(1, 10))
            status = rng.choice(["open", "in_progress", "resolved", "closed"])
            resolved = opened + timedelta(days=rng.randint(1, 14)) if status in ("resolved", "closed") else None
            issues.append({"i": iid, "c": customer, "o": oid, "cat": category, "p": rng.choice(["low", "medium", "high"]),
                           "s": status, "od": opened.isoformat(), "rd": resolved.isoformat() if resolved else None,
                           "t": ISSUE_TEXT[category]})
        conn.execute(text("INSERT INTO issues VALUES (:i, :c, :o, :cat, :p, :s, :od, :rd, :t)"), issues)


if __name__ == "__main__":
    rebuild(create_engine(f"sqlite:///{DB_PATH}"))
    print(f"Rebuilt {DB_PATH}")
