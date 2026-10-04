"""The library's data and the tools the assistant can call. The model never touches CATALOG directly."""
from langchain_core.tools import tool

# title -> book record; copies_available is mutated by checkout/return (in-memory demo data)
CATALOG = {
    "Dune": {"author": "Frank Herbert", "copies_available": 2},
    "Neuromancer": {"author": "William Gibson", "copies_available": 0},
    "The Hobbit": {"author": "J.R.R. Tolkien", "copies_available": 1},
    "Foundation": {"author": "Isaac Asimov", "copies_available": 3},
}
LOANS: list[tuple[str, str]] = []  # (member_name, title) for every book currently checked out


def _find_title(title: str) -> str | None:
    return next((t for t in CATALOG if t.lower() == title.strip().lower()), None)  # case-insensitive exact match


@tool
def search_books(query: str) -> str:
    """Search the library catalog by title or author and report how many copies are available."""
    q = query.lower()
    hits = [f"{t} by {b['author']} ({b['copies_available']} available)"
            for t, b in CATALOG.items() if q in t.lower() or q in b["author"].lower()]
    return "\n".join(hits) if hits else "No books found."


@tool
def checkout_book(title: str, member_name: str) -> str:
    """Check out one copy of a book (exact title) for a library member."""
    found = _find_title(title)
    if not found:
        return f"'{title}' is not in the catalog."
    if CATALOG[found]["copies_available"] == 0:  # refuse instead of going negative
        return f"'{found}' has no copies available right now."
    CATALOG[found]["copies_available"] -= 1
    LOANS.append((member_name, found))
    return f"Checked out '{found}' for {member_name}."


@tool
def return_book(title: str, member_name: str) -> str:
    """Return a book (exact title) that a library member previously checked out."""
    found = _find_title(title)
    if not found or (member_name, found) not in LOANS:
        return f"No loan of '{title}' found for {member_name}."
    LOANS.remove((member_name, found))
    CATALOG[found]["copies_available"] += 1
    return f"Returned '{found}' from {member_name}."


TOOLS = [search_books, checkout_book, return_book]
