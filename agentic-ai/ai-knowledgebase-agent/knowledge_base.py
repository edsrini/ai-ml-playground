"""The knowledge base: documents in docs/ -> pieces -> embeddings -> a Chroma collection saved on disk.

Run `python knowledge_base.py` to load the documents (it skips the work if they are already loaded) and
`python knowledge_base.py --rebuild` to throw the collection away and load everything again.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads .env into environment variables
DOCS_DIR = Path(__file__).parent / os.getenv("DOCS_DIR")  # where the source documents live
CHROMA_DIR = Path(__file__).parent / os.getenv("CHROMA_DIR")  # where the vectors are saved
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))  # characters per piece
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))  # characters shared between neighbouring pieces


def get_embeddings():
    """The model that turns text into vectors. The same one must embed the pieces and the questions."""
    return OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL"), base_url=os.getenv("OLLAMA_BASE_URL"))


def get_vector_store(embeddings=None, collection_name=COLLECTION_NAME, persist=True):
    """Open the collection, creating it if needed. With persist=False it lives in memory only (used by tests)."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(CHROMA_DIR) if persist else None,
    )


def load_documents(docs_dir=DOCS_DIR):
    """Read every .pdf, .md and .txt file in the folder. Each page or file becomes a Document tagged with its source."""
    documents = []
    for path in sorted(Path(docs_dir).glob("*")):
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()  # one Document per page; needs a text-based PDF, not a scan
        elif path.suffix.lower() in (".md", ".txt"):
            loaded = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue  # TODO: other formats (docx, html) would go here
        for document in loaded:
            document.metadata["source"] = path.name  # so an answer can say which file it came from
        documents.extend(loaded)
    return documents


def split_documents(documents):
    """Cut documents into overlapping pieces that are small enough to search and to fit in a prompt."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)


def ingest(store, rebuild=False):
    """Load, split and embed the documents into the store. Returns how many pieces were added."""
    if rebuild:
        store.reset_collection()  # TODO: replace with "only re-embed files that changed"
    elif store.get(limit=1)["ids"]:  # something is already stored: do not add the same pieces again
        return 0
    pieces = split_documents(load_documents())
    if pieces:
        store.add_documents(pieces)
    return len(pieces)


if __name__ == "__main__":
    added = ingest(get_vector_store(), rebuild="--rebuild" in sys.argv)
    print(f"Added {added} pieces" if added else "Nothing to add: the knowledge base is already loaded (use --rebuild)")
