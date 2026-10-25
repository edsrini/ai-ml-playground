# AI Knowledge Base Agent (skeleton)

An agent that answers questions from your own documents, and from the web when the documents do not have the answer. It runs on local Ollama models, so no API key is needed for the knowledge base. This is a **skeleton**: the pieces work end to end, and the roadmap below lists what a real version still needs.

It has the same shape as [`text2sql-agent`](../text2sql-agent), with document search in place of SQL.

```
docs/ -> knowledge_base.py -> Chroma (saved on disk)
                                  |
question -> message_builder -> kb_agent <-> kb_tools (search_knowledge_base, web_search) -> answer with sources
```

| File | Role |
|---|---|
| `knowledge_base.py` | Loads `.pdf`, `.md` and `.txt` files from `docs/`, splits them into pieces, embeds them and saves them in Chroma. Skips the work if they are already loaded; `--rebuild` starts over |
| `tools.py` | `search_knowledge_base` (the closest passages, with source, page and distance) and `web_search` (Tavily, only when the documents do not have the answer). Errors come back as data |
| `agent.py` | The graph: prompt builder, model node, tool node and router. `python agent.py "question"` loads the documents if needed and answers |
| `test_tools.py` | 7 tests with fake embeddings and an in-memory store: no Ollama and no network |
| `docs/` | The documents. It ships with one small invented sample |
| `.env.example` | Models, folders, piece size, number of passages and the optional Tavily key |

## Run

```bash
pip install -r requirements.txt
cp .env.example .env              # then add TAVILY_API_KEY if you want the web fallback
ollama pull nomic-embed-text      # once: the model that makes the vectors
python knowledge_base.py          # load docs/ into the knowledge base (the agent also does this on first use)
python agent.py "What is a transformer?"
pytest
```

Add your own `.pdf`, `.md` or `.txt` files to `docs/`, then run `python knowledge_base.py --rebuild`. A PDF has to contain real text: a scanned PDF (pictures of pages) needs OCR first.

## Design notes

- **The model reads and writes; plain code sets the rules.** The number of passages, the fields returned, the step cap (`recursion_limit` 14) and the error handling are all in code, not in the prompt.
- **Answers cite sources.** Each passage carries its file name and page, and the prompt asks the model to name them.
- **Errors come back as data.** A broken store or a missing Tavily key returns `{"error": ...}`, so the model can say so and the run does not crash.
- **The vectors are saved on disk** (`chroma-data/`, git-ignored), so the documents are embedded once, not on every run. The same embedding model must be used for loading and for questions: if you change `OLLAMA_EMBED_MODEL`, run `--rebuild`.
- **Everything configurable is in `.env`**, read with `load_dotenv()` and `os.getenv(...)`.

## Roadmap (what is not done yet)

- [ ] Re-embed only the files that changed, instead of all or nothing (`ingest` in `knowledge_base.py`)
- [ ] A relevance threshold in plain code: treat passages that are too far away as "not found" before the model sees them
- [ ] More file types (docx, html) and OCR for scanned PDFs
- [ ] An evaluation set: questions with known answers, run after every prompt or chunking change, to measure accuracy and hallucinations
- [ ] Memory across questions (a checkpointer with a thread id per user)
- [ ] Access control: who may search which documents
- [ ] An answer-quality guard: refuse to answer when the passages do not support it
