# Log Search Lab

A local Elasticsearch + Kibana stack holding 1,000,000 synthetic application logs, and an agent on a local model that searches them to find what caused an incident. It follows the same structure as [`text2sql-agent`](../text2sql-agent), with log search in place of SQL. No cloud, no API keys, no real data.

```
START -> message_builder -> log_agent -> (tool call?) -> log_tools -> back to log_agent
                                      \-> (no tool call) -> END
```

| File | Same role in text2sql-agent | What it does |
|---|---|---|
| `database.py` | `database.py` | Creates the demo logs and loads them into Elasticsearch (`python database.py`) |
| `tools.py` | `tools.py` | Three tools: `get_log_fields`, `log_stats` (counts grouped by a field), `search_logs` (newest matching lines) |
| `agent.py` | `agent.py` | The graph: prompt builder, model node, tool node, router; run it with a question |
| `message.py` | `message.py` | Our own copy of langgraph's `MessagesState` |
| `test_tools.py` | `test_tools.py` | 8 tests with a stub client: no server and no LLM needed |
| `docker-compose.yml` | none | Elasticsearch 9.5.5 and Kibana 9.5.5 on `127.0.0.1`, security off (local only) |

## Run

```bash
pip install -r requirements.txt
cp .env.example .env              # URLs, index name, log count and date range, Ollama model
docker compose up -d              # first run downloads about 3 GB; wait until both are healthy (docker compose ps)
python database.py                # loads 1,000,000 logs (about 40 seconds)
python agent.py "What caused the errors on 8 October?"
pytest
```

Kibana is at http://localhost:5601. Open **Discover** directly (http://localhost:5601/app/discover). The `app-logs` data view and two saved searches (*Errors only*, *DB timeouts*) are already created, and the time range defaults to 1-15 Oct 2026.

## The hidden incidents

| When (UTC) | Service | Error code |
|---|---|---|
| 8 Oct, 02:00-02:45 | `claims-ingest` | `DB_TIMEOUT` |
| 11 Oct, 14:00-14:20 | `eligibility-api` | `UPSTREAM_503` |
| 13 Oct, 09:00-09:30 | `payment-gateway` | `PAYMENT_DECLINED` |

Everything else is normal traffic with about 2% errors. Ask the agent about each one and compare its answer with Kibana before trusting it.

Example run for the 8 October question: the agent called `log_stats` grouped by error code, saw `DB_TIMEOUT` far ahead of the rest (5,173 of 6,253 errors that day), called `search_logs` for that code, and answered `claims-ingest` / `DB_TIMEOUT`, with an example message. It took about 30 seconds.

## Design notes

- **The model only reads and reasons.** Plain code sets the limits: the index is fixed, results are capped at 20 lines, and errors from Elasticsearch come back as data (`{"error": ...}`) so the model can react instead of the run crashing.
- **A step cap** (`recursion_limit` 14) stops the agent from looping on tool calls.
- **Seeded data**, so the same command always creates the same logs and answers can be checked.
- **Local only.** Security is off and ports bind to `127.0.0.1`. Never expose this stack as it is.
- **Disk.** Elasticsearch data lives in the `esdata` Docker volume inside Docker Desktop's disk image. `docker compose down` stops the stack; add `-v` to delete the data too.
- Kibana's *Elasticsearch* start page (`/app/elasticsearch/home`) shows "Unable to load page" because it calls a security API that does not exist when security is off. Use Discover instead.
