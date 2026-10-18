"""The demo logs: synthetic application logs stored in an Elasticsearch index (no real data).

Run `python database.py` to (re)load them. The data is seeded, so every run creates the same logs.
Three incidents are hidden in it so the agent has something to find:
  8 Oct 02:00-02:45   claims-ingest    DB_TIMEOUT
  11 Oct 14:00-14:20  eligibility-api  UPSTREAM_503
  13 Oct 09:00-09:30  payment-gateway  PAYMENT_DECLINED
"""
import os
import random
import uuid
from datetime import datetime, timedelta

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv()  # reads .env into environment variables
ES_URL = os.getenv("ES_URL")
LOG_INDEX = os.getenv("LOG_INDEX")
N_LOGS = int(os.getenv("N_LOGS"))
START = datetime.fromisoformat(os.getenv("LOG_START"))
END = datetime.fromisoformat(os.getenv("LOG_END"))

# service -> the endpoints it serves
SERVICES = {
    "claims-ingest": ["POST /claims", "POST /claims/batch"],
    "eligibility-api": ["GET /eligibility"],
    "payment-gateway": ["POST /payments", "POST /refunds"],
    "auth-service": ["POST /login", "POST /token"],
    "notification-worker": ["SEND email", "SEND sms"],
}
BUSY = [40, 25, 15, 12, 8]  # how busy each service is, in the same order as SERVICES

# error code -> (message, the service it normally comes from)
ERRORS = {
    "DB_TIMEOUT": ("database connection timed out after 30s", "claims-ingest"),
    "SCHEMA_INVALID": ("claim payload failed schema validation", "claims-ingest"),
    "DUP_CLAIM": ("duplicate claim id rejected", "claims-ingest"),
    "UPSTREAM_503": ("upstream eligibility service returned 503", "eligibility-api"),
    "PAYMENT_DECLINED": ("payment declined by processor", "payment-gateway"),
    "AUTH_EXPIRED": ("access token expired", "auth-service"),
}

# (service, error code, from, to): inside these windows about half of that service's lines are errors
INCIDENTS = [
    ("claims-ingest", "DB_TIMEOUT", "2026-10-08T02:00:00+00:00", "2026-10-08T02:45:00+00:00"),
    ("eligibility-api", "UPSTREAM_503", "2026-10-11T14:00:00+00:00", "2026-10-11T14:20:00+00:00"),
    ("payment-gateway", "PAYMENT_DECLINED", "2026-10-13T09:00:00+00:00", "2026-10-13T09:30:00+00:00"),
]

# The fields of every log line and their types (keyword = exact value, text = searchable words)
MAPPING = {"properties": {
    "@timestamp": {"type": "date"}, "service": {"type": "keyword"}, "level": {"type": "keyword"},
    "host": {"type": "keyword"}, "endpoint": {"type": "keyword"}, "trace_id": {"type": "keyword"},
    "claim_id": {"type": "keyword"}, "error_code": {"type": "keyword"}, "status": {"type": "integer"},
    "latency_ms": {"type": "float"}, "message": {"type": "text"},
}}


def get_client():
    """Return a client for the local Elasticsearch."""
    return Elasticsearch(ES_URL)


def make_log(rng, when, service, error_code=None):
    """Build one log line. With an error code it is an ERROR; otherwise INFO, or WARN when slow."""
    endpoint = rng.choice(SERVICES[service])
    latency = round(rng.lognormvariate(4.6, 0.6), 1)  # median about 100 ms, with a long tail of slow calls
    log = {
        "@timestamp": when.isoformat(), "service": service, "endpoint": endpoint,
        "host": f"{service}-{rng.randint(1, 4)}", "trace_id": str(uuid.UUID(int=rng.getrandbits(128))),
        "latency_ms": latency,
    }
    if service == "claims-ingest":
        log["claim_id"] = f"CLM-{rng.randint(100000, 999999)}"
    if error_code:
        log.update(level="ERROR", error_code=error_code, status=503 if "503" in error_code else 500,
                   latency_ms=round(latency * 5, 1), message=f"{endpoint} failed: {ERRORS[error_code][0]}")
    elif latency > 400:
        log.update(level="WARN", status=200, message=f"slow response: {latency} ms on {endpoint}")
    else:
        log.update(level="INFO", status=200, message=f"{endpoint} completed in {latency} ms")
    return log


def make_logs(n):
    """Yield n log lines: mostly normal traffic (about 2% errors), plus 3% inside the incident windows."""
    rng = random.Random(11)  # fixed seed: the same logs every run
    seconds = int((END - START).total_seconds())
    own_errors = {s: [c for c, (_, svc) in ERRORS.items() if svc == s] for s in SERVICES}
    in_incidents = int(n * 0.03)
    for i in range(n):
        if i < in_incidents:  # a line inside one of the incident windows
            service, code, a, b = INCIDENTS[i % len(INCIDENTS)]
            a, b = datetime.fromisoformat(a), datetime.fromisoformat(b)
            when = a + timedelta(seconds=rng.uniform(0, (b - a).total_seconds()))
            yield make_log(rng, when, service, code if rng.random() < 0.5 else None)
        else:  # a normal line at a random time
            service = rng.choices(list(SERVICES), BUSY)[0]
            when = START + timedelta(seconds=rng.uniform(0, seconds))
            has_error = own_errors[service] and rng.random() < 0.02
            yield make_log(rng, when, service, rng.choice(own_errors[service]) if has_error else None)


def load_logs(client, n=N_LOGS):
    """Delete and recreate the index, then bulk-load n log lines into it."""
    if client.indices.exists(index=LOG_INDEX):
        client.indices.delete(index=LOG_INDEX)  # start clean, so a rerun never doubles the data
    client.indices.create(index=LOG_INDEX, mappings=MAPPING,
                          settings={"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"})
    actions = ({"_index": LOG_INDEX, "_source": log} for log in make_logs(n))
    for done, _ in enumerate(helpers.streaming_bulk(client, actions, chunk_size=5000), start=1):
        if done % 100_000 == 0:
            print(f"  {done:,} lines loaded")
    client.indices.put_settings(index=LOG_INDEX, settings={"refresh_interval": "1s"})  # make them searchable
    client.indices.refresh(index=LOG_INDEX)
    return client.count(index=LOG_INDEX)["count"]


if __name__ == "__main__":
    print(f"Loaded {load_logs(get_client()):,} lines into '{LOG_INDEX}'")
