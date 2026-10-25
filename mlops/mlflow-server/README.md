# MLflow server (standalone)

A local MLflow tracking server in Docker, from the official `ghcr.io/mlflow/mlflow` image. Independent of
`reproducible-ml-pipeline`: nothing to build, and no Miniconda.

```bash
docker compose up -d          # first run downloads the image
docker compose ps             # wait for "healthy"
```

Open http://localhost:5000. Point a client at it:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```

- **Data** (runs, metrics, artifacts) is in `./mlruns`, next to the compose file, and is git-ignored.
- **Stop** with `docker compose down`. The data stays in `./mlruns`; delete that folder to start fresh.
- The port is bound to `127.0.0.1` only, because the server has no authentication.
- The course's MLflow *projects* (conda environments) run from the client side, with `mlflow run`; they do not need
  conda inside this container.
