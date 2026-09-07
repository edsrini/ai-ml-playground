# Reproducible ML Pipeline

Hands-on implementation of a reproducible machine learning workflow — the practices that let a team trust that a model in production can be traced back to the exact data, code, and parameters that produced it.

## Skills demonstrated

- **Artifact versioning** — logging datasets, cleaned data, and trained models as versioned, hash-deduplicated artifacts (Weights & Biases) instead of untracked local files
- **Data validation** — automated `pytest` checks (schema, distribution, Kolmogorov-Smirnov drift tests) gating a pipeline before training runs on it
- **Experiment tracking** — comparing runs, hyperparameters, and metrics across a `scikit-learn` random forest pipeline with a TF-IDF + numeric/categorical feature pipeline
- **Configuration management** — `Hydra`/`OmegaConf`-driven multi-step pipelines with hyperparameter sweeps, instead of hardcoded scripts
- **Pipeline orchestration & reproducibility** — chaining independent MLflow project steps (download → preprocess → validate → segregate → train → evaluate) into one reproducible, re-runnable pipeline
- **Containerized environments** — Docker Compose setup for a JupyterLab + self-hosted MLflow tracking server, so the environment itself is reproducible, not just the code

## Structure

The code is organized by lesson, progressing from a single-step artifact upload through to a fully assembled, released pipeline:

| Lesson | Focus |
|---|---|
| `lesson-1-machine-learning-pipelines` | MLflow projects, artifact versioning, chaining pipeline steps |
| `lesson-2-data-exploration-and-preparation` | EDA, automated data profiling, cleaning/preprocessing |
| `lesson-3-data-validation` | Data quality tests with `pytest` and fixtures |
| `lesson-4-training-validation-experiment-tracking` | Model training pipelines, Hydra sweeps, W&B experiment tracking |
| `lesson-5-final-pipeline-release-and-deploy` | End-to-end pipeline assembly, tagged releases, deployment |

Each lesson contains `demo/` (worked examples) and `exercises/exercise_N/` (a `README.md`, plus `starter/` and `solution/` implementations).

## Stack

MLflow · Weights & Biases · Hydra · scikit-learn · PyTorch · pandas · pytest · Docker

## Running it

```bash
docker compose up
```

Spins up a JupyterLab environment and a local MLflow tracking server. See each exercise's own `README.md` for exercise-specific instructions, and `CONDA.md` / `requirements.txt` for environment setup details.

---

**Srinivasan E** · [LinkedIn](https://www.linkedin.com/in/esrinivasan/) · [GitHub](https://github.com/edsrini)
