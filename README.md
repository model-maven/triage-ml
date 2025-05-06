# TriageML

**Production-grade NLP service that automatically classifies customer support
tickets** — so they get routed to the right team in milliseconds instead of
waiting in a manual queue.

[![CI](https://github.com/AmirEmami/triage-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/AmirEmami/triage-ml/actions)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange)

This is not a notebook. It is an end-to-end system: a trained model, a REST API
that serves it, a container that ships it, a CI pipeline that tests it, and
monitoring that watches it in production.

---

## Why it exists

Support teams drown in unrouted tickets. TriageML classifies each incoming
message into one of five buckets — `billing`, `technical`, `account_access`,
`feature_request`, `general_inquiry` — and **flags low-confidence predictions for
human review** so the system never confidently mis-routes an ambiguous ticket.

## What's inside

| Capability | How it's done |
|---|---|
| **Baseline model** | TF-IDF + Logistic Regression (`scikit-learn`) — fast, interpretable, sets the bar |
| **Deep model** | Fine-tuned DistilBERT (`PyTorch` + `transformers`) behind an optional extras group |
| **Serving** | `FastAPI` with `/predict`, `/predict/batch`, `/health`, `/metrics` |
| **Packaging** | Multi-stage `Dockerfile` (non-root, healthcheck) + `docker-compose` |
| **CI/CD** | GitHub Actions: lint → test on 3 Python versions → build & smoke-test the image |
| **Monitoring** | Prediction logging + PSI drift detection that exits non-zero to alert |
| **Config** | 12-factor env-var config via `pydantic-settings` |

## Results

The bundled (offline) dataset includes realistic annotator noise, so the score
reflects genuine generalisation rather than memorised templates:

| Model | Macro-F1 | Notes |
|---|---|---|
| TF-IDF + Logistic Regression | **~0.91** | trains in <0.1s on CPU |
| DistilBERT (fine-tuned) | run `make train-transformer` | GPU recommended |

## Quickstart

```bash
# 1. install
make install            # pip install -e ".[dev]"

# 2. generate data + train the baseline
make data
make train              # writes artifacts/baseline.joblib  (~0.1s)

# 3. serve it
make serve              # http://localhost:8000/docs
```

Then classify a ticket:

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "I was charged twice this month, please refund the duplicate."}'
```

```json
{
  "label": "billing",
  "confidence": 0.87,
  "needs_review": false,
  "scores": { "billing": 0.87, "account_access": 0.05, "...": "..." }
}
```

### Run it in Docker

```bash
docker compose up --build      # ships with the model pre-trained at build time
```

### Check for drift

```bash
python scripts/check_drift.py  # compares live traffic vs training distribution
```

## Using real data

The bundled dataset keeps the repo self-contained. To train on a real public
benchmark (e.g. **Banking77** intent classification), implement `load_huggingface`
in `src/triageml/data.py` to return the same `(text, label)` columns — nothing
downstream changes.

## Project layout

```
src/triageml/
├── config.py            # env-var settings
├── data.py              # load + stratified split
├── train_baseline.py    # scikit-learn pipeline
├── train_transformer.py # DistilBERT fine-tuning (optional extras)
├── predict.py           # unified Predictor over both backends
├── monitoring.py        # prediction logging + PSI drift
└── api/
    ├── main.py          # FastAPI app
    └── schemas.py       # request/response models
tests/                   # data, model, and API tests
scripts/                 # dataset generation + drift check
docs/architecture.md     # design write-up + diagram
```

See [`docs/architecture.md`](docs/architecture.md) for the design rationale and
cloud-deployment notes (AWS / GCP / Azure).

## Tech stack

`Python` · `scikit-learn` · `PyTorch` · `Hugging Face Transformers` · `FastAPI` ·
`Docker` · `GitHub Actions` · `pytest` · `ruff`

## License

MIT © Amir Emami
