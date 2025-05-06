# Architecture

TriageML is built around a clean separation between **training**, **inference**,
**serving**, and **monitoring**, so any one piece can change without touching the
others.

```
                         ┌─────────────────────────┐
                         │        Data layer        │
                         │  data.py  (CSV / HF)     │
                         └────────────┬────────────┘
                                      │ train/test split
                ┌─────────────────────┴─────────────────────┐
                ▼                                             ▼
   ┌────────────────────────┐                   ┌────────────────────────┐
   │  Baseline trainer       │                   │  Transformer trainer    │
   │  TF-IDF + LogReg        │                   │  DistilBERT (PyTorch)   │
   │  train_baseline.py      │                   │  train_transformer.py   │
   └───────────┬─────────────┘                   └───────────┬────────────┘
               │ baseline.joblib                             │ transformer/
               └──────────────────┬──────────────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │      Predictor           │  unified interface;
                     │      predict.py          │  backend chosen by config
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐        ┌──────────────────┐
                     │   FastAPI service        │ ─────▶ │   monitoring.py   │
                     │   api/main.py            │  logs  │  JSONL + PSI drift │
                     │  /health /predict /metrics│        └──────────────────┘
                     └─────────────────────────┘
```

## Design decisions

**Two models, one interface.** A TF-IDF + Logistic Regression baseline is trained
first. It is fast, interpretable, and sets the bar the transformer must beat. The
`Predictor` exposes an identical API for both, so promoting the transformer to
production is a single config flag (`TRIAGEML_MODEL_BACKEND=transformer`) — no API
code changes.

**Optional heavy dependencies.** `torch`/`transformers` live behind an extras
group and are imported lazily. The baseline, API, and CI run on a lean dependency
set; only transformer training pulls the deep-learning stack.

**Confidence-gated routing.** Predictions below a configurable confidence
threshold are flagged `needs_review`, mirroring how a real triage system escalates
uncertain tickets to a human instead of mis-routing them.

**Monitoring is first-class.** Every prediction is logged to JSONL for auditing
and to build a feedback-labelled dataset over time. A Population Stability Index
(PSI) check compares live label distribution against training, giving an early,
quantitative retraining signal.

## Cloud deployment

The container is provider-agnostic. Typical targets:

- **AWS** — push to ECR, run on ECS Fargate or App Runner behind an ALB.
- **GCP** — push to Artifact Registry, deploy to Cloud Run (scales to zero).
- **Azure** — push to ACR, deploy to Azure Container Apps.

For managed training/registry you can substitute SageMaker, Vertex AI, or Azure
ML, swapping only `train_*.py` artifact paths — the serving layer is unchanged.
