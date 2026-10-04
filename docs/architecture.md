# Credit Card Fraud Detection — MLOps Pipeline

## Business problem

A payments processor needs to flag fraudulent transactions automatically. Two failure
modes have different costs: a missed fraud (false negative) is a direct financial loss; a
false alarm on a legitimate transaction (false positive) blocks a real customer and costs
goodwill. Accuracy is a meaningless metric here — fraud is 0.172% of transactions, so a
model that predicts "never fraud" is 99.8% accurate and catches nothing. The real metrics
are precision, recall, and PR-AUC (not ROC-AUC, which is also overly optimistic under this
much imbalance).

## Dataset

Real ULB Credit Card Fraud dataset (Dal Pozzolo et al., 2015), fetched via the OpenML API
(dataset id 1597 — confirmed via `GET /api/v1/json/data/1597`, not guessed): 284,807 card
transactions made by European cardholders over two days in September 2013, 492 labeled as
fraud (0.172%). Features `V1`-`V28` are PCA components of the original (confidential)
transaction features; `Time` is seconds since the first transaction; `Amount` is the
transaction value; `Class` is the target (1 = fraud).

## Why this project, and what it's actually demonstrating

Not a model-complexity showcase — the point of this project is the production pipeline
around a deliberately simple model, aimed at what data scientist/analyst interviews
actually probe: can you track experiments reproducibly, version a model with a real
promotion rule (not just "pick the best number by eye"), serve it behind a real API, test
it, containerize it, and monitor it for drift after deployment. Each of those is a real,
separately verifiable piece below.

## Pipeline

```
Raw data (OpenML, fetched via API)
        |
        v
Data validation + time-based split   -- train/val/test split by transaction Time, not
        |                                random shuffling: mimics real deployment (train on
        |                                the past, evaluate on transactions that happen
        |                                after training data ends), and avoids a subtle
        |                                leakage risk a random split would have if nearby
        |                                transactions share structure.
        v
Feature engineering                  -- RobustScaler for Amount (heavy-tailed, real
        |                                outliers matter for fraud), an engineered
        |                                HourOfDay feature from Time (fraud timing patterns
        |                                are a real, known signal in this literature).
        v
Model training + MLflow tracking     -- Logistic Regression baseline, Random Forest
        |                                (tuned). Every run logs params/metrics/artifacts
        |                                to a local MLflow tracking store.
        v
MLflow Model Registry                -- champion-challenger promotion: a new model is
        |                                promoted to "Production" only if its PR-AUC beats
        |                                the current Production model's, not by manual pick.
        v
FastAPI serving layer                -- loads the Production-stage model, /predict,
        |                                /health, /model-info; every prediction logged
        |                                (structured, timestamped) for monitoring.
        v
Docker                               -- containerized API, built and run locally.
        |
        v
CI (GitHub Actions)                  -- lint + unit tests + API tests + Docker build,
        |                                on every push.
        v
Monitoring                           -- real KS-test-based feature drift report comparing
                                         a held-out "recent" batch against the training
                                         distribution; rolling-window precision/recall on
                                         the time-ordered test set to show performance isn't
                                         assumed constant after deployment.
```

## A deliberate deviation from real-world practice, stated plainly

`mlruns/` (the MLflow tracking store and model registry) is committed to this repo. In a
real deployment this would live in a remote MLflow server or object store, never in git —
committing it here is purely so this portfolio project's Docker image and API are
reproducible directly from a clone, without requiring `scripts/train.py` to be re-run
first just to get a working API demo. Called out explicitly so it doesn't read as an
accidental best-practice violation.

## What this project deliberately does not include

No orchestrator (Airflow/Prefect) and no live cloud deployment — explicitly scoped out in
favor of depth on the core experiment-tracking/registry/serving/monitoring loop, which is
what DS/analyst-adjacent MLOps questions in interviews actually focus on.
