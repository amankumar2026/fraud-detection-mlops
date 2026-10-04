# Fraud Detection — MLOps Pipeline

A full MLOps loop — experiment tracking, model registry with real
champion-challenger promotion, API serving, containerization, CI, and
post-deployment monitoring — built around credit card fraud detection on the
real ULB dataset (284,807 transactions, 0.172% fraud rate, fetched via the
OpenML API). The model itself is deliberately simple; the pipeline around it
is the point. Full design rationale: [docs/architecture.md](docs/architecture.md).
Every real number, 6 real bugs found and fixed, and live end-to-end API
verification: [docs/results_and_findings.md](docs/results_and_findings.md).

## Headline real results

- **Random Forest selected over Logistic Regression** by val PR-AUC (0.869 vs. 0.843) —
  and for good reason: Logistic Regression's "balanced" recall (92.7%) comes with
  precision of just 0.030 (1,667 false positives on validation alone), which would block
  ~4% of legitimate transactions in production. Random Forest: precision=1.000,
  recall=0.745 on validation.
- **Test-set PR-AUC (0.776) is honestly lower than validation (0.869)** — reported as a
  real generalization gap, not hidden by only showing the better number.
- Live API verified against real held-out transactions: 97.6% fraud probability on a known
  fraud case, 0% on a known legitimate case.
- Real drift report: all 30 features show statistically-significant KS-test drift (largely
  a large-sample-size artifact), but only 8 exceed the more meaningful PSI > 0.2 threshold
  — and the biggest of those (`HourOfDay`) is flagged as likely a structural artifact of
  this dataset's 2-day span, not real behavioral drift.

## Stack

MLflow (experiment tracking + model registry) · scikit-learn · FastAPI ·
Docker · GitHub Actions · pytest

## Reproduce end to end

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts\load_data.py       # fetch + parse the real dataset
python scripts\split_data.py      # time-based train/val/test split
python scripts\train.py           # trains, logs to MLflow, registers + promotes the winner

uvicorn api.main:app --reload     # serving layer, loads the Production model
# in another terminal:
curl http://localhost:8000/health
curl http://localhost:8000/model-info

pytest tests/ -v                  # unit + API tests

docker build -t fraud-detector-api .
docker run -p 8000:8000 fraud-detector-api

python monitoring\drift_report.py            # real KS-test + PSI drift report
python monitoring\performance_over_time.py   # rolling precision/recall on time-ordered test data
```

View experiment runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db` (from the project root).

## Repository layout

```
src/              feature engineering (shared by training and serving)
scripts/          load_data, split_data, train (MLflow tracking + registry promotion)
api/              FastAPI app -- loads the Production model, /predict /health /model-info
monitoring/        drift_report.py (KS-test + PSI), performance_over_time.py
tests/            unit tests (features) + API integration tests
.github/workflows/ CI: lint, test, Docker build
docs/             architecture.md (design) + results_and_findings.md (real results, bugs found/fixed)
mlruns/, mlflow.db  MLflow artifact store + SQLite tracking/registry (committed -- see docs/architecture.md)
```
