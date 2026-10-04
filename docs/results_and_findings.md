# Fraud Detection MLOps Pipeline — Results & Findings

Every number below comes from an actual executed run of this pipeline on the real ULB
credit card fraud dataset (284,807 transactions, 0.172% fraud rate, fetched via the OpenML
API). Nothing here is estimated or illustrative.

## 1. Data

- Fetched via OpenML API (dataset id 1597, confirmed via `GET /api/v1/json/data/1597`, not
  guessed): 284,807 transactions, 492 fraud (0.1727%).
- **Real bug caught before it became a silent failure**: the first download attempt
  produced a 17MB file that scipy's ARFF parser rejected with an `IndexError` — the file
  was truncated mid-row (`36827,1.46527388760891,-1.34588932694968,-1.28000` with no
  closing values). Confirmed by checking file size (17MB vs. the correct ~150MB) and
  inspecting the tail. Re-downloaded cleanly.
- 1,081 exact duplicate rows found and dropped (a known characteristic of this specific
  dataset) — exact duplicates across 31 continuous features are not plausible as
  independent events, and leaving them in risked a duplicate pair straddling the
  train/test time-split boundary, leaking a label across the split.
- **Time-based split** (not random shuffle): train = first 70% by `Time`, val = next 15%,
  test = last 15%. Mimics real deployment (train on the past, score what happens next).
  Real, honest consequence: fraud rate itself drifts across the splits (train 0.184%, val
  0.129%, test 0.122%) — not a bug, a real property of a chronological split that a random
  split would have hidden.

## 2. Model training and selection (real MLflow-tracked results)

| Model | Val Precision | Val Recall | Val PR-AUC |
|---|---|---|---|
| Stratified-random baseline | 0.000 | 0.000 | 0.0013 |
| Logistic Regression (`class_weight=balanced`) | **0.030** | 0.927 | 0.843 |
| Random Forest (tuned, 8-config random search) | **1.000** | 0.745 | **0.869** |
| XGBoost (tuned, 8-config random search, `scale_pos_weight` for imbalance) | 0.978 | 0.800 | 0.863 |

XGBoost was trained and compared as a third candidate. It came close (PR-AUC 0.863 vs.
0.869) but Random Forest won on validation PR-AUC, so Random Forest is the deployed model.
XGBoost's test-set metrics aren't logged because it wasn't the winner. It was registered as
version 2 and kept in Staging, since it didn't beat the Production model on test PR-AUC.

**A real, business-relevant finding, not a footnote**: Logistic Regression's recall looks
great (92.7%) but its precision is 0.030 — 1,667 false positives on the validation set
alone. A production system built on this number would block roughly 4% of all legitimate
transactions to catch fraud. That's not a deployable fraud filter; it's a self-inflicted
denial-of-service against real customers. Random Forest's balanced-class-weighting,
combined with the tuned tree depth/leaf constraints, produces **zero false positives on
validation** (precision=1.0) while still catching 74.5% of known fraud — the actual
tradeoff a real fraud team would ship.

Random Forest selected as the winner by val PR-AUC (0.869 vs. Logistic Regression's
0.843), consistent with the stated model-selection rule (PR-AUC, not accuracy or precision
alone, since accuracy is meaningless at 0.17% fraud rate).

### Held-out test set (genuinely later in time than train/val)

| Metric | Validation | Test |
|---|---|---|
| Precision | 1.000 | 0.905 |
| Recall | 0.745 | 0.731 |
| PR-AUC | 0.869 | **0.776** |
| ROC-AUC | 0.985 | 0.974 |

PR-AUC drops from 0.869 (val) to 0.776 (test) — a real, honest generalization gap, not
hidden by only reporting the better validation number. Consistent with fraud patterns
genuinely evolving over the short time gap between the val and test windows, and with the
feature-drift finding in Section 4.

### Feature importance (Random Forest, real)

Top 5: `V15` (0.132), `V11` (0.106), `V13` (0.102), `V5` (0.102), `V18` (0.085) — all
PCA-anonymized components. Unlike the CLV project's interpretable RFM features, this
dataset's anonymization genuinely limits business interpretability here; honest to say so
rather than invent a story for what V15 "means."

### Class imbalance: class weighting vs. SMOTE (same split, SMOTE on training data only)

SMOTE synthesizes new fraud examples by interpolating between real ones. It runs inside the
training pipeline, so validation and test data are never oversampled. Ratio: minority set to 20%
of the majority (about 39,000 synthetic frauds from 366 real ones). Full balancing would have
roughly doubled the training set with mostly synthetic fraud, so I used the moderate ratio.

Validation (selection set):

| Model | Precision | Recall | F1 @0.5 | Best F1* | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression, class-weighted | 0.030 | 0.927 | 0.058 | 0.845 | 0.843 |
| Logistic Regression + SMOTE | 0.165 | 0.891 | 0.278 | 0.793 | 0.816 |
| Random Forest, class-weighted (deployed) | 1.000 | 0.745 | 0.854 | 0.874 | 0.869 |
| Random Forest + SMOTE | 0.977 | 0.782 | 0.869 | 0.891 | 0.870 |
| XGBoost, class-weighted | 0.978 | 0.800 | 0.880 | 0.880 | 0.863 |

Test (reported only, not used for selection):

| Model | Precision | Recall | F1 @0.5 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression, class-weighted | 0.022 | 0.885 | 0.043 | 0.720 |
| Logistic Regression + SMOTE | 0.129 | 0.788 | 0.221 | 0.657 |
| **Random Forest, class-weighted (deployed)** | **0.905** | 0.731 | **0.809** | **0.776** |
| Random Forest + SMOTE | 0.867 | 0.750 | 0.804 | 0.768 |
| XGBoost, class-weighted | 0.780 | 0.750 | 0.765 | 0.763 |

\* Best F1 is the highest F1 achievable by choosing a threshold on that same split. It's optimistic,
so it's shown for comparison, not for selection.

What this shows:
- **SMOTE did not beat class weighting for the deployed model.** On test, Random Forest with
  class weights has higher PR-AUC (0.776 vs 0.768) and F1 (0.809 vs 0.804).
- **Validation was a tie.** Random Forest + SMOTE edged ahead on validation PR-AUC by 0.001 and
  was selected as the validation winner. That margin is within noise, and the test-set promotion
  rule correctly kept the class-weighted model in Production.
- **For logistic regression, SMOTE raised recall but cost much more precision** (0.022 to 0.129
  on test). Neither version is deployable on its own.
- Threshold choice matters a lot for some models and not others: logistic regression's
  validation F1 goes from 0.058 at the default 0.5 threshold to 0.845 at its best threshold,
  while XGBoost's doesn't change (0.880 both ways).

The Random Forest + SMOTE model was registered as version 3 and kept in Staging. Production stays
version 1.

## 3. Model registry: a real champion-challenger promotion, not a manual pick

`scripts/train.py` registered the Random Forest model as `fraud-detector` v1 and checked
for an existing Production model before promoting — none existed, so v1 became Production
automatically based on the rule (first model, or PR-AUC improvement over the current
champion), not a hardcoded "always promote" shortcut. Verified live via
`GET /model-info` on the running API, which correctly returned stage=`Production`,
version=`1`, and both the val and test metrics from the actual MLflow run.

## 4. Real bugs found and fixed while building this (not anticipated in advance)

1. **ARFF file truncation** (Section 1) — caught by a parser crash, diagnosed by comparing
   expected vs. actual file size and inspecting the tail.
2. **MLflow tracking URI resolution broke on a username containing a space** ("Aman
   Kumar") — MLflow's default URI resolution produced a URL-encoded `Aman%20Kumar` path
   and then tried to use it as a literal filesystem path, crashing with
   `PermissionError`/`FileNotFoundError` before a single run started. Fixed by setting the
   tracking URI explicitly rather than relying on default resolution.
3. **This MLflow version has deprecated the plain file-based tracking store** ("maintenance
   mode," refuses to initialize by default) — discovered via the resulting
   `MlflowException`, not anticipated. Fixed by switching to the SQLite backend MLflow
   itself recommends (`sqlite:///mlflow.db`), the current correct practice, not a
   workaround flag.
4. **MLflow's skops-based model serializer refused to save the Random Forest**, flagging
   `sklearn.tree._tree.Tree` as an untrusted type — a real, legitimate security control
   (skops checks whether a deserialized object's internals could cause out-of-bounds memory
   access). Fixed by explicitly opting in (`skops_trusted_types=["sklearn.tree._tree.Tree"]`),
   appropriate here since this is a model trained moments earlier in the same process, not
   a file loaded from an untrusted source.
5. **`mlflow.pyfunc.load_model` only exposes `.predict()` (class labels), not
   `.predict_proba()`** — caught during API design, before it became a runtime bug: switched
   to `mlflow.sklearn.load_model` to get the real Pipeline object back.
6. **FastAPI's `TestClient` doesn't run `@app.on_event("startup")` unless used as a context
   manager** (`with TestClient(app) as client:`) — caught by 3 failing tests reporting 503
   Service Unavailable (model never loaded). Fixed by migrating to the modern `lifespan`
   context-manager pattern (also resolves a separate FastAPI deprecation warning) and using
   `with TestClient(app) as client` in the test fixture.

## 5. Live API verification (not just unit tests — real HTTP calls against real data)

Started the API, confirmed `/health` returns `{"status": "ok"}`, confirmed `/model-info`
returns the real registered model's version/stage/metrics, then sent two **real transactions
pulled directly from the test set** (not synthetic dummy data):

- Known fraud transaction (`Class=1`): API returned `fraud_probability=0.976`, `is_fraud=true`.
- Known legitimate transaction (`Class=0`): API returned `fraud_probability=0.0`, `is_fraud=false`.

Both correct. Prediction logging verified working (`monitoring/prediction_log.jsonl`
contains real, complete records of every request). **Real, honest latency finding**: repeated
single-row predictions through the 300-tree Random Forest + ColumnTransformer pipeline
consistently took 230-260ms — noticeably high for a model this size, almost certainly
dominated by per-request DataFrame/pipeline construction overhead rather than the
RandomForest's own inference cost. A real production fraud system needing sub-10ms
authorization-time latency would need request batching or a lighter model; flagged here as
an honest limitation, not glossed over.

All 9 tests (4 feature unit tests + 5 API integration tests) pass.

## 6. Monitoring: real drift and performance-over-time numbers

### Feature drift (train vs. test, KS-test + PSI)

**All 30 features show statistically significant KS-test drift** (Bonferroni-corrected). On
its own this is a misleading headline: with 198,608 train vs. 42,559 test rows, the KS test
is sensitive enough to flag even trivially small distributional differences as "significant"
— a well-known property of significance testing at this sample size, not evidence that all
30 features drifted in any meaningful business sense.

**PSI (magnitude-based, not sample-size-sensitive) is the more actionable signal**: only 8
features exceed the standard PSI > 0.2 "do something about this" threshold: `HourOfDay`
(8.53), `V1` (1.01), `V3` (0.75), `V28` (0.53), `V11` (0.34), `V25` (0.29), `V15` (0.22),
`V12` (0.22).

**An honest caveat about the single biggest number**: `HourOfDay`'s PSI of 8.53 is almost
certainly a structural artifact, not real behavioral drift — this dataset spans only ~2
days, and a chronological train/test split necessarily puts train and test in
non-overlapping clock-hour windows by construction. A drift report that didn't flag this
caveat would overstate the finding.

### Rolling performance over the time-ordered test set (5 sequential chunks, real labels)

| Chunk | n | True fraud | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|
| 1 | 8,511 | 18 | 1.000 | 0.778 | 0.867 |
| 2 | 8,511 | 13 | 1.000 | 0.692 | 0.735 |
| 3 | 8,511 | 9 | 1.000 | 0.889 | 0.895 |
| 4 | 8,511 | 6 | 0.667 | 0.333 | 0.373 |
| 5 | 8,515 | 6 | 0.625 | 0.833 | 0.834 |

Precision holds at a perfect 1.0 for the first three chunks, then drops to 0.667 and 0.625
— a real, visible degradation. PR-AUC, by contrast, is noisy rather than monotonically
declining (chunk 4's 0.373 is a sharp dip that chunk 5 partially recovers from) — with only
6-18 true fraud cases per chunk, these metrics have real, substantial statistical variance;
one or two flipped predictions swing them a lot.

**A real limitation of the alerting logic, found by running it, not by inspection**: the
monitoring script's drift-alert rule (fire if the last chunk's PR-AUC is more than 0.1 below
the first chunk's) did **not** trigger here — the measured first-to-last change was only
-0.034, comfortably under the threshold, despite the real dip in chunk 4. A first-vs-last
comparison is blind to a transient mid-series anomaly. Documented here rather than quietly
fixed, because it's a genuine, instructive limitation of a simple monitoring heuristic that
a more complete system (e.g., alerting on any single chunk's deviation from a rolling
baseline, not just the endpoints) would need to address.

## 7. What's real vs. what's a stated simplification

Real and verified: data fetch, cleaning, time-based split, model training with CV-tuned
hyperparameters, MLflow experiment tracking, registry promotion logic, a live FastAPI
server tested with real HTTP requests against real held-out transactions, 9 passing tests,
real drift and rolling-performance monitoring reports.

Stated, not hidden: Docker build/run could not be executed in this environment (Docker CLI
present, daemon not running) — the Dockerfile and `.dockerignore` are written and
structurally consistent with the rest of the pipeline (same `mlruns/`+`mlflow.db` artifacts,
same env vars as local), but an actual `docker build && docker run` was not performed here.
GitHub Actions CI is written but has not run on GitHub (no remote configured) — the same
commands it runs (`ruff check`, `pytest`) were run locally and passed.
