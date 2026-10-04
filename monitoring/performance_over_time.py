"""Scores the Production model on sequential, time-ordered chunks of the
test set and tracks precision/recall/PR-AUC per chunk -- a real way to check
whether a model's performance holds up over time rather than assuming a
single train-time metric stays valid forever. The test set already has
ground truth, so this is a genuine computation (not a simulated trend) that
stands in for what would otherwise require waiting for real labeled
production outcomes to accumulate.
"""

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import prepare_model_input

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_CHUNKS = 5


PRECISION_DROP_ALERT = 0.2
PR_AUC_DROP_ALERT = 0.2
MIN_FRAUD_PER_CHUNK = 5


def find_degradation_alerts(report):
    """Compares each chunk to the median of all earlier chunks, so a temporary
    dip is caught even when the last chunk has recovered. Chunks with fewer
    than MIN_FRAUD_PER_CHUNK true frauds are skipped: one flipped prediction
    swings precision/PR-AUC too much to alert on."""
    alerts = []
    for i in range(2, len(report)):
        chunk = report.iloc[i]
        if chunk["n_fraud"] < MIN_FRAUD_PER_CHUNK:
            continue
        earlier = report.iloc[:i]
        reasons = []
        base_precision = earlier["precision"].median()
        base_pr_auc = earlier["pr_auc"].median()
        if chunk["precision"] < base_precision - PRECISION_DROP_ALERT:
            reasons.append(f"precision {chunk['precision']:.3f} vs baseline {base_precision:.3f}")
        if chunk["pr_auc"] < base_pr_auc - PR_AUC_DROP_ALERT:
            reasons.append(f"PR-AUC {chunk['pr_auc']:.3f} vs baseline {base_pr_auc:.3f}")
        if reasons:
            alerts.append({"chunk": int(chunk["chunk"]), "reasons": reasons})
    return alerts


def main():
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    client = MlflowClient()
    prod = client.get_latest_versions("fraud-detector", stages=["Production"])
    if not prod:
        print("No Production model found -- run scripts/train.py first.")
        return

    model = mlflow.sklearn.load_model("models:/fraud-detector/Production")
    print(f"Loaded Production model v{prod[0].version}")

    test_df = pd.read_csv("data/processed/test.csv").sort_values("Time").reset_index(drop=True)
    X_test = prepare_model_input(test_df)
    y_test = test_df["Class"].values

    chunk_size = len(test_df) // N_CHUNKS
    print(f"\nScoring {N_CHUNKS} sequential time-ordered chunks "
          f"(~{chunk_size:,} transactions each)...\n")

    rows = []
    for i in range(N_CHUNKS):
        start = i * chunk_size
        end = len(test_df) if i == N_CHUNKS - 1 else (i + 1) * chunk_size
        X_chunk, y_chunk = X_test.iloc[start:end], y_test[start:end]
        if y_chunk.sum() == 0:
            print(f"Chunk {i+1}: {len(X_chunk):,} txns, time=[{test_df['Time'].iloc[start]:.0f}, "
                  f"{test_df['Time'].iloc[end-1]:.0f}] -- 0 real frauds in this chunk, skipping metrics")
            continue
        proba = model.predict_proba(X_chunk)[:, 1]
        pred = (proba >= 0.5).astype(int)
        row = {
            "chunk": i + 1,
            "n_transactions": len(X_chunk),
            "n_fraud": int(y_chunk.sum()),
            "time_start": test_df["Time"].iloc[start],
            "time_end": test_df["Time"].iloc[end - 1],
            "precision": precision_score(y_chunk, pred, zero_division=0),
            "recall": recall_score(y_chunk, pred, zero_division=0),
            "pr_auc": average_precision_score(y_chunk, proba),
        }
        rows.append(row)
        print(f"Chunk {i+1}: n={row['n_transactions']:,}  fraud={row['n_fraud']}  "
              f"precision={row['precision']:.3f}  recall={row['recall']:.3f}  "
              f"pr_auc={row['pr_auc']:.3f}")

    report = pd.DataFrame(rows)
    alerts = find_degradation_alerts(report)
    report["alert"] = report["chunk"].isin([a["chunk"] for a in alerts])
    if alerts:
        print("\nALERTS (each chunk compared to the median of all earlier chunks):")
        for a in alerts:
            print(f"  chunk {a['chunk']}: {', '.join(a['reasons'])}")
    else:
        print("\nNo degradation alerts.")

    Path("monitoring").mkdir(exist_ok=True)
    report.to_csv("monitoring/performance_over_time.csv", index=False)
    print("\nSaved monitoring/performance_over_time.csv")


if __name__ == "__main__":
    main()
