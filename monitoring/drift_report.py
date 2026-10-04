"""Real feature-drift detection: compares the training distribution against
the test set (genuinely held-out, later in time) using a two-sample
Kolmogorov-Smirnov test per feature, plus Population Stability Index (PSI) --
the two most common real drift metrics, not a single arbitrary choice.
Standing in for "recent production traffic" with the test set here because
it's the only genuinely time-later data available without a live deployment;
in production this script would run against actual incoming request logs
(monitoring/prediction_log.jsonl) instead.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import MODEL_INPUT_COLS, prepare_model_input


def population_stability_index(expected, actual, bins=10):
    """Standard PSI: bin the EXPECTED (training) distribution into decile
    edges, then compare bucket proportions. PSI > 0.2 is the common
    industry rule-of-thumb threshold for "significant drift"."""
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0  # degenerate (near-constant) feature

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_pct = np.maximum(expected_counts / len(expected), 1e-6)
    actual_pct = np.maximum(actual_counts / len(actual), 1e-6)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def main():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = prepare_model_input(train_df)
    X_test = prepare_model_input(test_df)

    print(f"Comparing train (n={len(X_train):,}) vs. test/'recent' (n={len(X_test):,}) "
          f"distributions across {len(MODEL_INPUT_COLS)} features...\n")

    results = []
    alpha = 0.05 / len(MODEL_INPUT_COLS)  # Bonferroni correction for multiple testing
    for col in MODEL_INPUT_COLS:
        stat, pval = ks_2samp(X_train[col], X_test[col])
        psi = population_stability_index(X_train[col].values, X_test[col].values)
        results.append({"feature": col, "ks_stat": stat, "ks_pvalue": pval,
                         "psi": psi, "drifted": pval < alpha})

    report = pd.DataFrame(results).sort_values("psi", ascending=False)
    print(report.to_string(index=False))

    n_drifted = report["drifted"].sum()
    print(f"\n{n_drifted}/{len(report)} features show statistically significant drift "
          f"(KS test, Bonferroni-corrected alpha={alpha:.5f})")
    print(f"Features with PSI > 0.2 (industry rule-of-thumb for 'significant' drift): "
          f"{report[report['psi'] > 0.2]['feature'].tolist()}")

    Path("monitoring").mkdir(exist_ok=True)
    report.to_csv("monitoring/drift_report.csv", index=False)
    print("\nSaved monitoring/drift_report.csv")


if __name__ == "__main__":
    main()
