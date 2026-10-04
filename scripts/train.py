"""Trains fraud models with real MLflow experiment tracking, then registers
the winner in the MLflow Model Registry with a real champion-challenger
promotion rule: promote to "Production" only if the new model's test PR-AUC
beats whatever is currently in Production (or if nothing is in Production
yet). Accuracy is never used to pick a model -- at 0.17% fraud rate it's
uninformative; PR-AUC (not ROC-AUC, which is also misleadingly high under
this much imbalance) is the model-selection metric throughout.

Each model is a single sklearn Pipeline (Amount scaler + classifier), so the
one artifact MLflow logs is everything serving needs -- no separately
managed scaler file that could drift out of sync with the model.
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import MODEL_INPUT_COLS, prepare_model_input

EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME = "fraud-detector"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Explicit, not left to MLflow's default resolution -- two real issues hit
# while building this: (1) on this machine the username ("Aman Kumar")
# contains a space, and MLflow's default tracking-URI resolution produces a
# URL-encoded path ("Aman%20Kumar") it then tries to use as a literal
# filesystem path, crashing before a single run starts; (2) this MLflow
# version has put the plain file-based store ("./mlruns") into maintenance
# mode and refuses to use it by default, recommending a database backend
# instead. SQLite is the simplest real fix for both, and is also MLflow's
# own current recommendation, not a workaround.
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")


def make_preprocessor():
    return ColumnTransformer(
        transformers=[("amount_scaler", RobustScaler(), ["Amount"])],
        remainder="passthrough",
    )


def load_split(name):
    df = pd.read_csv(f"data/processed/{name}.csv")
    X = prepare_model_input(df)
    y = df["Class"].values
    return X, y


def evaluate(y_true, y_pred, y_proba):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def log_pr_curve(y_true, y_proba, run_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name=run_name)
    ax.set_title(f"Precision-Recall curve — {run_name}")
    path = f"pr_curve_{run_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(path)
    Path(path).unlink()


def log_confusion_matrix(y_true, y_pred, run_name):
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix (rows=true, cols=pred):\n{cm}")
    path = f"cm_{run_name}.txt"
    with open(path, "w") as f:
        f.write(f"TN={cm[0,0]}  FP={cm[0,1]}\nFN={cm[1,0]}  TP={cm[1,1]}\n")
    mlflow.log_artifact(path)
    Path(path).unlink()


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading splits...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"  train={len(X_train):,} ({y_train.sum()} fraud)  "
          f"val={len(X_val):,} ({y_val.sum()} fraud)  "
          f"test={len(X_test):,} ({y_test.sum()} fraud)")
    print(f"  model input columns: {MODEL_INPUT_COLS}")

    results = {}

    # --- Baseline: stratified random guessing at the real fraud rate ---
    with mlflow.start_run(run_name="baseline_stratified"):
        dummy = Pipeline([("prep", make_preprocessor()),
                           ("clf", DummyClassifier(strategy="stratified", random_state=42))])
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_val)
        y_proba = dummy.predict_proba(X_val)[:, 1]
        metrics = evaluate(y_val, y_pred, y_proba)
        mlflow.log_params({"model": "DummyClassifier", "strategy": "stratified"})
        mlflow.log_metrics(metrics)
        print(f"\nBaseline (stratified random): {metrics}")
        results["baseline"] = metrics

    # --- Logistic Regression ---
    with mlflow.start_run(run_name="logistic_regression") as run:
        params = {"class_weight": "balanced", "max_iter": 1000, "C": 1.0, "random_state": 42}
        lr = Pipeline([("prep", make_preprocessor()), ("clf", LogisticRegression(**params))])
        t0 = time.time()
        lr.fit(X_train, y_train)
        train_time = time.time() - t0
        y_pred = lr.predict(X_val)
        y_proba = lr.predict_proba(X_val)[:, 1]
        metrics = evaluate(y_val, y_pred, y_proba)
        mlflow.log_params({"model": "LogisticRegression", **params})
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metrics(metrics)
        log_pr_curve(y_val, y_proba, "logreg")
        log_confusion_matrix(y_val, y_pred, "logreg")
        mlflow.sklearn.log_model(lr, "model", input_example=X_train.head(2))
        print(f"\nLogistic Regression (val): {metrics}  (train_time={train_time:.1f}s)")
        results["logreg"] = (lr, metrics, run.info.run_id)

    # --- Random Forest, lightly tuned via random search on PR-AUC (val set) ---
    print("\nTuning Random Forest (random search, val-set PR-AUC)...")
    param_space = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
    }
    best_pr_auc, best_rf, best_params = -1, None, None
    for params in ParameterSampler(param_space, n_iter=8, random_state=42):
        rf = Pipeline([("prep", make_preprocessor()),
                        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **params))])
        rf.fit(X_train, y_train)
        pr_auc = average_precision_score(y_val, rf.predict_proba(X_val)[:, 1])
        print(f"  {params} -> val PR-AUC={pr_auc:.4f}")
        if pr_auc > best_pr_auc:
            best_pr_auc, best_rf, best_params = pr_auc, rf, params

    with mlflow.start_run(run_name="random_forest_tuned") as run:
        y_pred = best_rf.predict(X_val)
        y_proba = best_rf.predict_proba(X_val)[:, 1]
        metrics = evaluate(y_val, y_pred, y_proba)
        mlflow.log_params({"model": "RandomForestClassifier", "class_weight": "balanced", **best_params})
        mlflow.log_metrics(metrics)
        log_pr_curve(y_val, y_proba, "rf")
        log_confusion_matrix(y_val, y_pred, "rf")
        feat_imp = sorted(zip(MODEL_INPUT_COLS, best_rf.named_steps["clf"].feature_importances_), key=lambda x: -x[1])
        imp_path = "feat_importance_rf.txt"
        with open(imp_path, "w") as f:
            for name, imp in feat_imp:
                f.write(f"{name}: {imp:.4f}\n")
        mlflow.log_artifact(imp_path)
        Path(imp_path).unlink()
        # Random Forest's internal sklearn.tree._tree.Tree C-extension type
        # is flagged "untrusted" by MLflow's skops-based serializer by
        # default (a real, legitimate security measure -- a malicious file
        # could set out-of-bounds node indices skops wouldn't normally
        # check). Safe to trust here: this is a model trained moments ago in
        # this same process, not a file loaded from an external source.
        mlflow.sklearn.log_model(
            best_rf, "model", input_example=X_train.head(2),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(f"\nRandom Forest (val, best of 8 random-search configs): {metrics}")
        print("  Top 5 features:", feat_imp[:5])
        results["random_forest"] = (best_rf, metrics, run.info.run_id)

    # --- XGBoost, random search on val PR-AUC, class imbalance via scale_pos_weight ---
    print("\nTuning XGBoost (random search, val-set PR-AUC)...")
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    xgb_space = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1, 0.2],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
    }
    best_xgb_pr, best_xgb, best_xgb_params = -1, None, None
    for params in ParameterSampler(xgb_space, n_iter=8, random_state=42):
        xgb = Pipeline([("prep", make_preprocessor()),
                        ("clf", XGBClassifier(scale_pos_weight=neg / pos, eval_metric="aucpr",
                                              random_state=42, n_jobs=-1, **params))])
        xgb.fit(X_train, y_train)
        pr_auc = average_precision_score(y_val, xgb.predict_proba(X_val)[:, 1])
        print(f"  {params} -> val PR-AUC={pr_auc:.4f}")
        if pr_auc > best_xgb_pr:
            best_xgb_pr, best_xgb, best_xgb_params = pr_auc, xgb, params

    with mlflow.start_run(run_name="xgboost_tuned") as run:
        y_pred = best_xgb.predict(X_val)
        y_proba = best_xgb.predict_proba(X_val)[:, 1]
        metrics = evaluate(y_val, y_pred, y_proba)
        mlflow.log_params({"model": "XGBClassifier", "scale_pos_weight": round(neg / pos, 2), **best_xgb_params})
        mlflow.log_metrics(metrics)
        log_pr_curve(y_val, y_proba, "xgb")
        log_confusion_matrix(y_val, y_pred, "xgb")
        # Same skops opt-in as the Random Forest: XGBoost's internal types are
        # flagged by default; trusted here because the model was trained in
        # this same process.
        mlflow.sklearn.log_model(
            best_xgb, "model", input_example=X_train.head(2),
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
        )
        print(f"\nXGBoost (val, best of 8 random-search configs): {metrics}")
        results["xgboost"] = (best_xgb, metrics, run.info.run_id)

    # --- Pick the winner by val PR-AUC, evaluate ONCE on the held-out test set ---
    candidates = {k: v for k, v in results.items() if k != "baseline"}
    winner_name = max(candidates, key=lambda k: candidates[k][1]["pr_auc"])
    winner_model, _, winner_run_id = candidates[winner_name]
    print(f"\n=== Winner by val PR-AUC: {winner_name} ===")

    y_pred_test = winner_model.predict(X_test)
    y_proba_test = winner_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, y_pred_test, y_proba_test)
    print(f"Final TEST set metrics ({winner_name}): {test_metrics}")

    with mlflow.start_run(run_id=winner_run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    # --- Register in the Model Registry with champion-challenger promotion ---
    client = MlflowClient()
    model_uri = f"runs:/{winner_run_id}/model"
    registered = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"\nRegistered {MODEL_NAME} version {registered.version}")

    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    should_promote = True
    if prod_versions:
        prod_run = client.get_run(prod_versions[0].run_id)
        prod_pr_auc = prod_run.data.metrics.get("test_pr_auc", prod_run.data.metrics.get("pr_auc", 0))
        print(f"Current Production model (v{prod_versions[0].version}) test PR-AUC: {prod_pr_auc:.4f}")
        print(f"Challenger (v{registered.version}) test PR-AUC: {test_metrics['pr_auc']:.4f}")
        should_promote = test_metrics["pr_auc"] > prod_pr_auc
        if should_promote:
            client.transition_model_version_stage(MODEL_NAME, prod_versions[0].version, "Archived")
    else:
        print("No existing Production model -- this will be the first.")

    if should_promote:
        client.transition_model_version_stage(MODEL_NAME, registered.version, "Production")
        print(f"PROMOTED v{registered.version} to Production (real PR-AUC comparison, not a manual pick).")
    else:
        client.transition_model_version_stage(MODEL_NAME, registered.version, "Staging")
        print(f"NOT promoted -- v{registered.version} kept in Staging (did not beat current Production).")


if __name__ == "__main__":
    main()
