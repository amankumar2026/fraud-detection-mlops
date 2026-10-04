"""Fraud detection dashboard. Reads real project outputs: the Production model
from the MLflow registry, the test split, and the monitoring reports.

Run from the project root:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import prepare_model_input

MODEL_NAME = "fraud-detector"
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

st.set_page_config(page_title="Fraud Detection", layout="wide")


@st.cache_resource
def load_production():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    run = client.get_run(mv.run_id)
    return model, mv.version, run.data.metrics


@st.cache_data
def load_test_scores():
    model, _, _ = load_production()
    df = pd.read_csv(PROJECT_ROOT / "data/processed/test.csv")
    X = prepare_model_input(df)
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame({"fraud_probability": proba, "Class": df["Class"].values, "Time": df["Time"].values})


st.title("Credit Card Fraud Detection")
st.caption("Production model served from the MLflow registry. Test set = last 15% of transactions by time.")

model, version, metrics = load_production()
scores = load_test_scores()

total = len(scores)
frauds = int(scores["Class"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Test transactions", f"{total:,}")
k2.metric("Known fraud cases", f"{frauds}")
k3.metric("Test PR-AUC (model v{})".format(version), f"{metrics.get('test_pr_auc', float('nan')):.3f}")
k4.metric("Fraud rate in test", f"{frauds / total:.3%}")

st.divider()

st.subheader("Threshold tradeoff")
st.write(
    "Lowering the threshold catches more fraud but blocks more legitimate customers. "
    "Pick the point your business can live with."
)
threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)

pred = (scores["fraud_probability"] >= threshold).astype(int)
y = scores["Class"].values
precision = precision_score(y, pred, zero_division=0)
recall = recall_score(y, pred, zero_division=0)
f1 = f1_score(y, pred, zero_division=0)
cm = confusion_matrix(y, pred, labels=[0, 1])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision", f"{precision:.3f}")
m2.metric("Recall", f"{recall:.3f}")
m3.metric("F1", f"{f1:.3f}")
m4.metric("Legitimate txns blocked", f"{int(cm[0, 1]):,}")

cm_df = pd.DataFrame(cm, index=["Actual legitimate", "Actual fraud"], columns=["Predicted legitimate", "Predicted fraud"])
st.dataframe(cm_df, width="stretch")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Performance over time")
    perf_path = PROJECT_ROOT / "monitoring/performance_over_time.csv"
    if perf_path.exists():
        perf = pd.read_csv(perf_path)
        fig = px.line(perf, x="chunk", y=["precision", "recall", "pr_auc"], markers=True,
                      labels={"value": "score", "chunk": "time chunk (test set)", "variable": "metric"})
        st.plotly_chart(fig, width="stretch")
        st.caption("Each chunk has only 6–18 true fraud cases, so these numbers are noisy. "
                   "Precision drop in later chunks is the signal to watch.")
    else:
        st.info("Run monitoring/performance_over_time.py to generate this chart.")

with right:
    st.subheader("Feature drift (PSI, train vs. test)")
    drift_path = PROJECT_ROOT / "monitoring/drift_report.csv"
    if drift_path.exists():
        drift = pd.read_csv(drift_path).sort_values("psi", ascending=False).head(12)
        fig = px.bar(drift, x="psi", y="feature", orientation="h",
                     labels={"psi": "PSI", "feature": ""})
        fig.add_vline(x=0.2, line_dash="dash", annotation_text="0.2 = significant")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
        st.caption("HourOfDay's high PSI is largely an artifact of this dataset's 2-day span, "
                   "not real behavioral drift.")
    else:
        st.info("Run monitoring/drift_report.py to generate this chart.")
