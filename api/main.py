"""FastAPI serving layer. Loads whichever model is currently in the MLflow
Registry's "Production" stage at startup -- the same promotion mechanism
scripts/train.py uses, so the API always serves the real current champion,
not a hardcoded file path that could silently go stale.
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import prepare_model_input
from api.schemas import ModelInfoResponse, PredictionResponse, TransactionRequest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# SQLite, not a plain file store -- see scripts/train.py for why (this
# MLflow version puts the file store in maintenance mode by default).
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MODEL_NAME = os.environ.get("MODEL_NAME", "fraud-detector")
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.5"))
PREDICTION_LOG_PATH = PROJECT_ROOT / "monitoring" / "prediction_log.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraud-api")

state = {"model": None, "version": None, "stage": None, "run_id": None,
         "val_metrics": {}, "test_metrics": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # lifespan, not the deprecated @app.on_event("startup") -- also matters
    # for testing: FastAPI's TestClient only runs startup/shutdown logic
    # when used as a context manager (`with TestClient(app) as client:`),
    # which only works correctly with lifespan handlers in current versions.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not prod_versions:
        logger.error(f"No model in Production stage for '{MODEL_NAME}'. "
                      f"Run scripts/train.py first.")
        raise RuntimeError(f"No Production model found for '{MODEL_NAME}'. Run scripts/train.py first.")

    mv = prod_versions[0]
    # sklearn.load_model (not pyfunc) -- pyfunc's generic predict() only
    # exposes class labels; this app needs predict_proba for the fraud
    # probability score, so it needs the real sklearn Pipeline object.
    state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    state["version"] = mv.version
    state["stage"] = mv.current_stage
    state["run_id"] = mv.run_id

    run = client.get_run(mv.run_id)
    all_metrics = run.data.metrics
    state["val_metrics"] = {k: v for k, v in all_metrics.items() if not k.startswith("test_")}
    state["test_metrics"] = {k[5:]: v for k, v in all_metrics.items() if k.startswith("test_")}

    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loaded model '{MODEL_NAME}' v{mv.version} (stage={mv.current_stage}, "
                f"run_id={mv.run_id}) — val PR-AUC={state['val_metrics'].get('pr_auc', 'n/a')}")
    yield
    state["model"] = None


app = FastAPI(title="Fraud Detection API", version="1.0", lifespan=lifespan)


def log_prediction(request_data: dict, proba: float, is_fraud: bool, latency_ms: float):
    record = {
        "timestamp": time.time(),
        "model_version": state["version"],
        "fraud_probability": proba,
        "is_fraud": is_fraud,
        "latency_ms": latency_ms,
        **request_data,
    }
    try:
        with open(PREDICTION_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.warning(f"Could not write prediction log: {e}")


@app.get("/health")
def health():
    return {"status": "ok" if state["model"] is not None else "model not loaded"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        version=str(state["version"]),
        stage=state["stage"],
        run_id=state["run_id"],
        val_metrics=state["val_metrics"],
        test_metrics=state["test_metrics"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    row = pd.DataFrame([transaction.model_dump()])
    X = prepare_model_input(row)
    proba = float(state["model"].predict_proba(X)[0, 1])
    is_fraud = proba >= PREDICTION_THRESHOLD
    latency_ms = (time.time() - t0) * 1000

    log_prediction(transaction.model_dump(), proba, is_fraud, latency_ms)

    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=is_fraud,
        threshold=PREDICTION_THRESHOLD,
        model_version=str(state["version"]),
        model_stage=state["stage"],
        latency_ms=latency_ms,
    )
