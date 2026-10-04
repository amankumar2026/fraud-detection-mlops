# Deployment Plan

Status: planned, not deployed. Nothing here has been run against a cloud account yet.

## What gets deployed

One Docker image containing:
- the FastAPI service (`api/`)
- the model registry and artifacts (`mlflow.db`, `mlruns/`), with artifact paths rewritten to the container location at build time (`scripts/rewrite_artifact_paths.py`)

The API loads the model currently in the Production stage at startup, so promoting a new model
in the registry and restarting the service is the release process.

## Recommended target: Google Cloud Run

Why:
- runs a container directly, scales to zero, and has a free tier suitable for a portfolio project
- HTTPS endpoint out of the box
- the same image runs locally, so there is no platform-specific code

Alternatives that also fit: Azure Container Apps, AWS App Runner, Render, Railway.

## Steps

1. **Build and test locally** (already verified): `docker build -t fraud-detector-api .` then
   `docker run -p 8000:8000 fraud-detector-api`, and check `/health` and `/predict`.
2. **Push the image to a registry**: Google Artifact Registry, e.g.
   `gcloud artifacts repositories create fraud --repository-format=docker --location=us-central1`
   then tag and push `us-central1-docker.pkg.dev/<project>/fraud/fraud-detector-api:<git-sha>`.
3. **Deploy to Cloud Run**:
   `gcloud run deploy fraud-api --image <image> --region us-central1 --allow-unauthenticated --port 8000 --memory 1Gi --set-env-vars PREDICTION_THRESHOLD=0.5`
   Use `--no-allow-unauthenticated` for anything beyond a demo, since the endpoint scores transactions.
4. **Smoke test**: `curl https://<service-url>/health`, then one `/predict` call.
5. **Automate with GitHub Actions** (extends `.github/workflows/ci.yml`): on push to `main`,
   after tests pass, build the image, push it, and deploy it. Authenticate with a Workload Identity
   Federation service account rather than a stored key file.

## Operational concerns

- **Latency**: single-row predictions take roughly 200-260 ms locally (see `docs/results_and_findings.md`).
  Acceptable for a demo; a real authorization path would need batching or a lighter model.
- **Model rollback**: change the Production stage in the registry to the previous version and
  redeploy. The API reads the stage at startup, so no code change is needed.
- **Prediction log**: `monitoring/prediction_log.jsonl` lives inside the container, so it is lost
  on restart. For real monitoring, send these records to a durable store such as BigQuery or Cloud Storage.
- **Secrets**: none are needed today. Anything added later goes in Secret Manager, not in the image.
- **Cost**: Cloud Run with scale-to-zero costs nothing while idle. Check current pricing before deploying.

## Not done yet

- Real deployment and a live URL
- Durable prediction logging
- Authentication on the endpoint
- Automated deployment from GitHub Actions (CI currently builds the image but does not deploy)
