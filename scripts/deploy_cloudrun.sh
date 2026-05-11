#!/usr/bin/env bash
# One-shot Cloud Run deploy.
#
# Prereqs (one-time, see DEPLOY.md):
#   1. gcloud auth login
#   2. gcloud projects create mosaicmind-rag      (or pick an existing project)
#   3. gcloud config set project mosaicmind-rag
#   4. gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
#                              cloudbuild.googleapis.com secretmanager.googleapis.com
#   5. gcloud artifacts repositories create mosaicmind --location=us-central1 \
#                                                       --repository-format=docker
#   6. echo -n "$YOUR_GEMINI_KEY" | gcloud secrets create mosaicmind-google-api-key --data-file=-
#      echo -n "$YOUR_GROQ_KEY"   | gcloud secrets create mosaicmind-groq-api-key   --data-file=-
#
# Then just run:  bash scripts/deploy_cloudrun.sh
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-mosaicmind}"
REPO="${REPO:-mosaicmind}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:${IMAGE_TAG}"

if [[ -z "${PROJECT}" ]]; then
  echo "ERROR: no GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

echo "==> Project : ${PROJECT}"
echo "==> Region  : ${REGION}"
echo "==> Image   : ${IMAGE}"

echo "==> Building image via Cloud Build (this is the slow step, ~5-10 min)"
gcloud builds submit --tag "${IMAGE}" --timeout=1800s .

echo "==> Deploying to Cloud Run"
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --cpu=2 \
  --memory=4Gi \
  --timeout=300 \
  --concurrency=10 \
  --min-instances=0 \
  --max-instances=3 \
  --no-cpu-throttling \
  --set-env-vars="MOSAIC_DATA_DIR=/app/data,MOSAIC_UPLOAD_DIR=/app/data/uploads,MOSAIC_INDEX_DIR=/app/data/index,MOSAIC_CHROMA_DIR=/app/data/chroma,MLFLOW_TRACKING_URI=/app/data/mlruns" \
  --update-secrets="GOOGLE_API_KEY=mosaicmind-google-api-key:latest,GROQ_API_KEY=mosaicmind-groq-api-key:latest"

URL=$(gcloud run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)')
echo ""
echo "==> Live: ${URL}"
echo "==> Health: ${URL}/healthz"
echo "==> OpenAPI UI: ${URL}/docs"
