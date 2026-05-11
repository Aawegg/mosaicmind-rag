# Deploying MosaicMind to Google Cloud Run

Two paths:

- **Path A — One-shot manual deploy** (`scripts/deploy_cloudrun.sh`). Easiest.
- **Path B — GitOps via GitHub Actions** (`deploy-cloudrun.yml`). Auto-deploys on every push to `main`.

Pick A first, then layer B on top once you're happy.

---

## Prerequisites (one-time, ~10 minutes)

You need:
- A Google account with billing enabled (Cloud Run + Build are free-tier friendly but require billing on file).
- The `gcloud` CLI installed (`brew install --cask google-cloud-sdk`).
- A Gemini API key from [aistudio.google.com](https://aistudio.google.com/apikey).
- (Optional) A Groq API key from [console.groq.com](https://console.groq.com/keys) for fast Whisper.

### 1. Authenticate and pick a project

```bash
gcloud auth login        # opens a browser

# Create a fresh project (recommended), or use an existing one:
gcloud projects create mosaicmind-rag-$(date +%s) --name="MosaicMind RAG"
gcloud config set project YOUR_PROJECT_ID

# Link a billing account (replace BILLING_ID with your account id from `gcloud billing accounts list`):
gcloud billing accounts list
gcloud billing projects link $(gcloud config get-value project) --billing-account=BILLING_ID
```

### 2. Enable the APIs we need

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  iamcredentials.googleapis.com
```

### 3. Create the Artifact Registry repository

```bash
gcloud artifacts repositories create mosaicmind \
  --location=us-central1 \
  --repository-format=docker \
  --description="MosaicMind container images"
```

### 4. Stash your API keys in Secret Manager

```bash
# REPLACE the values with your real keys.
echo -n "AIzaSy_YOUR_GEMINI_KEY" | gcloud secrets create mosaicmind-google-api-key --data-file=-
echo -n "gsk_YOUR_GROQ_KEY"      | gcloud secrets create mosaicmind-groq-api-key   --data-file=-
```

If you don't have a Groq key yet, just put a dummy:

```bash
echo -n "dummy" | gcloud secrets create mosaicmind-groq-api-key --data-file=-
```

(You can update later: `echo -n "REAL_KEY" | gcloud secrets versions add mosaicmind-groq-api-key --data-file=-`)

### 5. Grant the Cloud Run runtime access to the secrets

```bash
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for secret in mosaicmind-google-api-key mosaicmind-groq-api-key; do
  gcloud secrets add-iam-policy-binding "$secret" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/secretmanager.secretAccessor"
done
```

---

## Path A — manual deploy (the fast win)

```bash
bash scripts/deploy_cloudrun.sh
```

That script will:
1. Submit a Cloud Build job that builds the Docker image (~5-10 min the first time, ~2-3 min cached afterwards).
2. Push the image to Artifact Registry.
3. Create / update the Cloud Run service `mosaicmind` with 2 vCPU, 4 GiB RAM, public access.
4. Wire `GOOGLE_API_KEY` and `GROQ_API_KEY` from Secret Manager.
5. Print the live URL.

You'll get something like:

```
==> Live: https://mosaicmind-xxxxxxxxxx-uc.a.run.app
==> Health: https://mosaicmind-xxxxxxxxxx-uc.a.run.app/healthz
==> OpenAPI UI: https://mosaicmind-xxxxxxxxxx-uc.a.run.app/docs
```

Test it:

```bash
URL=https://mosaicmind-xxxxxxxxxx-uc.a.run.app   # paste yours
curl -s "$URL/healthz" | jq

curl -F "file=@some.pdf" "$URL/ingest"

curl -X POST "$URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the document"}' | jq
```

---

## Path B — GitOps with GitHub Actions

After Path A works, set this up so every push to `main` redeploys automatically.

### 1. Create a deploy service account

```bash
PROJECT_ID=$(gcloud config get-value project)

gcloud iam service-accounts create gh-deployer \
  --display-name="GitHub Actions Cloud Run deployer"

DEPLOYER="gh-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

for role in \
  roles/run.admin \
  roles/artifactregistry.writer \
  roles/iam.serviceAccountUser \
  roles/cloudbuild.builds.editor \
  roles/storage.admin; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${DEPLOYER}" --role="$role" --condition=None
done
```

### 2. Set up Workload Identity Federation (keyless auth, no service-account JSON)

```bash
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

gcloud iam workload-identity-pools create gh-pool \
  --location=global --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc gh-provider \
  --location=global \
  --workload-identity-pool=gh-pool \
  --display-name="GitHub OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.actor=assertion.actor" \
  --attribute-condition="assertion.repository=='Aawegg/mosaicmind-rag'"

gcloud iam service-accounts add-iam-policy-binding "gh-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/gh-pool/attribute.repository/Aawegg/mosaicmind-rag"

WIP="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/gh-pool/providers/gh-provider"
echo "GCP_WORKLOAD_IDENTITY_PROVIDER = $WIP"
echo "GCP_SERVICE_ACCOUNT = gh-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
echo "GCP_PROJECT_ID = $PROJECT_ID"
```

### 3. Drop those three values into GitHub repo secrets, then enable the workflow

```bash
gh secret set GCP_PROJECT_ID --body "$PROJECT_ID"
gh secret set GCP_SERVICE_ACCOUNT --body "gh-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER --body "$WIP"

# The deploy workflow is gated on this repo variable so it doesn't fail on every
# push before the secrets exist.  Flip it on once the three secrets above are set:
gh variable set DEPLOY_ENABLED --body "true"
```

### 4. Push and watch it deploy

```bash
git commit --allow-empty -m "trigger first auto-deploy"
git push
gh run watch
```

The workflow `.github/workflows/deploy-cloudrun.yml` will build and deploy on every push to main.

---

## Resource & cost notes

| Knob | Default | Adjust if |
|---|---|---|
| `--cpu` | 2 | bump to 4 if uploads are CPU-heavy (PDF/image processing) |
| `--memory` | 4Gi | CLIP + chunks fit comfortably; raise to 8Gi if you index huge PDFs |
| `--min-instances` | 0 | set to 1 to kill cold starts (~$15/mo extra) |
| `--max-instances` | 3 | safety cap to prevent runaway billing |
| `--concurrency` | 10 | each instance handles 10 simultaneous requests |
| `--timeout` | 300 | LLM-heavy `/query` can take 30-60s; keep generous |

**Free tier covers ~2 million requests/month** plus 360k vCPU-seconds and 180k GiB-seconds — plenty for a portfolio demo.

The hidden cost is **Gemini API tokens** (billed by AI Studio, not Cloud Run). Watch your AI Studio quota; for casual demos it stays inside the free tier.

---

## State persistence (optional upgrade)

Cloud Run instances are ephemeral, so the Chroma DB and uploaded files are wiped when an instance scales down. For a real demo this is fine (you re-ingest every cold start). For a long-lived store, mount a GCS bucket:

```bash
# Create bucket
gsutil mb -l us-central1 gs://${PROJECT_ID}-mosaic-data

# Re-deploy with a GCS volume mount
gcloud run services update mosaicmind --region=us-central1 \
  --add-volume=name=data,type=cloud-storage,bucket=${PROJECT_ID}-mosaic-data \
  --add-volume-mount=volume=data,mount-path=/app/data
```

---

## Custom domain (optional)

```bash
gcloud beta run domain-mappings create \
  --service=mosaicmind --region=us-central1 \
  --domain=mosaic.your-domain.com
```

Add the CNAME / A records it prints to your DNS.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `permission denied on secret ...` | Re-run the IAM grant in step 5 above. |
| `Container failed to start. Failed to start and then listen on the port defined by the PORT environment variable` | Confirm Dockerfile binds to `$PORT` (it does). |
| `the user does not have access to project XYZ` in CI | Workload Identity binding is missing; re-run step 2 of Path B. |
| Cold start times out | Bump `--timeout` to 600 and `--memory` to 8Gi, or set `--min-instances=1`. |
| Image pull is slow | Image is large (~3 GB). Use `--cpu-boost` on cold start: add `--cpu-boost` flag. |

---

## Tearing it down

```bash
gcloud run services delete mosaicmind --region=us-central1
gcloud artifacts repositories delete mosaicmind --location=us-central1
gcloud secrets delete mosaicmind-google-api-key
gcloud secrets delete mosaicmind-groq-api-key
# Optional: delete the project entirely
gcloud projects delete YOUR_PROJECT_ID
```
