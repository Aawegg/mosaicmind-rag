#!/usr/bin/env bash
# One-shot Hugging Face Spaces deploy.  Free tier, no billing required.
#
# Prereqs (one-time):
#   1. Create a free account at https://huggingface.co/join
#   2. Generate a WRITE token at https://huggingface.co/settings/tokens
#   3. export HF_TOKEN=hf_xxx     (or pass --token)
#   4. export HF_USERNAME=Aawegg  (your HF username, NOT email)
#
# Then just run:  bash scripts/deploy_hf_space.sh
#
# What this does:
#   - Creates a Docker Space `<HF_USERNAME>/mosaicmind` if it doesn't exist.
#   - Sets GOOGLE_API_KEY (and GROQ_API_KEY if present) as Space secrets.
#   - Adds an `hf` git remote and pushes the current branch.
#   - HF auto-builds the Dockerfile and exposes the public URL.

set -euo pipefail

HF_USERNAME="${HF_USERNAME:-}"
HF_TOKEN="${HF_TOKEN:-}"
SPACE_NAME="${SPACE_NAME:-mosaicmind}"

if [[ -z "${HF_USERNAME}" || -z "${HF_TOKEN}" ]]; then
  echo "ERROR: set HF_USERNAME and HF_TOKEN first."
  echo "  export HF_USERNAME=your_hf_username"
  echo "  export HF_TOKEN=hf_xxx                # write token from https://huggingface.co/settings/tokens"
  exit 1
fi

# Pull keys from .env if not already in environment.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "ERROR: GOOGLE_API_KEY not set. Put it in .env or export it."
  exit 1
fi

echo "==> HF user   : ${HF_USERNAME}"
echo "==> Space     : ${HF_USERNAME}/${SPACE_NAME}"

# 1. Create the Space (idempotent) and set secrets via the HF Python API.
#    Uses the huggingface_hub package which is already installed (transitive
#    dep of sentence-transformers).
HF_TOKEN="${HF_TOKEN}" HF_USERNAME="${HF_USERNAME}" SPACE_NAME="${SPACE_NAME}" \
GOOGLE_API_KEY="${GOOGLE_API_KEY}" GROQ_API_KEY="${GROQ_API_KEY:-}" \
.venv/bin/python <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = f"{os.environ['HF_USERNAME']}/{os.environ['SPACE_NAME']}"

print(f"==> Ensuring Space exists: {repo_id}")
api.create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    private=False,
)

print("==> Setting GOOGLE_API_KEY secret")
api.add_space_secret(repo_id=repo_id, key="GOOGLE_API_KEY", value=os.environ["GOOGLE_API_KEY"])

groq = os.environ.get("GROQ_API_KEY", "")
if groq and groq != "dummy_for_now":
    print("==> Setting GROQ_API_KEY secret")
    api.add_space_secret(repo_id=repo_id, key="GROQ_API_KEY", value=groq)
else:
    print("==> Skipping GROQ_API_KEY (not set or dummy)")

print("==> Setting model-routing variables (non-secret)")
for k, v in {
    "MOSAIC_HEAVY_MODEL": "gemini-3.1-pro-preview",
    "MOSAIC_FAST_MODEL": "gemini-3.1-flash-lite",
    "MOSAIC_MM_MODEL": "gemini-3.1-pro-preview",
    "MOSAIC_TEXT_EMBED": "gemini-embedding-001",
    "MOSAIC_TEXT_EMBED_DIM": "3072",
    "MOSAIC_LOG_LEVEL": "INFO",
}.items():
    api.add_space_variable(repo_id=repo_id, key=k, value=v)

print(f"==> Done. URL: https://huggingface.co/spaces/{repo_id}")
PY

# 2. Add the HF remote and push current branch.
HF_REMOTE_URL="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

if git remote get-url hf >/dev/null 2>&1; then
  git remote set-url hf "${HF_REMOTE_URL}"
else
  git remote add hf "${HF_REMOTE_URL}"
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "==> Pushing ${BRANCH} to HF Space (this triggers the Docker build)"
git push hf "${BRANCH}:main" --force

# Strip the token back out of the remote URL so it isn't sitting in .git/config.
git remote set-url hf "https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo ""
echo "==> Live URL    : https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
echo "==> Direct API  : https://${HF_USERNAME}-${SPACE_NAME}.hf.space"
echo "==> Health      : https://${HF_USERNAME}-${SPACE_NAME}.hf.space/healthz"
echo "==> OpenAPI UI  : https://${HF_USERNAME}-${SPACE_NAME}.hf.space/docs"
echo ""
echo "First build takes ~5-10 minutes (downloads Torch + sentence-transformers + ffmpeg)."
echo "Watch progress at the Space URL above."
