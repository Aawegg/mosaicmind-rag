# MosaicMind

> Multimodal RAG assistant. Drop PDFs, images, audio, or video. Ask anything. Get cited answers.
> Built with **FastAPI + LangGraph + LlamaIndex + Chroma**, powered by **Gemini 3.x + Groq Whisper**, tracked with **MLflow**, orchestrated by **Airflow**, shipped via **Docker** + **GitHub Actions**.

---

## Why this exists

Modern enterprise AI roles ask for one specific stack:

> *Python · FastAPI · LlamaIndex · LangChain · LangGraph · multimodal LLMs · RAG · Docker · MLflow · Airflow · AWS · MLOps*

MosaicMind is one project that exercises **all** of them end-to-end.

---

## Architecture

```text
              ┌────────────────────────────────────────────┐
              │          FastAPI (async)                   │
              │   /ingest  /query  /eval  /healthz         │
              └───────────────┬────────────────────────────┘
                              │
       ┌──────────────────────┼─────────────────────────┐
       ▼                      ▼                         ▼
┌──────────────────┐  ┌────────────────────┐   ┌────────────────────┐
│ Ingestion router │  │ LangGraph agent    │   │ MLflow tracking    │
│ pdf · image      │  │ plan → retrieve →  │   │ params · metrics · │
│ audio · video    │  │ synthesize →       │   │ artifacts · prompts│
└────────┬─────────┘  │ reflect (loop)     │   └────────────────────┘
         │            └─────────┬──────────┘
         ▼                      │
┌──────────────────┐            ▼
│ Multimodal       │   ┌─────────────────────┐
│ workers          │   │ LangChain tools      │
│ ── PyMuPDF (PDF) │   │ ── search_text       │
│ ── Gemini Vision │   │ ── search_images     │
│   (image cap)    │   └──────────┬──────────┘
│ ── Groq Whisper  │              │
│   (audio)        │              ▼
│ ── ffmpeg + CV   │   ┌─────────────────────────────────────┐
│   (video)        │   │ LlamaIndex over Chroma (2 stores)   │
└────────┬─────────┘   │ ── mosaic_text  (Gemini embed-001)  │
         │             │ ── mosaic_images (CLIP ViT-B/32)    │
         └────────────►└─────────────────────────────────────┘

              Airflow (orchestration)        Docker / docker-compose
              ── reindex_dag (nightly)       ── api + mlflow services
              ── eval_sweep_dag (weekly)     ── airflow profile (opt)
```

### Model routing

| Role | Model | Why |
|---|---|---|
| Heavy reasoning + synth | `gemini-3.1-pro-preview` | Top-tier reasoning, 200k context |
| Fast routing + reflect | `gemini-3.1-flash-lite` | Sub-second, free of LangChain overhead |
| Multimodal (image cap) | `gemini-3.1-pro-preview` | Native vision via OpenAI-compat |
| Audio transcription | Groq `whisper-large-v3-turbo` | Cheap and fast; Gemini fallback available |
| Text embeddings | `gemini-embedding-001` (3072-dim) | One unified embedding family |
| Image embeddings | `clip-ViT-B-32` (local sentence-transformers) | Free, offline, CLIP cross-modal |

All Gemini calls go through Gemini's **OpenAI-compatible endpoint** (`/v1beta/openai/`), so the entire stack uses the `openai` SDK shape — no provider-specific glue inside LangChain or LlamaIndex.

---

## Quick start

### 1. Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

You'll need `ffmpeg` on your `PATH` for video/audio ingestion:

```bash
brew install ffmpeg          # macOS
sudo apt install -y ffmpeg   # Ubuntu/Debian
```

### 2. Configure

```bash
cp .env.example .env
# fill in GOOGLE_API_KEY (AI Studio) and GROQ_API_KEY
```

### 3. Run the API

```bash
mosaicmind                         # starts FastAPI on :8000
# or:  uvicorn mosaicmind.api.main:app --reload
```

Visit `http://localhost:8000/docs` for the OpenAPI UI.

### 4. Ingest some files

```bash
curl -F "file=@my_paper.pdf"      http://localhost:8000/ingest
curl -F "file=@diagram.png"       http://localhost:8000/ingest
curl -F "file=@meeting.mp3"       http://localhost:8000/ingest
curl -F "file=@lecture.mp4"       http://localhost:8000/ingest
```

Each upload returns a `doc_id` and per-modality chunk counts.

### 5. Ask

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the key findings, and show me any chart that supports them."}'
```

You'll get back:
- A natural-language answer
- Numbered `citations` with source name + page / timestamp
- The agent's `plan` and a step-by-step `trace`
- End-to-end `latency_ms`

### 6. Run the eval suite

```bash
python scripts/run_eval.py
```

This runs every case in `scripts/eval_cases.json` end-to-end through the agent, asks Gemini 3.1 Pro to grade each candidate vs. the reference, and logs the run to MLflow.

---

## Docker

```bash
docker compose up --build
# api: http://localhost:8000
# mlflow ui: http://localhost:5000
```

Bring up the optional Airflow stack:

```bash
docker compose --profile orchestration up
# airflow: http://localhost:8080  (admin / admin)
```

---

## Airflow DAGs

| DAG | Schedule | What it does |
|---|---|---|
| `mosaicmind_reindex` | nightly @ 02:00 | Scans `data/uploads/`, ingests new files, writes a `.seen.txt` marker. |
| `mosaicmind_eval_sweep` | weekly Sun @ 04:00 | Runs `scripts/eval_cases.json` through the agent, logs metrics to MLflow. |

Drop the files in `airflow/dags/` into any existing Airflow deployment, or use the docker-compose `orchestration` profile.

---

## MLflow

Every API call logs to MLflow:

| Event | Params | Metrics | Artifacts |
|---|---|---|---|
| `ingest` | `kind`, `doc_id`, `source` | `text_nodes`, `image_nodes` | – |
| `query` | model names, question size | `latency_ms`, `n_citations` | `question.txt`, `answer.txt`, `plan.json` |
| `eval` | `n_cases` | `avg_score`, `avg_latency_ms` | full `rows.json` |

Open the UI at `http://localhost:5000` (or set `MLFLOW_TRACKING_URI` to a remote server).

---

## Project layout

```
mosaicmind/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── airflow/dags/
│   ├── reindex_dag.py
│   └── eval_sweep_dag.py
├── scripts/
│   ├── seed_data.py
│   ├── run_eval.py
│   └── eval_cases.json
├── src/mosaicmind/
│   ├── config.py
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routes/{health,ingest,query,eval}.py
│   ├── ingestion/
│   │   ├── router.py
│   │   ├── pdf.py
│   │   ├── image.py
│   │   ├── audio.py
│   │   └── video.py
│   ├── indexing/store.py            # LlamaIndex + Chroma (text + image)
│   ├── agents/
│   │   ├── graph.py                 # LangGraph state machine
│   │   ├── tools.py                 # LangChain tools over LlamaIndex
│   │   └── prompts.py
│   ├── llm/
│   │   ├── gemini.py                # OpenAI-compat + native client
│   │   └── groq.py                  # Chat + Whisper
│   ├── mlops/
│   │   ├── tracking.py              # MLflow wrappers
│   │   └── eval.py                  # LLM-as-judge harness
│   └── utils/
│       ├── logging.py
│       └── ids.py
└── tests/
```

---

## Testing

```bash
ruff check src tests
pytest -q
```

Tests are designed to run **without** real API keys — LLM calls are monkeypatched.

CI (`.github/workflows/ci.yml`) runs ruff + pytest on every PR and builds the Docker image.

---

## AWS deployment notes

This is built to drop into AWS with minimal code changes:

| Local | AWS production |
|---|---|
| Local file uploads | **S3** (presigned PUT) |
| Chroma persistent dir | **OpenSearch Serverless** (vector engine) or **pgvector on RDS** |
| MLflow file backend | MLflow on **ECS Fargate** + **RDS Postgres** + **S3 artifact store** |
| Airflow standalone | **MWAA** (Managed Workflows for Apache Airflow) |
| `uvicorn` on a laptop | **ECS Fargate** behind ALB, or **App Runner**, or **Lambda + API Gateway** (with `mangum` adapter) |
| Gemini Studio key | Same key, stored in **Secrets Manager** and injected via task definition |
| Groq key | Same, in Secrets Manager |
| Local CloudWatch substitute (loguru) | Same loguru, ECS auto-ships stdout to **CloudWatch Logs** |

Sketch of the deploy step (CI extension):

```yaml
- run: aws ecr get-login-password | docker login --username AWS --password-stdin $ECR
- run: docker tag mosaicmind:latest $ECR/mosaicmind:$GITHUB_SHA
- run: docker push $ECR/mosaicmind:$GITHUB_SHA
- run: aws ecs update-service --cluster mosaic --service api --force-new-deployment
```

---

## Skills mapped to JD checkboxes

| JD requirement | Where in this repo |
|---|---|
| Python 3.12 | everywhere |
| FastAPI | `src/mosaicmind/api/` |
| LlamaIndex *(must-have)* | `src/mosaicmind/indexing/store.py` |
| LangChain | `src/mosaicmind/agents/tools.py` |
| LangGraph | `src/mosaicmind/agents/graph.py` |
| GPT / LLMs | `src/mosaicmind/llm/gemini.py`, `groq.py` |
| Multimodal AI (text/image/audio/video) | `src/mosaicmind/ingestion/*` |
| CNN | CLIP ViT-B/32 image embedder + image-caption pipeline |
| RAG | `agents/graph.py` + `indexing/store.py` |
| MLflow | `src/mosaicmind/mlops/tracking.py` |
| Airflow | `airflow/dags/*.py` |
| Docker | `Dockerfile`, `docker-compose.yml` |
| MLOps / CI | `.github/workflows/ci.yml` |
| AWS | this README's deployment section |

---

## Roadmap

- [ ] Streaming `/query` via SSE
- [ ] Reranker (BGE cross-encoder) before synthesis
- [ ] Hybrid BM25 + dense via OpenSearch
- [ ] `gemini-2.5-computer-use-preview` tool for live browsing
- [ ] Imagen-4 / Veo-3 generation tools (visual answers)
- [ ] LlamaIndex `SubQuestionQueryEngine` for hard multi-doc questions
- [ ] Auth + per-tenant isolation

---

## License

MIT
