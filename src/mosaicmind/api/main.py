"""FastAPI app entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from mosaicmind import __version__
from mosaicmind.api.routes import eval as eval_routes
from mosaicmind.api.routes import health, ingest, query
from mosaicmind.config import get_settings
from mosaicmind.mlops.tracking import init_mlflow
from mosaicmind.utils.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()
    init_mlflow()
    logger.info(f"MosaicMind v{__version__} ready: heavy={settings.heavy_model} fast={settings.fast_model}")
    yield


app = FastAPI(
    title="MosaicMind",
    description="Multimodal RAG assistant on Gemini 3.x + Groq + LangGraph + LlamaIndex.",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(eval_routes.router)


_LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>MosaicMind</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  :root { color-scheme: light dark; }
  body { font: 16px/1.55 -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
         max-width: 720px; margin: 4rem auto; padding: 0 1.25rem; }
  h1 { font-size: 2rem; margin: 0 0 .25rem; }
  .tag { color: #888; margin-bottom: 2rem; }
  ul { padding-left: 1.25rem; }
  li { margin: .35rem 0; }
  code { background: rgba(127,127,127,.18); padding: .12rem .35rem; border-radius: 4px; }
  a { text-decoration: none; }
  a:hover { text-decoration: underline; }
  .pill { display: inline-block; background: rgba(127,127,127,.15); padding: .15rem .55rem; border-radius: 999px; font-size: .85rem; margin-right: .35rem; }
  footer { margin-top: 3rem; color: #888; font-size: .85rem; }
</style>
</head>
<body>
<h1>MosaicMind</h1>
<p class="tag">Multimodal RAG assistant on Gemini 3.x + Groq + LangGraph + LlamaIndex.</p>

<p>
  <span class="pill">FastAPI</span>
  <span class="pill">LangGraph</span>
  <span class="pill">LlamaIndex</span>
  <span class="pill">Chroma</span>
  <span class="pill">MLflow</span>
  <span class="pill">Airflow</span>
  <span class="pill">Docker</span>
  <span class="pill">Cloud Run</span>
</p>

<h3>API</h3>
<ul>
  <li><a href="/docs">/docs</a> &mdash; interactive OpenAPI playground</li>
  <li><a href="/healthz">/healthz</a> &mdash; service health + active model config</li>
  <li><code>POST /ingest</code> &mdash; multipart upload (PDF / image / audio / video)</li>
  <li><code>POST /query</code> &mdash; ask a question, get a cited answer</li>
  <li><code>POST /eval</code> &mdash; run an LLM-as-judge eval set</li>
</ul>

<h3>Try it</h3>
<pre><code>curl -F "file=@paper.pdf" $URL/ingest
curl -X POST $URL/query \\
  -H 'content-type: application/json' \\
  -d '{"question":"summarize the key findings"}'</code></pre>

<footer>
  <a href="https://github.com/Aawegg/mosaicmind-rag">github.com/Aawegg/mosaicmind-rag</a>
</footer>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing() -> HTMLResponse:
    return HTMLResponse(_LANDING_HTML)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


def run() -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run("mosaicmind.api.main:app", host=s.api_host, port=s.api_port, reload=False)


if __name__ == "__main__":
    run()
