"""FastAPI app entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from mosaicmind import __version__
from mosaicmind.api.routes import corpus, health, ingest, query
from mosaicmind.api.routes import eval as eval_routes
from mosaicmind.config import get_settings
from mosaicmind.mlops.tracking import init_mlflow
from mosaicmind.utils.logging import logger, setup_logging

STATIC_DIR = Path(__file__).parent / "static"


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
app.include_router(corpus.router)

# Serve the chat UI: GET / returns index.html, /static/* serves assets.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


def run() -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run("mosaicmind.api.main:app", host=s.api_host, port=s.api_port, reload=False)


if __name__ == "__main__":
    run()
