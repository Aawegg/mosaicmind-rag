"""Centralized typed configuration loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM keys / endpoints ---
    google_api_key: str = Field(default="", validation_alias="GOOGLE_API_KEY")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        validation_alias="GEMINI_BASE_URL",
    )
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")

    # --- Model routing ---
    heavy_model: str = Field(default="gemini-3.1-pro-preview", validation_alias="MOSAIC_HEAVY_MODEL")
    fast_model: str = Field(default="gemini-3.1-flash-lite", validation_alias="MOSAIC_FAST_MODEL")
    mm_model: str = Field(default="gemini-3.1-pro-preview", validation_alias="MOSAIC_MM_MODEL")
    asr_provider: str = Field(default="groq", validation_alias="MOSAIC_ASR_PROVIDER")
    asr_model: str = Field(default="whisper-large-v3-turbo", validation_alias="MOSAIC_ASR_MODEL")
    text_embed_model: str = Field(default="gemini-embedding-001", validation_alias="MOSAIC_TEXT_EMBED")
    text_embed_dim: int = Field(default=3072, validation_alias="MOSAIC_TEXT_EMBED_DIM")
    image_embed_model: str = Field(default="clip-ViT-B-32", validation_alias="MOSAIC_IMAGE_EMBED")

    # --- Storage ---
    data_dir: Path = Field(default=Path("./data"), validation_alias="MOSAIC_DATA_DIR")
    upload_dir: Path = Field(default=Path("./data/uploads"), validation_alias="MOSAIC_UPLOAD_DIR")
    index_dir: Path = Field(default=Path("./data/index"), validation_alias="MOSAIC_INDEX_DIR")
    chroma_dir: Path = Field(default=Path("./data/chroma"), validation_alias="MOSAIC_CHROMA_DIR")

    # --- MLflow ---
    mlflow_uri: str = Field(default="./data/mlruns", validation_alias="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = Field(default="mosaicmind", validation_alias="MLFLOW_EXPERIMENT_NAME")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", validation_alias="MOSAIC_API_HOST")
    api_port: int = Field(default=8000, validation_alias="MOSAIC_API_PORT")
    log_level: str = Field(default="INFO", validation_alias="MOSAIC_LOG_LEVEL")

    # --- AWS (optional) ---
    aws_region: str | None = Field(default=None, validation_alias="AWS_REGION")
    aws_s3_bucket: str | None = Field(default=None, validation_alias="AWS_S3_BUCKET")

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.index_dir, self.chroma_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
