"""Gemini access through native Google integrations.

We deliberately avoid the OpenAI-compat shim because the native bindings
expose Gemini-specific features (system instructions, tools, caching,
native multimodal parts) that the compat layer flattens.

  - LangChain  : ChatGoogleGenerativeAI  (used by the LangGraph nodes)
  - LlamaIndex : GoogleGenAI / GoogleGenAIEmbedding  (used by query engines)
  - Native SDK : google.genai.Client      (used for File API + audio/video)
"""
from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from mosaicmind.config import get_settings


@lru_cache(maxsize=8)
def gemini_chat_model(model: str | None = None, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """LangChain chat model bound to a Gemini model."""
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=model or s.heavy_model,
        google_api_key=s.google_api_key,
        temperature=temperature,
        timeout=120,
        max_retries=2,
    )


@lru_cache(maxsize=2)
def gemini_embed_model() -> GoogleGenAIEmbedding:
    """LlamaIndex embedding model bound to gemini-embedding-001 (3072-dim)."""
    s = get_settings()
    return GoogleGenAIEmbedding(
        model_name=s.text_embed_model,
        api_key=s.google_api_key,
        embed_batch_size=100,
    )


@lru_cache(maxsize=4)
def gemini_llamaindex_llm(model: str | None = None, temperature: float = 0.2) -> GoogleGenAI:
    """LlamaIndex LLM bound to a Gemini model."""
    s = get_settings()
    return GoogleGenAI(
        model=model or s.heavy_model,
        api_key=s.google_api_key,
        temperature=temperature,
    )


@lru_cache(maxsize=1)
def gemini_native_client():
    """Native google-genai client for File API + native audio/video parts."""
    from google import genai

    return genai.Client(api_key=get_settings().google_api_key)


# Kept for any callers that still want the raw OpenAI-compat client (optional).
def gemini_openai_client():
    """Optional: a raw OpenAI client pointed at Gemini's compat endpoint.

    Only used as an escape hatch for code paths that prefer the OpenAI SDK
    shape. Importing here lazily so `openai` is not a hard dependency.
    """
    from openai import OpenAI as OpenAIClient

    s = get_settings()
    return OpenAIClient(api_key=s.google_api_key, base_url=s.gemini_base_url)
