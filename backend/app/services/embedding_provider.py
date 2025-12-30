"""
Embedding Provider

Provides a pluggable interface for text embeddings.
Current implementation uses Azure OpenAI embeddings configured via environment variables.
"""

import logging
import os
from typing import List, Protocol

import numpy as np
import openai

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Protocol for embedding model implementations."""

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:  # pragma: no cover - interface only
        ...


class AzureOpenAIEmbeddingModel:
    """Embedding model that calls Azure OpenAI embeddings."""

    def __init__(self, deployment_name: str) -> None:
        if not deployment_name:
            raise ValueError("Azure OpenAI embedding deployment name is required.")

        # Client uses Azure environment variables:
        # - AZURE_OPENAI_ENDPOINT
        # - AZURE_OPENAI_API_KEY
        # - OPENAI_API_VERSION
        # These are read from the environment by the AzureOpenAI client.
        self._client = openai.AzureOpenAI()
        self._deployment_name = deployment_name

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode a batch of texts as embeddings.

        Extra keyword arguments are accepted for compatibility with
        sentence-transformers but are ignored.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        # Normalize input to a list of strings
        if isinstance(texts, str):
            inputs = [texts]
        else:
            inputs = list(texts)

        try:
            response = self._client.embeddings.create(
                model=self._deployment_name,
                input=inputs,
            )
        except Exception as exc:
            logger.error("Error calling Azure OpenAI embeddings: %s", exc)
            raise

        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype=np.float32)


def create_embedding_model(settings: Settings) -> EmbeddingModel:
    """Factory that returns the configured embedding model implementation.

    Configuration is read from environment variables first, then from Settings.

    Required:
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT: deployment name for the embedding model.

    Optional:
        EMBEDDING_BACKEND: currently only `azure_openai` is supported.
    """
    backend = os.getenv("EMBEDDING_BACKEND", "azure_openai").lower()

    if backend not in {"azure_openai", "azure"}:
        raise ValueError(f"Unsupported EMBEDDING_BACKEND: {backend}")

    deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or getattr(
        settings, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""
    )

    if not deployment_name:
        raise ValueError(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not configured. "
            "Set it in the environment or add it to Settings."
        )

    logger.info("Using Azure OpenAI embedding backend with deployment %s", deployment_name)
    return AzureOpenAIEmbeddingModel(deployment_name)
