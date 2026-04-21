"""
Shared sentence-transformer encoder — lazy-loaded singleton.

All modules that need text embeddings import from here to avoid loading the
model multiple times per process.
"""
from __future__ import annotations

import os
from typing import List, Optional

import yaml

_encoder = None


def _load_model_id() -> str:
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "classifiers.yaml"
    )
    with open(cfg_path) as fh:
        return (
            yaml.safe_load(fh)
            .get("embeddings", {})
            .get("model_id", "all-mpnet-base-v2")
        )


def get_encoder():
    """Return the shared SentenceTransformer instance (lazy-loaded). Returns None if unavailable."""
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_id = _load_model_id()
        _encoder = SentenceTransformer(model_id)
        return _encoder
    except Exception:
        return None


def encode(text: str) -> Optional[List[float]]:
    """
    Encode *text* into a normalized embedding vector.

    Returns a plain Python list of floats, or None when the model is unavailable.
    """
    enc = get_encoder()
    if enc is None:
        return None
    try:
        return enc.encode(text, normalize_embeddings=True).tolist()
    except Exception:
        return None
