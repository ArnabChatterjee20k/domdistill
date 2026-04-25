from __future__ import annotations

from functools import lru_cache

import numpy as np

from .embeddings import EmbeddingFn, get_embedding


def get_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_score_for_chunk(
    query: str,
    heading: str,
    chunk: str,
    penalty: float = 0.0001,
    embedding_fn: EmbeddingFn | None = None,
) -> float:
    embed = embedding_fn or get_embedding

    @lru_cache(maxsize=None)
    def embed_cached(text: str):
        return np.asarray(embed(text), dtype=float)

    query_embedding = embed_cached(query)
    heading_embedding = embed_cached(heading)
    chunk_embedding = embed_cached(chunk)

    query_similarity_score = get_cosine_similarity(query_embedding, chunk_embedding)
    heading_similarity_score = get_cosine_similarity(heading_embedding, chunk_embedding)
    length_penalty = penalty * np.sqrt(max(len(chunk), 1))
    return max(query_similarity_score, heading_similarity_score) - length_penalty
