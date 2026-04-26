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
    batch_size: int = 25,
) -> float:
    if embedding_fn is None:
        vectors = get_embedding([query, heading, chunk], batch_size=batch_size)
        query_embedding, heading_embedding, chunk_embedding = vectors
    else:
        query_embedding = np.asarray(embedding_fn(query), dtype=float)
        heading_embedding = np.asarray(embedding_fn(heading), dtype=float)
        chunk_embedding = np.asarray(embedding_fn(chunk), dtype=float)

    query_similarity_score = get_cosine_similarity(query_embedding, chunk_embedding)
    heading_similarity_score = get_cosine_similarity(heading_embedding, chunk_embedding)
    length_penalty = penalty * np.sqrt(max(len(chunk), 1))
    return max(query_similarity_score, heading_similarity_score) - length_penalty


def get_chunks(
    chunks: list[str] | None = None,
    query: str = "",
    heading: str = "",
    penalty: float = 0.0001,
    embedding_fn: EmbeddingFn | None = None,
    batch_size: int = 25,
) -> tuple[float, list[str], list[str]]:
    if chunks is None:
        chunks = []
    embed = embedding_fn or get_embedding

    @lru_cache(maxsize=None)
    def score(chunk: str) -> float:
        return get_score_for_chunk(
            query=query,
            heading=heading,
            chunk=chunk,
            penalty=penalty,
            embedding_fn=embed,
            batch_size=batch_size,
        )

    memo: dict[int, tuple[float, list[str], list[str]]] = {}

    def helper(i: int) -> tuple[float, list[str], list[str]]:
        if i == len(chunks):
            return 0.0, [], []
        if i in memo:
            return memo[i]

        current_best: tuple[float, list[str], list[str]] = (-float("inf"), [], [])

        rest_score, rest_segment, rest_discarded = helper(i + 1)
        total_score_if_discard = rest_score - penalty
        if total_score_if_discard > current_best[0]:
            current_best = (total_score_if_discard, rest_segment, [chunks[i], *rest_discarded])

        merged_parts: list[str] = []
        for j in range(i, len(chunks)):
            merged_parts.append(chunks[j])
            merged_chunk = " ".join(merged_parts)
            current_chunk_score = score(merged_chunk)
            rest_score, rest_segment, rest_discarded = helper(j + 1)
            total_score = current_chunk_score + rest_score - penalty
            if total_score > current_best[0]:
                current_best = (total_score, [merged_chunk, *rest_segment], rest_discarded)

        memo[i] = current_best
        return current_best

    return helper(0)
