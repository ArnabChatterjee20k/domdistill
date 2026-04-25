from __future__ import annotations

from functools import lru_cache

from .embeddings import EmbeddingFn, get_embedding
from .scoring import get_score_for_chunk


def get_best_chunks_only(
    chunks: list[str] | None = None,
    query: str = "",
    heading: str = "",
    penalty: float = 0.0001,
    embedding_fn: EmbeddingFn | None = None,
) -> tuple[float, list[str]]:
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
        )

    @lru_cache(maxsize=None)
    def helper(i: int) -> tuple[float, tuple[str, ...]]:
        if i == len(chunks):
            return 0.0, ()
        current_best: tuple[float, tuple[str, ...]] = (-float("inf"), ())
        merged_parts: list[str] = []
        for j in range(i, len(chunks)):
            merged_parts.append(chunks[j])
            merged_chunk = " ".join(merged_parts)
            current_chunk_score = score(merged_chunk)
            rest_score, rest_segment = helper(j + 1)
            total_score = current_chunk_score + rest_score - penalty
            if total_score > current_best[0]:
                current_best = (total_score, (merged_chunk, *rest_segment))
        return current_best

    best_score, selected = helper(0)
    return best_score, list(selected)


def get_chunks(
    chunks: list[str] | None = None,
    query: str = "",
    heading: str = "",
    penalty: float = 0.0001,
    embedding_fn: EmbeddingFn | None = None,
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


def select_relevant_chunks(
    chunks: list[str],
    query: str,
    heading: str,
    penalty: float = 0.0001,
    embedding_fn: EmbeddingFn | None = None,
) -> tuple[float, list[str], list[str]]:
    return get_chunks(
        chunks=chunks,
        query=query,
        heading=heading,
        penalty=penalty,
        embedding_fn=embedding_fn,
    )
