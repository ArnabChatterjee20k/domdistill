from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import EmbeddingFn, get_embedding


@dataclass(frozen=True)
class ChunkSelection:
    score: float
    selected_chunks: list[str]
    discarded_chunks: list[str]
    selected_scores: dict[str, float]


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


def _score_candidates(
    *,
    candidate_chunks: list[str],
    query: str,
    heading: str,
    penalty: float,
    embedding_fn: EmbeddingFn | None,
    batch_size: int,
) -> dict[str, float]:
    if not candidate_chunks:
        return {}

    if embedding_fn is None:
        vectors = get_embedding([query, heading, *candidate_chunks], batch_size=batch_size)
        query_embedding = vectors[0]
        heading_embedding = vectors[1]
        chunk_embeddings = vectors[2:]
    else:
        query_embedding = np.asarray(embedding_fn(query), dtype=float)
        heading_embedding = np.asarray(embedding_fn(heading), dtype=float)
        chunk_embeddings = [
            np.asarray(embedding_fn(candidate_chunk), dtype=float)
            for candidate_chunk in candidate_chunks
        ]

    scores: dict[str, float] = {}
    for candidate_chunk, chunk_embedding in zip(candidate_chunks, chunk_embeddings):
        query_similarity_score = get_cosine_similarity(query_embedding, chunk_embedding)
        heading_similarity_score = get_cosine_similarity(heading_embedding, chunk_embedding)
        length_penalty = penalty * np.sqrt(max(len(candidate_chunk), 1))
        scores[candidate_chunk] = max(query_similarity_score, heading_similarity_score) - length_penalty
    return scores


def select_chunks(
    chunks: list[str] | None = None,
    query: str = "",
    heading: str = "",
    penalty: float = 0.0001,
    embedding_fn: EmbeddingFn | None = None,
    batch_size: int = 25,
) -> ChunkSelection:
    if chunks is None:
        chunks = []

    merged_by_span: dict[tuple[int, int], str] = {}
    candidate_chunks: list[str] = []
    seen_candidates: set[str] = set()
    for i in range(len(chunks)):
        merged_parts: list[str] = []
        for j in range(i, len(chunks)):
            merged_parts.append(chunks[j])
            merged_chunk = " ".join(merged_parts)
            merged_by_span[(i, j)] = merged_chunk
            if merged_chunk not in seen_candidates:
                seen_candidates.add(merged_chunk)
                candidate_chunks.append(merged_chunk)

    score_by_chunk = _score_candidates(
        candidate_chunks=candidate_chunks,
        query=query,
        heading=heading,
        penalty=penalty,
        embedding_fn=embedding_fn,
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

        for j in range(i, len(chunks)):
            merged_chunk = merged_by_span[(i, j)]
            current_chunk_score = score_by_chunk[merged_chunk]
            rest_score, rest_segment, rest_discarded = helper(j + 1)
            total_score = current_chunk_score + rest_score - penalty
            if total_score > current_best[0]:
                current_best = (total_score, [merged_chunk, *rest_segment], rest_discarded)

        memo[i] = current_best
        return current_best

    score, selected_chunks, discarded_chunks = helper(0)
    return ChunkSelection(
        score=score,
        selected_chunks=selected_chunks,
        discarded_chunks=discarded_chunks,
        selected_scores={chunk: score_by_chunk[chunk] for chunk in selected_chunks},
    )


def get_chunks(
    chunks: list[str] | None = None,
    query: str = "",
    heading: str = "",
    penalty: float = 0.0001,
    embedding_fn: EmbeddingFn | None = None,
    batch_size: int = 25,
) -> tuple[float, list[str], list[str]]:
    selection = select_chunks(
        chunks=chunks,
        query=query,
        heading=heading,
        penalty=penalty,
        embedding_fn=embedding_fn,
        batch_size=batch_size,
    )
    return selection.score, selection.selected_chunks, selection.discarded_chunks
