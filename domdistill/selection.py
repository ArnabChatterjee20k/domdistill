from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .embeddings import EmbeddingFn, get_embedding

DEFAULT_QUERY_WEIGHT = 0.85
DEFAULT_HEADING_WEIGHT = 0.15


@dataclass(frozen=True)
class ChunkSelection:
    score: float
    selected_chunks: list[str]
    discarded_chunks: list[str]
    selected_scores: dict[str, float]


@dataclass(frozen=True)
class ChunkCandidates:
    merged_by_span: dict[tuple[int, int], str]
    candidates: list[str]


@dataclass(frozen=True)
class SectionInput:
    section_index: int
    heading: str
    chunks: list[str]


@dataclass(frozen=True)
class SectionSelectionResult:
    score: float
    selected_chunks: list[str]
    discarded_chunks: list[str]
    heading: str
    section_index: int


@dataclass(frozen=True)
class RankedChunkResult:
    content: str
    score: float
    heading: str
    section_index: int


@dataclass(frozen=True)
class MultiSectionSelectionResult:
    query: str
    top_sections: list[SectionSelectionResult]
    top_chunks: list[RankedChunkResult]


@dataclass(frozen=True)
class SectionPreparedWork:
    section_index: int
    heading: str
    chunks: list[str]
    merged_by_span: dict[tuple[int, int], str]
    candidates: list[str]


@dataclass(frozen=True)
class SectionPrecomputedWork:
    section_index: int
    heading: str
    chunks: list[str]
    merged_by_span: dict[tuple[int, int], str]
    score_by_chunk: dict[str, float]
    penalty: float
    max_merge_span: int | None


def get_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def weighted_query_heading_similarity(
    query_similarity: float,
    heading_similarity: float,
    *,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    heading_weight: float = DEFAULT_HEADING_WEIGHT,
) -> float:
    """Blend query–chunk and heading–chunk cosine scores (weights sum-normalized)."""
    if query_weight < 0.0 or heading_weight < 0.0:
        raise ValueError("query_weight and heading_weight must be >= 0")
    denom = query_weight + heading_weight
    if denom <= 0.0:
        raise ValueError("query_weight + heading_weight must be > 0")
    return (
        query_weight * query_similarity + heading_weight * heading_similarity
    ) / denom


def get_score_for_chunk(
    query: str,
    heading: str,
    chunk: str,
    penalty: float = 0.0001,
    embedding_fn: EmbeddingFn | None = None,
    batch_size: int = 25,
    *,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    heading_weight: float = DEFAULT_HEADING_WEIGHT,
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
    combined = weighted_query_heading_similarity(
        query_similarity_score,
        heading_similarity_score,
        query_weight=query_weight,
        heading_weight=heading_weight,
    )
    return combined - length_penalty


def get_score_from_embeddings(
    *,
    query_embedding: np.ndarray,
    heading_embedding: np.ndarray,
    chunk_embedding: np.ndarray,
    chunk: str,
    penalty: float,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    heading_weight: float = DEFAULT_HEADING_WEIGHT,
) -> float:
    query_similarity_score = get_cosine_similarity(query_embedding, chunk_embedding)
    heading_similarity_score = get_cosine_similarity(heading_embedding, chunk_embedding)
    length_penalty = penalty * np.sqrt(max(len(chunk), 1))
    combined = weighted_query_heading_similarity(
        query_similarity_score,
        heading_similarity_score,
        query_weight=query_weight,
        heading_weight=heading_weight,
    )
    return combined - length_penalty


def build_chunk_candidates(
    chunks: list[str] | None = None,
    *,
    max_merge_span: int | None = None,
) -> ChunkCandidates:
    if chunks is None:
        chunks = []
    if max_merge_span is not None and max_merge_span < 1:
        raise ValueError("max_merge_span must be >= 1")

    merged_by_span: dict[tuple[int, int], str] = {}
    candidates: list[str] = []
    seen_candidates: set[str] = set()
    for i in range(len(chunks)):
        merged_parts: list[str] = []
        max_j = (
            len(chunks)
            if max_merge_span is None
            else min(len(chunks), i + max_merge_span)
        )
        for j in range(i, max_j):
            merged_parts.append(chunks[j])
            merged_chunk = " ".join(merged_parts)
            merged_by_span[(i, j)] = merged_chunk
            if merged_chunk not in seen_candidates:
                seen_candidates.add(merged_chunk)
                candidates.append(merged_chunk)
    return ChunkCandidates(merged_by_span=merged_by_span, candidates=candidates)


def _score_candidates(
    *,
    candidate_chunks: list[str],
    query: str,
    heading: str,
    penalty: float,
    embedding_fn: EmbeddingFn | None,
    batch_size: int,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    heading_weight: float = DEFAULT_HEADING_WEIGHT,
) -> dict[str, float]:
    if not candidate_chunks:
        return {}

    if embedding_fn is None:
        vectors = get_embedding(
            [query, heading, *candidate_chunks], batch_size=batch_size
        )
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
        heading_similarity_score = get_cosine_similarity(
            heading_embedding, chunk_embedding
        )
        length_penalty = penalty * np.sqrt(max(len(candidate_chunk), 1))
        combined = weighted_query_heading_similarity(
            query_similarity_score,
            heading_similarity_score,
            query_weight=query_weight,
            heading_weight=heading_weight,
        )
        scores[candidate_chunk] = combined - length_penalty
    return scores


def select_chunks(
    chunks: list[str] | None = None,
    query: str = "",
    heading: str = "",
    penalty: float = 0.0001,
    embedding_fn: EmbeddingFn | None = None,
    batch_size: int = 25,
    max_merge_span: int | None = None,
    *,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    heading_weight: float = DEFAULT_HEADING_WEIGHT,
) -> ChunkSelection:
    if chunks is None:
        chunks = []

    candidates = build_chunk_candidates(chunks, max_merge_span=max_merge_span)

    score_by_chunk = _score_candidates(
        candidate_chunks=candidates.candidates,
        query=query,
        heading=heading,
        penalty=penalty,
        embedding_fn=embedding_fn,
        batch_size=batch_size,
        query_weight=query_weight,
        heading_weight=heading_weight,
    )
    return select_chunks_with_scores(
        chunks=chunks,
        merged_by_span=candidates.merged_by_span,
        score_by_chunk=score_by_chunk,
        penalty=penalty,
        max_merge_span=max_merge_span,
    )


def select_chunks_with_scores(
    *,
    chunks: list[str],
    merged_by_span: dict[tuple[int, int], str],
    score_by_chunk: dict[str, float],
    penalty: float,
    max_merge_span: int | None = None,
) -> ChunkSelection:
    if max_merge_span is not None and max_merge_span < 1:
        raise ValueError("max_merge_span must be >= 1")

    memo: dict[int, tuple[float, list[str], list[str]]] = {}

    def helper(i: int) -> tuple[float, list[str], list[str]]:
        if i == len(chunks):
            return 0.0, [], []
        if i in memo:
            return memo[i]

        current_best: tuple[float, list[str], list[str]] = (-float("inf"), [], [])

        # discard the current node
        rest_score, rest_segment, rest_discarded = helper(i + 1)
        total_score_if_discard = rest_score - penalty
        if total_score_if_discard > current_best[0]:
            current_best = (
                total_score_if_discard,
                rest_segment,
                [chunks[i], *rest_discarded],
            )

        # take the current node
        max_j = (
            len(chunks)
            if max_merge_span is None
            else min(len(chunks), i + max_merge_span)
        )
        for j in range(i, max_j):
            merged_chunk = merged_by_span[(i, j)]
            current_chunk_score = score_by_chunk[merged_chunk]
            rest_score, rest_segment, rest_discarded = helper(j + 1)
            total_score = current_chunk_score + rest_score - penalty
            if total_score > current_best[0]:
                current_best = (
                    total_score,
                    [merged_chunk, *rest_segment],
                    rest_discarded,
                )

        memo[i] = current_best
        return current_best

    score, selected_chunks, discarded_chunks = helper(0)
    return ChunkSelection(
        score=score,
        selected_chunks=selected_chunks,
        discarded_chunks=discarded_chunks,
        selected_scores={chunk: score_by_chunk[chunk] for chunk in selected_chunks},
    )


def select_sections_document_batch(
    *,
    sections: list[SectionInput],
    query: str,
    penalty: float = 0.0001,
    top_k_chunks: int = 10,
    batch_size: int = 25,
    pool_size: int = 1,
    use_pool: bool = False,
    concurrency_section_threshold: int = 0,
    max_merge_span: int | None = None,
    max_chunks_per_section: int | None = None,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    heading_weight: float = DEFAULT_HEADING_WEIGHT,
) -> MultiSectionSelectionResult:
    if top_k_chunks < 1:
        raise ValueError("top_k_chunks must be >= 1")
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if concurrency_section_threshold < 0:
        raise ValueError("concurrency_section_threshold must be >= 0")
    if max_merge_span is not None and max_merge_span < 1:
        raise ValueError("max_merge_span must be >= 1")
    if max_chunks_per_section is not None and max_chunks_per_section < 1:
        raise ValueError("max_chunks_per_section must be >= 1")

    prepared_sections: list[SectionPreparedWork] = []
    all_texts: list[str] = [query]
    seen_texts: set[str] = {query}

    for section in sections:
        section_chunks = section.chunks
        if (
            max_chunks_per_section is not None
            and len(section_chunks) > max_chunks_per_section
        ):
            section_chunks = section_chunks[:max_chunks_per_section]
        candidates = build_chunk_candidates(
            section_chunks, max_merge_span=max_merge_span
        )
        prepared_sections.append(
            SectionPreparedWork(
                section_index=section.section_index,
                heading=section.heading,
                chunks=section_chunks,
                merged_by_span=candidates.merged_by_span,
                candidates=candidates.candidates,
            )
        )
        for text in [section.heading, *candidates.candidates]:
            if text not in seen_texts:
                seen_texts.add(text)
                all_texts.append(text)

    vectors = get_embedding(all_texts, batch_size=batch_size)
    embedding_by_text = dict(zip(all_texts, vectors))
    query_embedding = embedding_by_text[query]

    precomputed_work: list[SectionPrecomputedWork] = []
    for prepared in prepared_sections:
        heading_embedding = embedding_by_text[prepared.heading]
        score_by_chunk = {
            candidate: get_score_from_embeddings(
                query_embedding=query_embedding,
                heading_embedding=heading_embedding,
                chunk_embedding=embedding_by_text[candidate],
                chunk=candidate,
                penalty=penalty,
                query_weight=query_weight,
                heading_weight=heading_weight,
            )
            for candidate in prepared.candidates
        }
        precomputed_work.append(
            SectionPrecomputedWork(
                section_index=prepared.section_index,
                heading=prepared.heading,
                chunks=prepared.chunks,
                merged_by_span=prepared.merged_by_span,
                score_by_chunk=score_by_chunk,
                penalty=penalty,
                max_merge_span=max_merge_span,
            )
        )

    use_dp_process_pool = (
        use_pool
        and concurrency_section_threshold > 0
        and len(precomputed_work) > concurrency_section_threshold
        and pool_size > 1
    )
    if use_dp_process_pool:
        with ProcessPoolExecutor(max_workers=pool_size) as executor:
            section_work_results = list(
                executor.map(select_precomputed_section, precomputed_work)
            )
    else:
        section_work_results = [
            select_precomputed_section(work) for work in precomputed_work
        ]

    section_results: list[SectionSelectionResult] = []
    ranked_chunks: list[RankedChunkResult] = []
    for section_result, section_ranked_chunks in section_work_results:
        section_results.append(section_result)
        ranked_chunks.extend(section_ranked_chunks)

    ranked_sections = sorted(section_results, key=lambda item: item.score, reverse=True)
    top_chunks = sorted(ranked_chunks, key=lambda item: item.score, reverse=True)[
        : min(top_k_chunks, len(ranked_chunks))
    ]
    return MultiSectionSelectionResult(
        query=query,
        top_sections=ranked_sections,
        top_chunks=top_chunks,
    )


def select_precomputed_section(
    work: SectionPrecomputedWork,
) -> tuple[SectionSelectionResult, list[RankedChunkResult]]:
    selection = select_chunks_with_scores(
        chunks=work.chunks,
        merged_by_span=work.merged_by_span,
        score_by_chunk=work.score_by_chunk,
        penalty=work.penalty,
        max_merge_span=work.max_merge_span,
    )
    section_result = SectionSelectionResult(
        score=selection.score,
        selected_chunks=selection.selected_chunks,
        discarded_chunks=selection.discarded_chunks,
        heading=work.heading,
        section_index=work.section_index,
    )
    ranked_chunks = [
        RankedChunkResult(
            content=selected_chunk,
            score=selection.selected_scores[selected_chunk],
            heading=work.heading,
            section_index=work.section_index,
        )
        for selected_chunk in selection.selected_chunks
    ]
    return section_result, ranked_chunks
