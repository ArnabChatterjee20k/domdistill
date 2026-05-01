from __future__ import annotations

import numpy as np

from domdistill import selection
from domdistill.selection import (
    SectionInput,
    SectionPrecomputedWork,
    build_chunk_candidates,
    select_chunks,
    select_precomputed_section,
    select_sections_document_batch,
)


def test_select_chunks_returns_selected_and_discarded(fake_embedder):
    selection_result = select_chunks(
        chunks=[
            "http server basics",
            "lorem ipsum " * 200,
            "cache and security",
        ],
        query="http server security",
        heading="web",
        penalty=0.05,
        embedding_fn=fake_embedder,
    )

    assert isinstance(selection_result.score, float)
    assert len(selection_result.selected_chunks) >= 1
    assert len(selection_result.discarded_chunks) >= 1


def test_build_chunk_candidates_returns_contiguous_merges():
    candidates = build_chunk_candidates(["a", "b", "c"])

    assert candidates.merged_by_span == {
        (0, 0): "a",
        (0, 1): "a b",
        (0, 2): "a b c",
        (1, 1): "b",
        (1, 2): "b c",
        (2, 2): "c",
    }
    assert candidates.candidates == ["a", "a b", "a b c", "b", "b c", "c"]


def test_select_precomputed_section_reuses_scores_without_embedding():
    candidates = build_chunk_candidates(["http server", "cache"])
    work = SectionPrecomputedWork(
        section_index=4,
        heading="web",
        chunks=["http server", "cache"],
        merged_by_span=candidates.merged_by_span,
        score_by_chunk={
            "http server": 1.0,
            "http server cache": 3.0,
            "cache": 1.0,
        },
        penalty=0.01,
    )

    section_result, ranked_chunks = select_precomputed_section(work)

    assert section_result.section_index == 4
    assert section_result.selected_chunks == ["http server cache"]
    assert ranked_chunks[0].content == "http server cache"
    assert ranked_chunks[0].score == 3.0


def test_select_sections_document_batch_uses_one_embedding_call(monkeypatch):
    calls: list[list[str]] = []

    def fake_get_embedding(texts: list[str], batch_size: int = 25) -> np.ndarray:
        calls.append(texts)
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(lowered.count("http")),
                    float(lowered.count("cache")),
                    float(lowered.count("security")),
                    float(len(text)),
                ]
            )
        return np.asarray(vectors, dtype=float)

    monkeypatch.setattr(selection, "get_embedding", fake_get_embedding)

    result = select_sections_document_batch(
        sections=[
            SectionInput(
                section_index=0,
                heading="web",
                chunks=["http server", "cache layer"],
            ),
            SectionInput(
                section_index=1,
                heading="security",
                chunks=["security headers"],
            ),
        ],
        query="http cache",
        top_k_chunks=2,
        batch_size=32,
    )

    assert len(calls) == 1
    assert calls[0][0] == "http cache"
    assert len(result.top_sections) == 2
    assert len(result.top_chunks) == 2


def test_select_sections_document_batch_process_pool_matches_serial(monkeypatch):
    def fake_get_embedding(texts: list[str], batch_size: int = 25) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(lowered.count("http")),
                    float(lowered.count("cache")),
                    float(lowered.count("security")),
                    float(len(text)),
                ]
            )
        return np.asarray(vectors, dtype=float)

    sections = [
        SectionInput(section_index=0, heading="web", chunks=["http server", "cache layer"]),
        SectionInput(section_index=1, heading="security", chunks=["security headers"]),
    ]
    monkeypatch.setattr(selection, "get_embedding", fake_get_embedding)
    serial = select_sections_document_batch(
        sections=sections,
        query="http cache",
        top_k_chunks=2,
        pool_size=1,
        use_pool=False,
        concurrency_section_threshold=0,
    )
    pooled = select_sections_document_batch(
        sections=sections,
        query="http cache",
        top_k_chunks=2,
        pool_size=2,
        use_pool=True,
        concurrency_section_threshold=1,
    )

    assert pooled == serial
