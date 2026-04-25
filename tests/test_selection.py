from __future__ import annotations

from domdistill.selection import get_best_chunks_only, get_chunks


def test_get_chunks_returns_selected_and_discarded(fake_embedder):
    score, selected, discarded = get_chunks(
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

    assert isinstance(score, float)
    assert len(selected) >= 1
    assert len(discarded) >= 1


def test_get_best_chunks_only_returns_non_empty_selection(fake_embedder):
    score, selected = get_best_chunks_only(
        chunks=["database indexing", "query planning", "lorem ipsum " * 150],
        query="database query",
        heading="performance",
        penalty=0.03,
        embedding_fn=fake_embedder,
    )

    assert score > -100.0
    assert len(selected) >= 1
