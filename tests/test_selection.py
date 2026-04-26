from __future__ import annotations

from domdistill.selection import get_chunks


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
