from __future__ import annotations

import numpy as np

from domdistill.scoring import get_cosine_similarity, get_score_for_chunk


def test_get_cosine_similarity_handles_zero_vector():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])
    assert get_cosine_similarity(a, b) == 0.0


def test_score_prefers_semantic_match_with_injected_embedder(fake_embedder):
    score = get_score_for_chunk(
        query="http server",
        heading="web",
        chunk="http server request response",
        penalty=0.0001,
        embedding_fn=fake_embedder,
    )
    assert score > 0.0


def test_score_penalizes_long_chunks(fake_embedder):
    short_score = get_score_for_chunk(
        query="database",
        heading="storage",
        chunk="database cache",
        penalty=0.01,
        embedding_fn=fake_embedder,
    )
    long_score = get_score_for_chunk(
        query="database",
        heading="storage",
        chunk=("database cache " * 100).strip(),
        penalty=0.01,
        embedding_fn=fake_embedder,
    )
    assert short_score > long_score
