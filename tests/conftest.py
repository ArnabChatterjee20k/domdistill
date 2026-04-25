from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def fake_embedder():
    keywords = ["http", "server", "database", "cache", "security", "heading", "query"]

    def _embed(text: str) -> np.ndarray:
        lowered = text.lower()
        vec = [float(lowered.count(word)) for word in keywords]
        vec.append(float(len(text)))
        return np.asarray(vec, dtype=float)

    return _embed
