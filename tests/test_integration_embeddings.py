from __future__ import annotations

import pytest

from domdistill.selection import get_score_for_chunk


@pytest.mark.integration
def test_real_model_scoring_smoke():
    score = get_score_for_chunk(
        query="how http servers work",
        heading="web basics",
        chunk="HTTP servers accept requests and return responses.",
        penalty=0.001,
    )
    assert isinstance(score, float)
