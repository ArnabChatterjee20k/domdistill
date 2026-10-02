from __future__ import annotations

import pytest

from domdistill.chunker import HTMLIntentChunker
from domdistill.rerank import LayaReranker, Tev1Reranker, _normalize_ollama_host
from domdistill.selection import (
    build_chunk_candidates,
    score_candidates_rerank,
    select_chunks_reranked,
)


def _keyword_reranker(keywords: list[str]):
    """A deterministic stand-in for a real reranker.

    Scores each candidate by how many of ``keywords`` it contains, so tests do
    not need to download the laya model.
    """

    def _rerank(query: str, heading: str, candidates: list[str]) -> list[float]:
        return [
            float(sum(candidate.lower().count(word) for word in keywords))
            for candidate in candidates
        ]

    return _rerank


def test_score_candidates_rerank_applies_length_penalty():
    candidates = build_chunk_candidates(["http server", "cache"]).candidates
    rerank = _keyword_reranker(["http", "server"])

    scores = score_candidates_rerank(
        candidate_chunks=candidates,
        query="http server",
        heading="web",
        rerank_fn=rerank,
        penalty=0.0,
    )

    # "http server" scores 2 (http + server) with no penalty.
    assert scores["http server"] == pytest.approx(2.0)
    assert set(scores) == set(candidates)


def test_score_candidates_rerank_rejects_length_mismatch():
    def _bad_rerank(query, heading, candidates):
        return [1.0]  # wrong length

    with pytest.raises(ValueError):
        score_candidates_rerank(
            candidate_chunks=["a", "b"],
            query="q",
            heading="h",
            rerank_fn=_bad_rerank,
            penalty=0.0,
        )


def test_select_chunks_reranked_prefers_relevant_chunk():
    selection = select_chunks_reranked(
        chunks=[
            "http server security basics",
            "lorem ipsum unrelated filler " * 20,
        ],
        query="http server security",
        heading="web",
        rerank_fn=_keyword_reranker(["http", "server", "security"]),
        penalty=0.05,
    )

    joined = " ".join(selection.selected_chunks)
    assert "http server security" in joined


def test_chunker_routes_through_reranker():
    html = """
    <html><body>
      <h2>Networking</h2>
      <p>HTTP server security best practices and TLS.</p>
      <p>Completely unrelated cooking recipe about pasta.</p>
    </body></html>
    """
    chunker = HTMLIntentChunker(
        html,
        splitter_tags=("h1", "h2", "h3"),
        rerank_fn=_keyword_reranker(["http", "server", "security", "tls"]),
    )

    result = chunker.get_chunks("http server security", top_k_chunks=5)

    assert result.top_sections
    joined = " ".join(result.top_sections[0].selected_chunks).lower()
    assert "http server" in joined


def test_laya_reranker_is_import_safe_without_model():
    # Constructing and calling with no candidates must not import/load laya.
    reranker = LayaReranker()
    assert reranker("query", "heading", []) == []


def test_tev1_reranker_empty_candidates_makes_no_request():
    # No candidates => no HTTP call to Ollama.
    reranker = Tev1Reranker(host="http://ollama.invalid:11434")
    assert reranker("query", "heading", []) == []


def test_normalize_ollama_host_adds_scheme():
    assert _normalize_ollama_host("localhost:11434") == "http://localhost:11434"
    assert _normalize_ollama_host("http://h:1/") == "http://h:1"
    assert _normalize_ollama_host("https://h:1") == "https://h:1"


def test_tev1_reranker_scores_candidates_with_stubbed_transport(monkeypatch):
    # Stub _score_one so we exercise __call__ ordering without a live server.
    reranker = Tev1Reranker(max_workers=1)

    def _fake_score(query: str, candidate: str) -> float:
        return float(len(candidate))

    monkeypatch.setattr(reranker, "_score_one", _fake_score)
    scores = reranker("q", "h", ["ab", "abcd"])
    assert scores == [2.0, 4.0]
