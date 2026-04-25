from __future__ import annotations

from domdistill.chunker import HTMLIntentChunker


def test_html_intent_chunker_single_entrypoint(fake_embedder):
    html_content = """
    <html><body>
      <h1>Intro</h1>
      <p>HTTP servers accept requests.</p>
      <p>Caching helps with latency.</p>
    </body></html>
    """
    chunker = HTMLIntentChunker(
        html_content,
        penalty=0.01,
        splitter_tags=("h1", "h2"),
        embedding_fn=fake_embedder,
    )
    result = chunker.get_chunks("http server basics")
    assert isinstance(result.score, float)
    assert result.heading == "Intro"
    assert len(result.selected_chunks) >= 1
