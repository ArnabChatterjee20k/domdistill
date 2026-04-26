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
    result = chunker.get_chunks("http server basics", top_k_chunks=3)
    assert len(result.top_sections) >= 1
    assert result.top_sections[0].heading == "Intro"
    assert len(result.top_chunks) >= 1


def test_html_intent_chunker_all_sections_top_k(fake_embedder):
    html_content = """
    <html><body>
      <h1>Intro</h1>
      <p>HTTP servers accept requests.</p>
      <h2>Security</h2>
      <p>Use HTTPS and validate inputs.</p>
      <h2>Databases</h2>
      <p>Indexes improve query speed.</p>
    </body></html>
    """
    chunker = HTMLIntentChunker(
        html_content,
        penalty=0.01,
        splitter_tags=("h1", "h2"),
        embedding_fn=fake_embedder,
    )
    result = chunker.get_chunks(
        "http security",
        top_k_chunks=2,
    )
    assert len(result.top_sections) >= 2
    assert len(result.top_chunks) == 2
