from __future__ import annotations

from domdistill.chunker import HTMLIntentChunker
from domdistill.simhash import get_simhash, get_similarity


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


def test_get_fingerprint_returns_document_and_section_hashes():
    html_content = """
    <html><body>
      <h1>Intro</h1>
      <p>HTTP servers accept requests.</p>
      <h2>Security</h2>
      <p>Use HTTPS and validate inputs.</p>
    </body></html>
    """
    chunker = HTMLIntentChunker(html_content, splitter_tags=("h1", "h2"))
    fingerprint = chunker.get_fingerprint()

    assert isinstance(fingerprint.document_hash, int)
    assert fingerprint.document_hash != 0
    assert len(fingerprint.sections) == 2
    assert fingerprint.sections[0].heading == "Intro"
    assert fingerprint.sections[1].heading == "Security"
    assert all(isinstance(section.hash, int) and section.hash != 0 for section in fingerprint.sections)


def test_get_fingerprint_is_stable_for_same_html():
    html_content = """
    <html><body>
      <h1>Intro</h1>
      <p>HTTP servers accept requests.</p>
    </body></html>
    """
    chunker = HTMLIntentChunker(html_content, splitter_tags=("h1", "h2"))
    first = chunker.get_fingerprint()
    second = chunker.get_fingerprint()

    assert first == second


def test_get_fingerprint_detects_section_content_change():
    base_html = """
    <html><body>
      <h1>Intro</h1>
      <p>HTTP servers accept requests.</p>
    </body></html>
    """
    changed_html = """
    <html><body>
      <h1>Intro</h1>
      <p>HTTP servers reject invalid requests.</p>
    </body></html>
    """
    base = HTMLIntentChunker(base_html, splitter_tags=("h1", "h2")).get_fingerprint()
    changed = HTMLIntentChunker(changed_html, splitter_tags=("h1", "h2")).get_fingerprint()

    assert base.sections[0].hash != changed.sections[0].hash
    assert base.document_hash != changed.document_hash
    assert get_similarity(base.document_hash, changed.document_hash) > 0.0
    assert get_similarity(base.document_hash, changed.document_hash) < 1.0


def test_get_fingerprint_matches_manual_section_simhash():
    html_content = """
    <html><body>
      <h1>Intro</h1>
      <p>HTTP servers accept requests.</p>
      <p>Caching helps with latency.</p>
    </body></html>
    """
    chunker = HTMLIntentChunker(html_content, splitter_tags=("h1", "h2"))
    fingerprint = chunker.get_fingerprint()
    section = chunker.sections()[0]

    expected_section_hash = get_simhash(
        "Intro\n"
        + "\n".join(node.content for node in section.nodes if node.content.strip())
    )
    assert fingerprint.sections[0].hash == expected_section_hash
