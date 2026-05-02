from __future__ import annotations

from domdistill.dom_split import split_dom


def test_split_dom_groups_by_heading():
    html_content = """
    <html>
      <body>
        <h1>Intro</h1>
        <p>First paragraph</p>
        <p>Second paragraph</p>
        <h2>Details</h2>
        <p>Detail paragraph</p>
      </body>
    </html>
    """

    sections = split_dom(html_content)

    assert len(sections) == 2
    assert sections[0].heading.content == "Intro"
    assert [node.content for node in sections[0].nodes] == [
        "First paragraph",
        "Second paragraph",
    ]
    assert sections[1].heading.content == "Details"
    assert [node.content for node in sections[1].nodes] == ["Detail paragraph"]


def test_split_dom_compacts_table_to_column_value_format():
    html_content = """
    <html>
      <body>
        <h1>Report</h1>
        <table>
          <tr><th>Name</th><th>Role</th></tr>
          <tr><td>Ada</td><td>Dev</td></tr>
          <tr><td>Bob</td><td>QA</td></tr>
        </table>
        <p>After table</p>
      </body>
    </html>
    """
    sections = split_dom(html_content)
    assert len(sections) == 1
    tags = [n.tag for n in sections[0].nodes]
    assert tags == ["table", "p"]
    assert sections[0].nodes[0].content == ("Name:[Ada, Bob], Role:[Dev, QA]")


def test_split_dom_nested_table_skipped_as_separate_node():
    html_content = """
    <html><body>
      <h1>Outer</h1>
      <table>
        <tr><th>A</th></tr>
        <tr><td><table><tr><td>inner</td></tr></table></td></tr>
      </table>
    </body></html>
    """
    sections = split_dom(html_content)
    table_nodes = [n for n in sections[0].nodes if n.tag == "table"]
    assert len(table_nodes) == 1
    assert "inner" in table_nodes[0].content
    assert "A:" in table_nodes[0].content


def test_split_dom_drops_banner_and_footer_chrome():
    html_content = """
    <html><body>
      <header role="banner">
        <h2>Navigation Menu</h2>
        <ul><li>GitHub Copilot noise</li></ul>
      </header>
      <main>
        <h1>Profile</h1>
        <p>Arnab bio text</p>
      </main>
      <footer><p>Footer legal</p></footer>
    </body></html>
    """
    sections = split_dom(html_content)
    assert len(sections) == 1
    assert sections[0].heading.content == "Profile"
    flat = " ".join(n.content for n in sections[0].nodes)
    assert "Copilot" not in flat
    assert "Navigation Menu" not in flat
    assert "Footer legal" not in flat
    assert "Arnab" in flat


def test_split_dom_merges_short_inline_code_into_paragraph():
    html_content = """
    <html><body>
      <h1>T</h1>
      <p>Before <code>x = 1</code> after.</p>
    </body></html>
    """
    sections = split_dom(html_content, min_inline_segment_chars=40)
    assert len(sections[0].nodes) == 1
    assert sections[0].nodes[0].tag == "p"
    assert sections[0].nodes[0].content == "Before x = 1 after."


def test_split_dom_emits_long_inline_code_as_own_node():
    long_snippet = "def f(): return 42  # " + "x" * 25
    assert len(long_snippet) >= 40
    html_content = f"""
    <html><body>
      <h1>T</h1>
      <p>Intro <code>{long_snippet}</code> outro.</p>
    </body></html>
    """
    sections = split_dom(html_content, min_inline_segment_chars=40)
    tags = [n.tag for n in sections[0].nodes]
    contents = [n.content for n in sections[0].nodes]
    assert tags == ["p", "code", "p"]
    assert "Intro" in contents[0] and "outro" in contents[2]
    assert contents[1] == " ".join(long_snippet.split())


def test_split_dom_appends_resolved_url_next_to_anchor_text():
    html_content = """
    <html><body>
      <h1>Title</h1>
      <p>See <a href="/docs">the docs</a> for detail.</p>
    </body></html>
    """
    sections = split_dom(
        html_content,
        base_url="https://example.com/app/page",
    )
    assert sections[0].nodes[0].content == (
        "See the docs (https://example.com/docs) for detail."
    )


def test_split_dom_anchor_without_base_keeps_href_as_in_html():
    html_content = """
    <html><body><h1>T</h1><p><a href="relative/path">click</a></p></body></html>
    """
    sections = split_dom(html_content)
    assert sections[0].nodes[0].content == "click (relative/path)"


def test_split_dom_table_cell_includes_link():
    html_content = """
    <html><body>
      <h1>Rep</h1>
      <table>
        <tr><th>Link</th></tr>
        <tr><td><a href="https://api.example/x">API</a></td></tr>
      </table>
    </body></html>
    """
    sections = split_dom(html_content)
    assert "API (https://api.example/x)" in sections[0].nodes[0].content


def test_split_dom_button_text_not_injected_into_list_item():
    html_content = """
    <html><body>
      <h1>Section</h1>
      <ul><li>Hello <button type="button">Sign up</button> world</li></ul>
    </body></html>
    """
    sections = split_dom(html_content)
    assert sections[0].nodes[0].content == "Hello world"


def test_split_dom_accepts_custom_splitter_tags():
    html_content = """
    <html>
      <body>
        <h1>Main heading</h1>
        <p>Paragraph one</p>
        <h3>Custom split</h3>
        <p>Paragraph two</p>
      </body>
    </html>
    """
    sections = split_dom(html_content, splitter_tags=("h3",))
    assert len(sections) == 2
    assert sections[0].heading.content == "root"
    assert sections[1].heading.content == "Custom split"
