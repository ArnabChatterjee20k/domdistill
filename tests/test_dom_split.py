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
    assert [node.content for node in sections[0].nodes] == ["First paragraph", "Second paragraph"]
    assert sections[1].heading.content == "Details"
    assert [node.content for node in sections[1].nodes] == ["Detail paragraph"]


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
