from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from lxml import html

from .models import Node, SplittedDomNodes

SPLITTER_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
TEXT_BLOCK_TAGS = (
    "p",
    "li",
    "td",
    "th",
    "pre",
    "code",
    "blockquote",
    "figcaption",
    "caption",
    "summary",
    "label",
    "dt",
    "dd",
)
IGNORED_TAGS = ("script", "style", "img")


def _cache_key(html_content: str, splitter_tags: tuple[str, ...]) -> str:
    joined_tags = ",".join(splitter_tags)
    content = f"{joined_tags}\n{html_content}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def split_dom(
    html_content: str,
    cache_dir: str | Path | None = None,
    splitter_tags: tuple[str, ...] = SPLITTER_TAGS,
) -> list[SplittedDomNodes]:
    splitter_tags = tuple(tag.lower() for tag in splitter_tags)
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cached_file = (
            cache_path / f"split_dom_{_cache_key(html_content, splitter_tags)}.pkl"
        )
        if cached_file.exists():
            with cached_file.open("rb") as handle:
                return pickle.load(handle)

    tree = html.fromstring(html_content)
    for tag_name in IGNORED_TAGS:
        for ignored in tree.xpath(f"//{tag_name}"):
            ignored.drop_tree()
    dom_nodes: list[Node] = []

    for element in tree.iter():
        tag = getattr(element, "tag", "")
        if not isinstance(tag, str):
            continue

        normalized_tag = tag.lower()
        should_extract = (
            normalized_tag in splitter_tags or normalized_tag in TEXT_BLOCK_TAGS
        )
        if not should_extract:
            continue

        # Pull nested inline text into the nearest meaningful text block.
        text = " ".join(
            part.strip() for part in element.itertext() if part and part.strip()
        )
        if not text:
            continue
        dom_nodes.append(Node(tag=normalized_tag, content=text))

    sections: list[SplittedDomNodes] = []
    current_heading = Node(tag="root", content="root")
    current_nodes: list[Node] = []

    for node in dom_nodes:
        if node.tag in splitter_tags:
            if current_nodes:
                sections.append(
                    SplittedDomNodes(heading=current_heading, nodes=current_nodes)
                )
            current_heading = node
            current_nodes = []
        else:
            current_nodes.append(node)

    if current_nodes:
        sections.append(SplittedDomNodes(heading=current_heading, nodes=current_nodes))

    if cache_dir is not None:
        with cached_file.open("wb") as handle:
            pickle.dump(sections, handle)

    return sections
