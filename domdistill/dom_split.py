from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from urllib.parse import urlparse, urljoin

from lxml import html

from .models import Node, SplittedDomNodes

SPLITTER_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
TEXT_BLOCK_TAGS = (
    "p",
    "li",
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
# Removed before any text extraction so they cannot inflate parent `li`/`p` text.
IGNORED_TAGS = (
    "script",
    "style",
    "img",
    "picture",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "video",
    "audio",
    "source",
    "track",
    "template",
    "slot",
    "button",
    "input",
    "textarea",
    "select",
    "option",
    "optgroup",
    "datalist",
    "menu",
)

# Landmark / chrome subtrees (dropped entirely). Keeps in-page <nav> (e.g. repo tabs) if
# it is not inside a removed region.
CHROME_DROP_XPATH = (
    "//*[@role='banner']",
    "//*[@role='contentinfo']",
    "//footer",
    "//aside",
)

DOM_SPLIT_CACHE_VERSION = "5"

DEFAULT_MIN_INLINE_SEGMENT_CHARS = 40

# Inline phrasing content: merged into the parent block unless text length >= threshold.
INLINE_TAGS_FOR_SPLIT = frozenset(
    {
        "a",
        "abbr",
        "b",
        "bdi",
        "bdo",
        "cite",
        "code",
        "data",
        "dfn",
        "em",
        "i",
        "kbd",
        "mark",
        "q",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "time",
        "u",
        "var",
        "ruby",
        "rt",
        "rp",
    }
)


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _resolve_href(href: str, base_url: str | None) -> str:
    href = href.strip()
    if not href or base_url is None:
        return href
    if urlparse(href).scheme:
        return href
    return urljoin(base_url, href)


def _anchor_to_text(anchor, base_url: str | None) -> str:
    href_raw = (anchor.get("href") or "").strip()
    inner = _inline_text_with_links(anchor, base_url)
    inner = _collapse_ws(inner) if inner else ""
    if not href_raw:
        return inner
    low = href_raw.lower()
    if low.startswith("javascript:"):
        return inner
    resolved = _resolve_href(href_raw, base_url)
    if not inner:
        return resolved
    if inner == resolved or inner == href_raw:
        return resolved
    return f"{inner} ({resolved})"


def _inline_text_with_links(element, base_url: str | None) -> str:
    """Serialize element subtree: visible text plus ``label (absolute_url)`` for links."""
    chunks: list[str] = []
    if element.text:
        t = element.text.strip()
        if t:
            chunks.append(t)
    for child in element:
        ctag = child.tag.lower() if isinstance(child.tag, str) else ""
        if ctag == "a":
            piece = _anchor_to_text(child, base_url)
            if piece:
                chunks.append(piece)
        else:
            sub = _inline_text_with_links(child, base_url)
            if sub:
                chunks.append(sub)
        if child.tail:
            tt = child.tail.strip()
            if tt:
                chunks.append(tt)
    return " ".join(chunks)


def _cell_plain_text(cell, base_url: str | None) -> str:
    return _collapse_ws(_inline_text_with_links(cell, base_url))


def _segment_block_tags(splitter_tags: tuple[str, ...]) -> frozenset[str]:
    return frozenset(splitter_tags) | frozenset(TEXT_BLOCK_TAGS)


def _has_segmenting_block_ancestor(
    el,
    segment_tags: frozenset[str],
) -> bool:
    parent = el.getparent()
    while parent is not None:
        tag = parent.tag.lower() if isinstance(parent.tag, str) else ""
        if tag in segment_tags:
            return True
        parent = parent.getparent()
    return False


def _segment_block_element(
    element,
    block_tag: str,
    base_url: str | None,
    min_inline_chars: int,
) -> list[tuple[str, str]]:
    """Split ``block_tag`` into runs: short inlines fold into the block; long inlines
    become their own ``(tag, text)`` segments (preserving order)."""
    run: list[str] = []
    segments: list[tuple[str, str]] = []

    def flush_run() -> None:
        if not run:
            return
        merged = _collapse_ws(" ".join(run))
        if merged:
            segments.append((block_tag, merged))
        run.clear()

    if element.text and element.text.strip():
        run.append(element.text.strip())

    for child in element:
        ctag = child.tag.lower() if isinstance(child.tag, str) else ""
        if ctag in INLINE_TAGS_FOR_SPLIT:
            if ctag == "a":
                piece = _anchor_to_text(child, base_url).strip()
            else:
                piece = _inline_text_with_links(child, base_url).strip()
            if len(piece) >= min_inline_chars:
                flush_run()
                if piece:
                    segments.append((ctag, _collapse_ws(piece)))
            else:
                if piece:
                    run.append(piece)
        else:
            sub = _inline_text_with_links(child, base_url).strip()
            if sub:
                run.append(sub)
        if child.tail and child.tail.strip():
            run.append(child.tail.strip())

    flush_run()
    return segments


def _row_belongs_to_table(tr, table) -> bool:
    el = tr.getparent()
    while el is not None:
        if el == table:
            return True
        tag = getattr(el, "tag", "")
        if isinstance(tag, str) and tag.lower() == "table":
            return False
        el = el.getparent()
    return False


def _rows_for_table(table) -> list:
    return [tr for tr in table.xpath(".//tr") if _row_belongs_to_table(tr, table)]


def _row_cells(tr) -> list:
    return tr.xpath("./th|./td")


def _table_to_compact_text(table, base_url: str | None) -> str:
    """Serialize a table as ``col:[v1, v2], ...`` to avoid one node per cell."""
    rows = _rows_for_table(table)
    if not rows:
        return ""

    matrix: list[list[str]] = []
    for tr in rows:
        matrix.append([_cell_plain_text(c, base_url) for c in _row_cells(tr)])

    ncols = max(len(r) for r in matrix)
    first_cells = _row_cells(rows[0])
    has_th = any(isinstance(c.tag, str) and c.tag.lower() == "th" for c in first_cells)
    if has_th:
        header_texts = matrix[0] + [""] * (ncols - len(matrix[0]))
        data_start = 1
    else:
        header_texts = [f"column_{i}" for i in range(ncols)]
        data_start = 0

    parts: list[str] = []
    for j in range(ncols):
        raw_name = (
            header_texts[j].strip() if j < len(header_texts) else ""
        ) or f"column_{j}"
        vals: list[str] = []
        for i in range(data_start, len(matrix)):
            if j < len(matrix[i]):
                v = matrix[i][j].strip()
                if v:
                    vals.append(v)
        parts.append(f"{raw_name}:[{', '.join(vals)}]")
    return ", ".join(parts)


def _is_nested_table(table) -> bool:
    return bool(table.xpath("ancestor::table[1]"))


def _cache_key(
    html_content: str,
    splitter_tags: tuple[str, ...],
    base_url: str | None,
    min_inline_segment_chars: int,
) -> str:
    joined_tags = ",".join(splitter_tags)
    base = base_url or ""
    content = (
        f"{DOM_SPLIT_CACHE_VERSION}\n{joined_tags}\n{base}\n"
        f"{min_inline_segment_chars}\n"
        f"{','.join(IGNORED_TAGS)}\n{CHROME_DROP_XPATH}\n{html_content}"
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _drop_subtrees(elements) -> None:
    """Remove elements deepest-first so parents are still attached when dropping."""
    seen: set[int] = set()
    ordered: list = []
    for el in elements:
        i = id(el)
        if i in seen:
            continue
        seen.add(i)
        ordered.append(el)
    ordered.sort(key=lambda e: len(e.xpath("ancestor::*")), reverse=True)
    for el in ordered:
        if el.getparent() is not None:
            el.drop_tree()


def split_dom(
    html_content: str,
    cache_dir: str | Path | None = None,
    splitter_tags: tuple[str, ...] = SPLITTER_TAGS,
    base_url: str | None = None,
    min_inline_segment_chars: int = DEFAULT_MIN_INLINE_SEGMENT_CHARS,
) -> list[SplittedDomNodes]:
    splitter_tags = tuple(tag.lower() for tag in splitter_tags)
    if min_inline_segment_chars < 1:
        raise ValueError("min_inline_segment_chars must be >= 1")
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cached_file = (
            cache_path
            / f"split_dom_{_cache_key(html_content, splitter_tags, base_url, min_inline_segment_chars)}.pkl"
        )
        if cached_file.exists():
            with cached_file.open("rb") as handle:
                return pickle.load(handle)

    tree = html.fromstring(html_content)
    for tag_name in IGNORED_TAGS:
        for ignored in tree.xpath(f"//{tag_name}"):
            ignored.drop_tree()

    chrome_nodes: list = []
    for xpath in CHROME_DROP_XPATH:
        chrome_nodes.extend(tree.xpath(xpath))
    _drop_subtrees(chrome_nodes)

    dom_nodes: list[Node] = []
    segment_tags = _segment_block_tags(splitter_tags)

    for element in tree.iter():
        tag = getattr(element, "tag", "")
        if not isinstance(tag, str):
            continue

        normalized_tag = tag.lower()

        if normalized_tag == "table":
            if _is_nested_table(element):
                continue
            text = _table_to_compact_text(element, base_url)
            if text:
                dom_nodes.append(Node(tag="table", content=text))
            continue

        if normalized_tag != "table" and element.xpath("ancestor::table[1]"):
            continue

        if normalized_tag in INLINE_TAGS_FOR_SPLIT and _has_segmenting_block_ancestor(
            element, segment_tags
        ):
            continue

        should_extract = (
            normalized_tag in splitter_tags or normalized_tag in TEXT_BLOCK_TAGS
        )
        if not should_extract:
            continue

        if normalized_tag == "pre":
            text = _inline_text_with_links(element, base_url)
            if text:
                dom_nodes.append(Node(tag="pre", content=text))
            continue

        segments = _segment_block_element(
            element,
            normalized_tag,
            base_url,
            min_inline_segment_chars,
        )
        for seg_tag, seg_text in segments:
            if seg_text:
                dom_nodes.append(Node(tag=seg_tag, content=seg_text))

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
