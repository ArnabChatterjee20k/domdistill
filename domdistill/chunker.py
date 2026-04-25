from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dom_split import SPLITTER_TAGS, split_dom
from .embeddings import EmbeddingFn
from .models import SplittedDomNodes
from .selection import get_chunks


@dataclass(frozen=True)
class ChunkSelectionResult:
    score: float
    selected_chunks: list[str]
    discarded_chunks: list[str]
    heading: str
    section_index: int


class HTMLIntentChunker:
    def __init__(
        self,
        html_content: str,
        *,
        splitter_tags: tuple[str, ...] = SPLITTER_TAGS,
        penalty: float = 0.0001,
        cache_dir: str | Path | None = None,
        embedding_fn: EmbeddingFn | None = None,
    ) -> None:
        self.html_content = html_content
        self.splitter_tags = tuple(splitter_tags)
        self.penalty = penalty
        self.cache_dir = cache_dir
        self.embedding_fn = embedding_fn
        self._sections: list[SplittedDomNodes] | None = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        splitter_tags: tuple[str, ...] = SPLITTER_TAGS,
        penalty: float = 0.0001,
        cache_dir: str | Path | None = None,
        embedding_fn: EmbeddingFn | None = None,
        encoding: str = "utf-8",
    ) -> "HTMLIntentChunker":
        file_path = Path(path)
        return cls(
            html_content=file_path.read_text(encoding=encoding),
            splitter_tags=splitter_tags,
            penalty=penalty,
            cache_dir=cache_dir,
            embedding_fn=embedding_fn,
        )

    def sections(self) -> list[SplittedDomNodes]:
        if self._sections is None:
            self._sections = split_dom(
                self.html_content,
                cache_dir=self.cache_dir,
                splitter_tags=self.splitter_tags,
            )
        return self._sections

    def get_chunks(self, query: str, section_index: int = 0) -> ChunkSelectionResult:
        sections = self.sections()
        if not sections:
            raise ValueError("No sections found in document.")
        if section_index < 0 or section_index >= len(sections):
            raise ValueError(f"section_index out of bounds (0..{len(sections)-1})")

        section = sections[section_index]
        chunks = [node.content for node in section.nodes if node.content.strip()]
        score, selected_chunks, discarded_chunks = get_chunks(
            chunks=chunks,
            query=query,
            heading=section.heading.content,
            penalty=self.penalty,
            embedding_fn=self.embedding_fn,
        )
        return ChunkSelectionResult(
            score=score,
            selected_chunks=selected_chunks,
            discarded_chunks=discarded_chunks,
            heading=section.heading.content,
            section_index=section_index,
        )
