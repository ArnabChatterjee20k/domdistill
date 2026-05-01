from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .dom_split import SPLITTER_TAGS, split_dom
from .embeddings import EmbeddingFn
from .models import SplittedDomNodes
from .selection import (
    select_sections_document_batch,
    select_chunks,
    SectionInput,
)


def _section_worker(
    query: str,
    section_index: int,
    heading: str,
    chunks: list[str],
    penalty: float,
    embedding_fn: EmbeddingFn | None,
    batch_size: int,
) -> tuple["ChunkSelectionResult", list["RankedChunk"]]:
    selection = select_chunks(
        chunks=chunks,
        query=query,
        heading=heading,
        penalty=penalty,
        embedding_fn=embedding_fn,
        batch_size=batch_size,
    )
    section_result = ChunkSelectionResult(
        score=selection.score,
        selected_chunks=selection.selected_chunks,
        discarded_chunks=selection.discarded_chunks,
        heading=heading,
        section_index=section_index,
    )
    ranked_chunks = [
        RankedChunk(
            content=selected_chunk,
            score=selection.selected_scores[selected_chunk],
            heading=heading,
            section_index=section_index,
        )
        for selected_chunk in selection.selected_chunks
    ]
    return section_result, ranked_chunks


@dataclass(frozen=True)
class ChunkSelectionResult:
    score: float
    selected_chunks: list[str]
    discarded_chunks: list[str]
    heading: str
    section_index: int


@dataclass(frozen=True)
class RankedChunk:
    content: str
    score: float
    heading: str
    section_index: int


@dataclass(frozen=True)
class MultiSectionChunkResult:
    query: str
    top_sections: list[ChunkSelectionResult]
    top_chunks: list[RankedChunk]


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

    def _get_section_chunks(self, query: str, section_index: int) -> ChunkSelectionResult:
        sections = self.sections()
        if not sections:
            raise ValueError("No sections found in document.")
        if section_index < 0 or section_index >= len(sections):
            raise ValueError(f"section_index out of bounds (0..{len(sections)-1})")

        section = sections[section_index]
        chunks = [node.content for node in section.nodes if node.content.strip()]
        selection = select_chunks(
            chunks=chunks,
            query=query,
            heading=section.heading.content,
            penalty=self.penalty,
            embedding_fn=self.embedding_fn,
        )
        return ChunkSelectionResult(
            score=selection.score,
            selected_chunks=selection.selected_chunks,
            discarded_chunks=selection.discarded_chunks,
            heading=section.heading.content,
            section_index=section_index,
        )

    def get_chunks(
        self,
        query: str,
        *,
        top_k_chunks: int = 10,
        pool_size: int = 1,
        batch_size: int = 25,
        use_pool: bool = False,
        concurrency_section_threshold: int = 0,
    ) -> MultiSectionChunkResult:
        sections = self.sections()
        if not sections:
            raise ValueError("No sections found in document.")
        if top_k_chunks < 1:
            raise ValueError("top_k_chunks must be >= 1")
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if concurrency_section_threshold < 0:
            raise ValueError("concurrency_section_threshold must be >= 0")

        section_results: list[ChunkSelectionResult] = []
        ranked_chunks: list[RankedChunk] = []
        use_thread_pool = pool_size > 1

        # fastest route for the small documents(single thread + batch)
        if self.embedding_fn is None:
            section_inputs = [
                SectionInput(
                    section_index=section_index,
                    heading=section.heading.content,
                    chunks=[node.content for node in section.nodes if node.content.strip()],
                )
                for section_index, section in enumerate(sections)
            ]
            selection_result = select_sections_document_batch(
                query=query,
                sections=section_inputs,
                penalty=self.penalty,
                top_k_chunks=top_k_chunks,
                batch_size=batch_size,
                pool_size=pool_size,
                use_pool=use_pool,
                concurrency_section_threshold=concurrency_section_threshold,
            )
            return MultiSectionChunkResult(
                query=selection_result.query,
                top_sections=[
                    ChunkSelectionResult(
                        score=section.score,
                        selected_chunks=section.selected_chunks,
                        discarded_chunks=section.discarded_chunks,
                        heading=section.heading,
                        section_index=section.section_index,
                    )
                    for section in selection_result.top_sections
                ],
                top_chunks=[
                    RankedChunk(
                        content=chunk.content,
                        score=chunk.score,
                        heading=chunk.heading,
                        section_index=chunk.section_index,
                    )
                    for chunk in selection_result.top_chunks
                ],
            )

        if use_thread_pool:
            with ThreadPoolExecutor(
                max_workers=pool_size,
            ) as executor:
                section_futures = [
                    executor.submit(
                        _section_worker,
                        query,
                        section_index,
                        sections[section_index].heading.content,
                        [node.content for node in sections[section_index].nodes if node.content.strip()],
                        self.penalty,
                        self.embedding_fn,
                        batch_size,
                    )
                    for section_index in range(len(sections))
                ]
                for future in as_completed(section_futures):
                    section_result, section_ranked_chunks = future.result()
                    section_results.append(section_result)
                    ranked_chunks.extend(section_ranked_chunks)
        else:
            # Serial fallback for small workloads.
            for section_index in range(len(sections)):
                section_result, section_ranked_chunks = _section_worker(
                    query,
                    section_index,
                    sections[section_index].heading.content,
                    [node.content for node in sections[section_index].nodes if node.content.strip()],
                    self.penalty,
                    self.embedding_fn,
                    batch_size,
                )
                section_results.append(section_result)
                ranked_chunks.extend(section_ranked_chunks)

        ranked_sections = sorted(section_results, key=lambda item: item.score, reverse=True)
        top_chunks = sorted(ranked_chunks, key=lambda item: item.score, reverse=True)[
            : min(top_k_chunks, len(ranked_chunks))
        ]
        return MultiSectionChunkResult(query=query, top_sections=ranked_sections, top_chunks=top_chunks)

    async def get_chunks_async(
        self,
        query: str,
        *,
        top_k_chunks: int = 10,
        pool_size: int = 1,
        batch_size: int = 25,
        use_pool: bool = False,
        concurrency_section_threshold: int = 0,
    ) -> MultiSectionChunkResult:
        return await asyncio.to_thread(
            self.get_chunks,
            query,
            top_k_chunks=top_k_chunks,
            pool_size=pool_size,
            batch_size=batch_size,
            use_pool=use_pool,
            concurrency_section_threshold=concurrency_section_threshold,
        )
