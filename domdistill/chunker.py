from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .dom_split import SPLITTER_TAGS, split_dom
from .embeddings import EmbeddingFn
from .models import SplittedDomNodes
from .selection import get_chunks, get_score_for_chunk

def _section_worker(
    query: str,
    section_index: int,
    heading: str,
    chunks: list[str],
    penalty: float,
    embedding_fn: EmbeddingFn | None,
) -> "ChunkSelectionResult":
    score, selected_chunks, discarded_chunks = get_chunks(
        chunks=chunks,
        query=query,
        heading=heading,
        penalty=penalty,
        embedding_fn=embedding_fn,
    )
    return ChunkSelectionResult(
        score=score,
        selected_chunks=selected_chunks,
        discarded_chunks=discarded_chunks,
        heading=heading,
        section_index=section_index,
    )


def _score_chunk_batch_worker(
    query: str,
    chunk_entries: list[tuple[str, str, int]],
    penalty: float,
    embedding_fn: EmbeddingFn | None,
) -> list["RankedChunk"]:
    ranked: list[RankedChunk] = []
    for selected_chunk, heading, section_index in chunk_entries:
        ranked.append(
            RankedChunk(
                content=selected_chunk,
                score=get_score_for_chunk(
                    query=query,
                    heading=heading,
                    chunk=selected_chunk,
                    penalty=penalty,
                    embedding_fn=embedding_fn,
                ),
                heading=heading,
                section_index=section_index,
            )
        )
    return ranked


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

    def get_chunks(
        self,
        query: str,
        *,
        top_k_chunks: int = 10,
        pool_size: int = 1,
        batch_size: int = 25,
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

        section_results: list[ChunkSelectionResult] = []
        ranked_chunks: list[RankedChunk] = []
        use_thread_pool = pool_size > 1

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
                    )
                    for section_index in range(len(sections))
                ]
                for future in as_completed(section_futures):
                    section_results.append(future.result())

                chunk_entries: list[tuple[str, str, int]] = []
                for result in section_results:
                    for selected_chunk in result.selected_chunks:
                        chunk_entries.append((selected_chunk, result.heading, result.section_index))

                if chunk_entries:
                    chunk_batches = [
                        chunk_entries[idx : idx + batch_size]
                        for idx in range(0, len(chunk_entries), batch_size)
                    ]
                    batch_futures = [
                        executor.submit(
                            _score_chunk_batch_worker,
                            query,
                            batch,
                            self.penalty,
                            self.embedding_fn,
                        )
                        for batch in chunk_batches
                    ]
                    for future in as_completed(batch_futures):
                        ranked_chunks.extend(future.result())
        else:
            # Serial fallback for small workloads.
            for section_index in range(len(sections)):
                section_results.append(self._get_section_chunks(query, section_index))

            chunk_entries: list[tuple[str, str, int]] = []
            for result in section_results:
                for selected_chunk in result.selected_chunks:
                    chunk_entries.append((selected_chunk, result.heading, result.section_index))

            ranked_chunks.extend(
                _score_chunk_batch_worker(
                    query=query,
                    chunk_entries=chunk_entries,
                    penalty=self.penalty,
                    embedding_fn=self.embedding_fn,
                )
            )

        ranked_sections = sorted(section_results, key=lambda item: item.score, reverse=True)
        top_chunks = sorted(ranked_chunks, key=lambda item: item.score, reverse=True)[
            : min(top_k_chunks, len(ranked_chunks))
        ]
        return MultiSectionChunkResult(query=query, top_sections=ranked_sections, top_chunks=top_chunks)


