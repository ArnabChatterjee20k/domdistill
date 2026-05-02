from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunker import HTMLIntentChunker
from .selection import (
    DEFAULT_HEADING_WEIGHT,
    DEFAULT_QUERY_WEIGHT,
    select_chunks,
)


def run_demo() -> dict:
    synthetic_chunks = [
        "HTTP servers accept requests and return responses.",
        "lorem ipsum " * 1200,
        "Key takeaway: caching and HTTPS improve reliability and security.",
        "databases are a key part of the world",
        "i am arnab",
    ]
    selection = select_chunks(
        chunks=synthetic_chunks,
        query="how http servers work",
        heading="Key Takeaways",
        penalty=0.05,
    )
    return {
        "score": selection.score,
        "selected_count": len(selection.selected_chunks),
        "discarded_count": len(selection.discarded_chunks),
        "selected_chunks": selection.selected_chunks,
        "discarded_preview": [
            chunk[:120] + "..." if len(chunk) > 120 else chunk
            for chunk in selection.discarded_chunks
        ],
    }


def run_file(
    path: Path,
    query: str,
    penalty: float = 0.0001,
    top_k_chunks: int = 10,
    pool_size: int = 1,
    batch_size: int = 25,
    use_pool: bool = False,
    concurrency_section_threshold: int = 0,
    max_adjacent_chunks: int | None = 6,
    max_merge_span: int | None = None,
    max_chunks_per_section: int | None = 120,
    query_weight: float = DEFAULT_QUERY_WEIGHT,
    heading_weight: float = DEFAULT_HEADING_WEIGHT,
) -> dict:
    chunker = HTMLIntentChunker.from_file(path, penalty=penalty)
    result = chunker.get_chunks(
        query=query,
        top_k_chunks=top_k_chunks,
        pool_size=pool_size,
        batch_size=batch_size,
        use_pool=use_pool,
        concurrency_section_threshold=concurrency_section_threshold,
        max_adjacent_chunks=max_adjacent_chunks,
        max_merge_span=max_merge_span,
        max_chunks_per_section=max_chunks_per_section,
        query_weight=query_weight,
        heading_weight=heading_weight,
    )
    return {
        "query": result.query,
        "top_chunks": [
            {
                "content": item.content,
                "score": item.score,
                "heading": item.heading,
                "section_index": item.section_index,
            }
            for item in result.top_chunks
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intent-driven DOM chunk distillation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run synthetic demo")
    demo_parser.set_defaults(handler=lambda args: run_demo())

    file_parser = subparsers.add_parser(
        "file", help="Run distillation for one HTML file section"
    )
    file_parser.add_argument("path", type=Path)
    file_parser.add_argument("--query", required=True)
    file_parser.add_argument("--penalty", type=float, default=0.0001)
    file_parser.add_argument("--top-k-chunks", type=int, default=10)
    file_parser.add_argument("--pool-size", type=int, default=1)
    file_parser.add_argument("--batch-size", type=int, default=25)
    file_parser.add_argument(
        "--max-adjacent-chunks",
        type=int,
        default=6,
        help="Maximum adjacent chunks merged into one candidate (higher is slower).",
    )
    file_parser.add_argument(
        "--max-merge-span",
        type=int,
        default=None,
        help="Legacy alias for --max-adjacent-chunks.",
    )
    file_parser.add_argument(
        "--max-chunks-per-section",
        type=int,
        default=120,
        help="Cap nodes considered per section to bound candidate growth.",
    )
    file_parser.add_argument(
        "--use-pool",
        action="store_true",
        help="Use process-pool DP after document-level embedding when threshold is exceeded.",
    )
    file_parser.add_argument(
        "--concurrency-section-threshold",
        type=int,
        default=0,
        help="Use process-pool DP only when section count is greater than this value. 0 disables it.",
    )
    file_parser.add_argument(
        "--query-weight",
        type=float,
        default=DEFAULT_QUERY_WEIGHT,
        help="Weight for query–chunk cosine vs heading–chunk (normalized with --heading-weight).",
    )
    file_parser.add_argument(
        "--heading-weight",
        type=float,
        default=DEFAULT_HEADING_WEIGHT,
        help="Weight for heading–chunk cosine vs query–chunk.",
    )
    file_parser.set_defaults(
        handler=lambda args: run_file(
            path=args.path,
            query=args.query,
            penalty=args.penalty,
            top_k_chunks=args.top_k_chunks,
            pool_size=args.pool_size,
            batch_size=args.batch_size,
            use_pool=args.use_pool,
            concurrency_section_threshold=args.concurrency_section_threshold,
            max_adjacent_chunks=args.max_adjacent_chunks,
            max_merge_span=args.max_merge_span,
            max_chunks_per_section=args.max_chunks_per_section,
            query_weight=args.query_weight,
            heading_weight=args.heading_weight,
        )
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output = args.handler(args)
    print(json.dumps(output, indent=2))
    return 0
