from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunker import HTMLIntentChunker
from .selection import get_chunks


def run_demo() -> dict:
    synthetic_chunks = [
        "HTTP servers accept requests and return responses.",
        "lorem ipsum " * 1200,
        "Key takeaway: caching and HTTPS improve reliability and security.",
        "databases are a key part of the world",
        "i am arnab",
    ]
    sim_score, sim_selected, sim_discarded = get_chunks(
        chunks=synthetic_chunks,
        query="how http servers work",
        heading="Key Takeaways",
        penalty=0.05,
    )
    return {
        "score": sim_score,
        "selected_count": len(sim_selected),
        "discarded_count": len(sim_discarded),
        "selected_chunks": sim_selected,
        "discarded_preview": [
            chunk[:120] + "..." if len(chunk) > 120 else chunk for chunk in sim_discarded
        ],
    }


def run_file(path: Path, query: str, section_index: int = 0, penalty: float = 0.0001) -> dict:
    chunker = HTMLIntentChunker.from_file(path, penalty=penalty)
    result = chunker.get_chunks(query=query, section_index=section_index)
    return {
        "heading": result.heading,
        "score": result.score,
        "selected_chunks": result.selected_chunks,
        "discarded_chunks": result.discarded_chunks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intent-driven DOM chunk distillation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run synthetic demo")
    demo_parser.set_defaults(handler=lambda args: run_demo())

    file_parser = subparsers.add_parser("file", help="Run distillation for one HTML file section")
    file_parser.add_argument("path", type=Path)
    file_parser.add_argument("--query", required=True)
    file_parser.add_argument("--section-index", type=int, default=0)
    file_parser.add_argument("--penalty", type=float, default=0.0001)
    file_parser.set_defaults(
        handler=lambda args: run_file(
            path=args.path,
            query=args.query,
            section_index=args.section_index,
            penalty=args.penalty,
        )
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output = args.handler(args)
    print(json.dumps(output, indent=2))
    return 0
