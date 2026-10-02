"""Head-to-head retrieval quality + speed: bi-encoder vs. laya reranker.

Runs the same labeled cases used by ``eval_retrieval.py`` through two scorers:

* ``embedding`` — the default sentence-transformers bi-encoder (cosine).
* ``laya`` — the ``LayaReranker`` cross-encoder (``noul`` relevance probability).

For each engine it reports macro precision / recall / wrong-merge-rate and
wall-clock latency, then prints a Markdown comparison table (handy for CI job
summaries and PR descriptions) followed by the full JSON.

Usage:
    python benchmarks/eval_rerank.py \
        --html-file benchmarks/blog.html \
        --cases-file benchmarks/eval_cases.json \
        --penalty 0.01 \
        --engines embedding,laya
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

# eval_rerank.py lives next to eval_retrieval.py; reuse its per-case metric.
from eval_retrieval import evaluate_case

from domdistill.chunker import HTMLIntentChunker

SPLITTER_TAGS = ("h1", "h2", "h3")


def _build_chunker(engine: str, html_content: str, penalty: float) -> HTMLIntentChunker:
    if engine == "embedding":
        return HTMLIntentChunker(
            html_content, penalty=penalty, splitter_tags=SPLITTER_TAGS
        )
    if engine == "laya":
        from domdistill.rerank import LayaReranker

        return HTMLIntentChunker(
            html_content,
            penalty=penalty,
            splitter_tags=SPLITTER_TAGS,
            rerank_fn=LayaReranker(),
        )
    raise ValueError(f"unknown engine: {engine!r} (expected 'embedding' or 'laya')")


def run_engine(
    engine: str, html_content: str, cases: list[dict], penalty: float
) -> dict:
    chunker = _build_chunker(engine, html_content, penalty)

    # Warm up so model-load / first-call cost is not charged to case latency.
    warmup_started = time.perf_counter()
    evaluate_case(chunker, cases[0])
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    case_results: list[dict] = []
    latencies_ms: list[float] = []
    for case in cases:
        started = time.perf_counter()
        result = evaluate_case(chunker, case)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["latency_ms"] = elapsed_ms
        latencies_ms.append(elapsed_ms)
        case_results.append(result)

    precision_values = [result["precision"] for result in case_results]
    recall_values = [result["recall"] for result in case_results]
    wrong_merge_values = [result["wrong_merge_rate"] for result in case_results]

    return {
        "engine": engine,
        "cases_evaluated": len(case_results),
        "warmup_ms": warmup_ms,
        "macro_precision": statistics.fmean(precision_values)
        if precision_values
        else 0.0,
        "macro_recall": statistics.fmean(recall_values) if recall_values else 0.0,
        "macro_wrong_merge_rate": statistics.fmean(wrong_merge_values)
        if wrong_merge_values
        else 0.0,
        "latency_ms_avg": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        "latency_ms_max": max(latencies_ms) if latencies_ms else 0.0,
        "case_results": case_results,
    }


def _markdown_table(reports: list[dict]) -> str:
    header = (
        "| Engine | Precision | Recall | Wrong-merge | Avg latency (ms) | "
        "Warmup (ms) |\n"
        "|---|---|---|---|---|---|"
    )
    rows = [
        "| {engine} | {macro_precision:.3f} | {macro_recall:.3f} | "
        "{macro_wrong_merge_rate:.3f} | {latency_ms_avg:.1f} | {warmup_ms:.0f} |".format(
            **report
        )
        for report in reports
    ]
    return "\n".join([header, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bi-encoder vs. laya reranker retrieval quality/speed."
    )
    parser.add_argument("--html-file", type=Path, default=Path("benchmarks/blog.html"))
    parser.add_argument(
        "--cases-file", type=Path, default=Path("benchmarks/eval_cases.json")
    )
    parser.add_argument("--penalty", type=float, default=0.01)
    parser.add_argument(
        "--engines",
        type=str,
        default="embedding,laya",
        help="comma-separated engines to run (embedding, laya)",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="optional path to write the Markdown comparison table",
    )
    args = parser.parse_args()

    html_content = args.html_file.read_text(encoding="utf-8")
    cases = json.loads(args.cases_file.read_text(encoding="utf-8"))["cases"]
    engines = [item.strip() for item in args.engines.split(",") if item.strip()]

    reports = [
        run_engine(engine, html_content, cases, args.penalty) for engine in engines
    ]

    table = _markdown_table(reports)
    print(table)
    print()
    print(json.dumps({"penalty": args.penalty, "reports": reports}, indent=2))

    if args.markdown_out is not None:
        args.markdown_out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
