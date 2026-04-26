from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from domdistill.chunker import HTMLIntentChunker


def build_large_html(path: Path, repeat_factor: int) -> str:
    base = path.read_text(encoding="utf-8")
    repeated = "\n".join(base for _ in range(repeat_factor))
    return f"<html><body>{repeated}</body></html>"


def run_once(
    html_content: str,
    query: str,
    top_k_chunks: int,
    pool_size: int,
    penalty: float,
) -> float:
    chunker = HTMLIntentChunker(html_content, penalty=penalty)
    start = time.perf_counter()
    chunker.get_chunks(query=query, top_k_chunks=top_k_chunks, pool_size=pool_size)
    return time.perf_counter() - start


def benchmark(
    html_file: Path,
    repeat_factor: int,
    iterations: int,
    query: str,
    top_k_chunks: int,
    penalty: float,
    pool_sizes: list[int],
) -> dict:
    html_content = build_large_html(html_file, repeat_factor)
    unique_pool_sizes = sorted(set(pool_sizes))
    if 1 not in unique_pool_sizes:
        unique_pool_sizes.insert(0, 1)

    per_pool_results = []
    baseline_avg = None

    for pool_size in unique_pool_sizes:
        samples = [
            run_once(
                html_content,
                query,
                top_k_chunks,
                pool_size=pool_size,
                penalty=penalty,
            )
            for _ in range(iterations)
        ]
        latency_ms_avg = statistics.fmean(samples) * 1000.0
        if len(samples) > 1:
            latency_ms_p95 = statistics.quantiles(samples, n=20)[-1] * 1000.0
        else:
            latency_ms_p95 = latency_ms_avg
        if pool_size == 1:
            baseline_avg = latency_ms_avg
        speedup_x = (baseline_avg / latency_ms_avg) if baseline_avg and latency_ms_avg > 0 else 1.0
        per_pool_results.append(
            {
                "pool_size": pool_size,
                "latency_ms_avg": latency_ms_avg,
                "latency_ms_p95": latency_ms_p95,
                "speedup_vs_pool1_x": speedup_x,
            }
        )

    return {
        "iterations": iterations,
        "html_chars": len(html_content),
        "query": query,
        "pool_sizes": unique_pool_sizes,
        "results": per_pool_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-file", type=Path, default=Path("benchmarks/blog.html"))
    parser.add_argument("--repeat-factor", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--query", default="concurrency models in http servers")
    parser.add_argument("--top-k-chunks", type=int, default=5)
    parser.add_argument("--penalty", type=float, default=0.01)
    parser.add_argument(
        "--pool-sizes",
        default="1,2,4",
        help="Comma-separated pool sizes, e.g. 1,2,4,8",
    )
    args = parser.parse_args()
    pool_sizes = [int(item.strip()) for item in args.pool_sizes.split(",") if item.strip()]
    print(
        benchmark(
            html_file=args.html_file,
            repeat_factor=args.repeat_factor,
            iterations=args.iterations,
            query=args.query,
            top_k_chunks=args.top_k_chunks,
            penalty=args.penalty,
            pool_sizes=pool_sizes,
        )
    )


if __name__ == "__main__":
    main()
