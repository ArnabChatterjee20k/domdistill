from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from domdistill.chunker import HTMLIntentChunker


def build_large_html(path: Path, repeat_factor: int) -> str:
    base = path.read_text(encoding="utf-8")
    repeated = "\n".join(base for _ in range(repeat_factor))
    return f"<html><body>{repeated}</body></html>"


def run_once(
    chunker: HTMLIntentChunker,
    query: str,
    top_k_chunks: int,
    pool_size: int,
    penalty: float,
    batch_size: int,
    use_pool: bool,
    concurrency_section_threshold: int,
) -> float:
    start = time.perf_counter()
    chunker.get_chunks(
        query=query,
        top_k_chunks=top_k_chunks,
        pool_size=pool_size,
        batch_size=batch_size,
        use_pool=use_pool,
        concurrency_section_threshold=concurrency_section_threshold,
    )
    return time.perf_counter() - start


def benchmark(
    html_file: Path,
    repeat_factor: int,
    iterations: int,
    query: str,
    top_k_chunks: int,
    penalty: float,
    pool_sizes: list[int],
    warmup_runs: int,
    batch_sizes: list[int],
    use_pool: bool,
    concurrency_section_threshold: int,
) -> dict:
    html_content = build_large_html(html_file, repeat_factor)
    unique_pool_sizes = sorted(set(pool_sizes))
    if 1 not in unique_pool_sizes:
        unique_pool_sizes.insert(0, 1)
    unique_batch_sizes = sorted(set(batch_sizes))

    per_config_results = []
    baseline_avg: float | None = None
    baseline_key = (1, min(unique_batch_sizes))

    for batch_size in unique_batch_sizes:
        for pool_size in unique_pool_sizes:
            chunker = HTMLIntentChunker(html_content, penalty=penalty)
            for _ in range(warmup_runs):
                run_once(
                    chunker=chunker,
                    query=query,
                    top_k_chunks=top_k_chunks,
                    pool_size=pool_size,
                    penalty=penalty,
                    batch_size=batch_size,
                    use_pool=use_pool,
                    concurrency_section_threshold=concurrency_section_threshold,
                )

            samples = [
                run_once(
                    chunker=chunker,
                    query=query,
                    top_k_chunks=top_k_chunks,
                    pool_size=pool_size,
                    penalty=penalty,
                    batch_size=batch_size,
                    use_pool=use_pool,
                    concurrency_section_threshold=concurrency_section_threshold,
                )
                for _ in range(iterations)
            ]
            latency_ms_avg = statistics.fmean(samples) * 1000.0
            latency_ms_median = statistics.median(samples) * 1000.0
            if len(samples) > 1:
                latency_ms_p95 = statistics.quantiles(samples, n=20)[-1] * 1000.0
            else:
                latency_ms_p95 = latency_ms_avg

            if (pool_size, batch_size) == baseline_key:
                baseline_avg = latency_ms_avg

            speedup_x = (
                baseline_avg / latency_ms_avg
                if baseline_avg is not None and latency_ms_avg > 0
                else 1.0
            )

            per_config_results.append(
                {
                    "pool_size": pool_size,
                    "batch_size": batch_size,
                    "latency_ms_avg": latency_ms_avg,
                    "latency_ms_median": latency_ms_median,
                    "latency_ms_p95": latency_ms_p95,
                    "speedup_vs_baseline_x": speedup_x,
                }
            )

    per_config_results.sort(key=lambda item: item["latency_ms_avg"])
    best = per_config_results[0] if per_config_results else None
    return {
        "iterations": iterations,
        "warmup_runs": warmup_runs,
        "html_chars": len(html_content),
        "query": query,
        "pool_sizes": unique_pool_sizes,
        "batch_sizes": unique_batch_sizes,
        "use_pool": use_pool,
        "concurrency_section_threshold": concurrency_section_threshold,
        "baseline": {"pool_size": baseline_key[0], "batch_size": baseline_key[1]},
        "best": best,
        "results": per_config_results,
    }


def classify_status(current_latency_ms: float, target_latency_ms: float) -> str:
    if current_latency_ms > target_latency_ms * 1.05:
        return "bad"
    if current_latency_ms >= target_latency_ms * 0.95:
        return "stable"
    if current_latency_ms >= target_latency_ms * 0.85:
        return "small improvement"
    return "great improvement"


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch  # type: ignore

        torch.set_num_threads(1)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--html-file", type=Path, default=Path("benchmarks/blog.html"))
    parser.add_argument("--repeat-factor", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--query", default="concurrency models in http servers")
    parser.add_argument("--top-k-chunks", type=int, default=5)
    parser.add_argument("--penalty", type=float, default=0.01)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--batch-sizes",
        default="32,128",
        help="Comma-separated batch sizes, e.g. 32,64,128,256",
    )
    parser.add_argument("--baseline-pool-size", type=int, default=1)
    parser.add_argument("--baseline-batch-size", type=int, default=32)
    parser.add_argument(
        "--target-latency-ms",
        type=float,
        default=None,
        help="Expected max latency (avg ms) for baseline config.",
    )
    parser.add_argument(
        "--enforce-regression-check",
        action="store_true",
        help="Exit non-zero if status is bad.",
    )
    parser.add_argument(
        "--pool-sizes",
        default="1,2,4",
        help="Comma-separated pool sizes, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--use-pool",
        action="store_true",
        help="Use process-pool DP after document-level embedding when threshold is exceeded.",
    )
    parser.add_argument(
        "--concurrency-section-threshold",
        type=int,
        default=0,
        help="Use process-pool DP only when section count is greater than this value. 0 disables it.",
    )
    args = parser.parse_args()
    pool_sizes = [int(item.strip()) for item in args.pool_sizes.split(",") if item.strip()]
    batch_sizes = [int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()]
    result = benchmark(
        html_file=args.html_file,
        repeat_factor=args.repeat_factor,
        iterations=args.iterations,
        query=args.query,
        top_k_chunks=args.top_k_chunks,
        penalty=args.penalty,
        pool_sizes=pool_sizes,
        warmup_runs=args.warmup_runs,
        batch_sizes=batch_sizes,
        use_pool=args.use_pool,
        concurrency_section_threshold=args.concurrency_section_threshold,
    )

    if args.target_latency_ms is not None:
        baseline_match = next(
            (
                row
                for row in result["results"]
                if row["pool_size"] == args.baseline_pool_size
                and row["batch_size"] == args.baseline_batch_size
            ),
            None,
        )
        if baseline_match is None:
            raise ValueError("Baseline config not found in benchmark results.")

        status = classify_status(baseline_match["latency_ms_avg"], args.target_latency_ms)
        result["regression_check"] = {
            "baseline_pool_size": args.baseline_pool_size,
            "baseline_batch_size": args.baseline_batch_size,
            "target_latency_ms": args.target_latency_ms,
            "measured_latency_ms_avg": baseline_match["latency_ms_avg"],
            "status": status,
        }
        if args.enforce_regression_check and status == "bad":
            print(json.dumps(result, indent=2))
            raise SystemExit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
