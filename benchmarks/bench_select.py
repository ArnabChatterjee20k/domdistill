from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from domdistill.selection import get_chunks


def fake_embedder(text: str) -> np.ndarray:
    lowered = text.lower()
    return np.asarray(
        [
            float(lowered.count("http")),
            float(lowered.count("database")),
            float(lowered.count("cache")),
            float(lowered.count("security")),
            float(len(text)),
        ],
        dtype=float,
    )


def make_chunks(size: int) -> list[str]:
    base = [
        "http server architecture",
        "database indexing and transactions",
        "cache invalidation strategies",
        "security headers and tls",
    ]
    return [base[i % len(base)] for i in range(size)]


def run(size: int, iterations: int) -> dict[str, float]:
    chunks = make_chunks(size)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        get_chunks(
            chunks=chunks,
            query="improve http performance and security",
            heading="web backend",
            penalty=0.01,
            embedding_fn=fake_embedder,
        )
        samples.append(time.perf_counter() - start)
    return {
        "chunk_count": size,
        "iterations": iterations,
        "latency_ms_avg": statistics.fmean(samples) * 1000.0,
        "latency_ms_p95": statistics.quantiles(samples, n=20)[-1] * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    print(run(args.size, args.iterations))


if __name__ == "__main__":
    main()
