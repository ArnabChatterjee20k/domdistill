from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from domdistill.scoring import get_score_for_chunk


def fake_embedder(text: str) -> np.ndarray:
    lowered = text.lower()
    return np.asarray(
        [
            float(lowered.count("http")),
            float(lowered.count("server")),
            float(lowered.count("cache")),
            float(lowered.count("database")),
            float(len(text)),
        ],
        dtype=float,
    )


def run(iterations: int) -> dict[str, float]:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        get_score_for_chunk(
            query="http server",
            heading="web infra",
            chunk="http servers accept requests and cache responses",
            penalty=0.001,
            embedding_fn=fake_embedder,
        )
        samples.append(time.perf_counter() - start)
    return {
        "iterations": iterations,
        "latency_ms_avg": statistics.fmean(samples) * 1000.0,
        "latency_ms_p95": statistics.quantiles(samples, n=20)[-1] * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    print(run(args.iterations))


if __name__ == "__main__":
    main()
