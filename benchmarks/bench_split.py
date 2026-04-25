from __future__ import annotations

import argparse
import statistics
import time

from domdistill.dom_split import split_dom


HTML_SAMPLE = """
<html><body>
<h1>Intro</h1><p>HTTP servers process requests.</p><p>Caching helps.</p>
<h2>Database</h2><p>Indexes improve query speed.</p><p>Transactions keep consistency.</p>
<h2>Security</h2><p>TLS encrypts traffic.</p><p>Authz controls access.</p>
</body></html>
"""


def run(iterations: int) -> dict[str, float]:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        split_dom(HTML_SAMPLE)
        samples.append(time.perf_counter() - start)
    return {
        "iterations": iterations,
        "latency_ms_avg": statistics.fmean(samples) * 1000.0,
        "latency_ms_p95": statistics.quantiles(samples, n=20)[-1] * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    print(run(args.iterations))


if __name__ == "__main__":
    main()
