from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from domdistill.dom_split import split_dom


def _build_large_html(path: Path, repeat_factor: int) -> str:
    base = path.read_text(encoding="utf-8")
    # Repeat inside one root body to emulate a much larger scraping target.
    repeated = "\n".join(base for _ in range(repeat_factor))
    return f"<html><body>{repeated}</body></html>"


def run(iterations: int, html_content: str) -> dict[str, float]:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        split_dom(html_content)
        samples.append(time.perf_counter() - start)
    return {
        "iterations": iterations,
        "content_chars": len(html_content),
        "latency_ms_avg": statistics.fmean(samples) * 1000.0,
        "latency_ms_p95": statistics.quantiles(samples, n=20)[-1] * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--html-file", type=Path, default=Path("benchmarks/blog.html"))
    parser.add_argument("--repeat-factor", type=int, default=30)
    args = parser.parse_args()
    html_content = _build_large_html(args.html_file, args.repeat_factor)
    print(run(args.iterations, html_content))


if __name__ == "__main__":
    main()
