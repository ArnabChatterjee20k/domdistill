from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from domdistill.chunker import HTMLIntentChunker


def _contains_any(haystack: str, needles: list[str]) -> bool:
    lowered = haystack.lower()
    return any(needle.lower() in lowered for needle in needles)


def evaluate_case(chunker: HTMLIntentChunker, case: dict) -> dict:
    multi_result = chunker.get_chunks(
        query=case["query"],
        top_k_chunks=5,
    )
    if not multi_result.top_sections:
        raise ValueError("No sections available for evaluation.")
    best_section = multi_result.top_sections[0]
    chosen_section_index = best_section.section_index

    expected = case["expected_substrings"]
    forbidden = case.get("forbidden_substrings", [])

    expected_hits = 0
    selected_hits = 0

    for phrase in expected:
        if _contains_any(" ".join(best_section.selected_chunks), [phrase]):
            expected_hits += 1

    for selected in best_section.selected_chunks:
        if _contains_any(selected, expected):
            selected_hits += 1

    precision = (
        selected_hits / len(best_section.selected_chunks)
        if best_section.selected_chunks
        else 0.0
    )
    recall = expected_hits / len(expected) if expected else 0.0

    # Wrong merge rate: selected chunks that include relevant content plus forbidden-topic cues.
    wrong_merge_count = 0
    for selected in best_section.selected_chunks:
        has_expected = _contains_any(selected, expected)
        has_forbidden = _contains_any(selected, forbidden)
        if has_expected and has_forbidden:
            wrong_merge_count += 1

    wrong_merge_rate = (
        wrong_merge_count / len(best_section.selected_chunks)
        if best_section.selected_chunks
        else 0.0
    )

    return {
        "case_id": case["id"],
        "heading": best_section.heading,
        "section_index": chosen_section_index,
        "score": best_section.score,
        "selected_count": len(best_section.selected_chunks),
        "precision": precision,
        "recall": recall,
        "wrong_merge_rate": wrong_merge_rate,
    }


def run_eval(html_file: Path, cases_file: Path, penalty: float) -> dict:
    html_content = html_file.read_text(encoding="utf-8")
    cases_payload = json.loads(cases_file.read_text(encoding="utf-8"))

    chunker = HTMLIntentChunker(
        html_content,
        penalty=penalty,
        splitter_tags=("h1", "h2", "h3"),
    )

    case_results = [evaluate_case(chunker, case) for case in cases_payload["cases"]]
    precision_values = [result["precision"] for result in case_results]
    recall_values = [result["recall"] for result in case_results]
    wrong_merge_values = [result["wrong_merge_rate"] for result in case_results]

    return {
        "html_file": str(html_file),
        "cases_file": str(cases_file),
        "penalty": penalty,
        "cases_evaluated": len(case_results),
        "macro_precision": statistics.fmean(precision_values)
        if precision_values
        else 0.0,
        "macro_recall": statistics.fmean(recall_values) if recall_values else 0.0,
        "macro_wrong_merge_rate": statistics.fmean(wrong_merge_values)
        if wrong_merge_values
        else 0.0,
        "case_results": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate long-blog retrieval quality."
    )
    parser.add_argument("--html-file", type=Path, default=Path("benchmarks/blog.html"))
    parser.add_argument(
        "--cases-file", type=Path, default=Path("benchmarks/eval_cases.json")
    )
    parser.add_argument("--penalty", type=float, default=0.01)
    args = parser.parse_args()
    result = run_eval(args.html_file, args.cases_file, args.penalty)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
