# domdistill

Intent-driven semantic chunk distillation for DOM/HTML content.

The library splits HTML into heading-aware sections, scores merged text chunks using
query and local-heading relevance, and uses dynamic programming to choose the best
set of selected chunks.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## CLI Usage

Run the synthetic demo:

```bash
python -m domdistill demo
```

Run distillation on a local HTML file:

```bash
python -m domdistill file benchmarks/blog.html --query "how http servers work" --top-k-chunks 10 --max-adjacent-chunks 6 --max-chunks-per-section 120
```

This always runs retrieval across the whole document and returns only top chunks.

## Library Usage

```python
from domdistill import HTMLIntentChunker

html_content = "<h1>Intro</h1><p>HTTP servers accept requests.</p><p>Caching helps.</p>"
chunker = HTMLIntentChunker(
    html_content,
    penalty=0.01,
    splitter_tags=("h1", "h2", "h3"),  # user-configurable
)
result = chunker.get_chunks("http server basics")

print(result.top_sections[0].heading)
print(result.top_chunks[0].content)
```

Whole-document retrieval with top-k:

```python
from domdistill import HTMLIntentChunker

chunker = HTMLIntentChunker(html_content, penalty=0.01)
multi = chunker.get_chunks(
    "http server basics",
    top_k_chunks=10,
    max_adjacent_chunks=6,
    max_chunks_per_section=120,
)
print(multi.top_sections[0].heading)
print(multi.top_chunks[0].content)
```

Performance knobs:
- `max_adjacent_chunks`: limits how many neighboring chunks can be merged into one candidate.
  Lower values reduce runtime/memory (`1` means no merge across chunk boundaries).
- `max_chunks_per_section`: caps how many chunks are considered inside each section.
  Useful for very large/noisy pages.
- `max_merge_span`: legacy alias for `max_adjacent_chunks` (still accepted for backward compatibility).

## Discrete Building Blocks (and Glueing)

Use the high-level `HTMLIntentChunker` when you want a single entrypoint.
If you need custom behavior, you can use lower-level modules directly and glue
them together in your own pipeline.

### 1) Split HTML into sections

```python
from domdistill.dom_split import split_dom

sections = split_dom(
    html_content,
    splitter_tags=("h1", "h2", "h3"),  # custom heading tags
    cache_dir=".cache/dom_split",      # optional deterministic cache
)
```

### 2) Score/select chunks for one section

```python
from domdistill.selection import select_chunks

section = sections[0]
chunks = [node.content for node in section.nodes if node.content.strip()]
selection = select_chunks(
    chunks=chunks,
    query="http server basics",
    heading=section.heading.content,
    penalty=0.01,
)
print(selection.score)
print(selection.selected_chunks)
```

### 3) Custom embedder injection (for tests or custom models)

```python
import numpy as np
from domdistill.selection import select_chunks

def my_embedder(text: str) -> np.ndarray:
    # Plug in your own model/provider here.
    return np.asarray([float(len(text))], dtype=float)

selection = select_chunks(
    chunks=["a", "b", "c"],
    query="sample",
    heading="intro",
    penalty=0.01,
    embedding_fn=my_embedder,
)
print(selection.selected_chunks)
```

### 4) Glue everything across all sections

```python
from domdistill.dom_split import split_dom
from domdistill.selection import select_chunks

results = []
for idx, section in enumerate(split_dom(html_content, splitter_tags=("h1", "h2"))):
    chunks = [node.content for node in section.nodes if node.content.strip()]
    if not chunks:
        continue
    selection = select_chunks(
        chunks=chunks,
        query="what should I extract?",
        heading=section.heading.content,
        penalty=0.01,
    )
    results.append(
        {
            "section_index": idx,
            "heading": section.heading.content,
            "score": selection.score,
            "selected": selection.selected_chunks,
            "discarded": selection.discarded_chunks,
        }
    )
```

Use this discrete approach when you need custom ranking logic, extra metadata,
or post-processing that differs from the default high-level API behavior.

## Testing

Run default unit tests:

```bash
pytest
```

Run integration tests (loads embedding model):

```bash
pytest -m integration
```

## Benchmarking

Benchmark split, scoring, and selection:

```bash
python benchmarks/bench_split.py --html-file benchmarks/blog.html --repeat-factor 30 --iterations 30
python benchmarks/bench_score.py --iterations 1000
python benchmarks/bench_select.py --size 12 --iterations 50
python benchmarks/bench_chunker_pool.py --html-file benchmarks/blog.html --repeat-factor 20 --warmup-runs 1 --iterations 3 --pool-sizes 1,2,4,8 --batch-sizes 32,128
```

Use fixed benchmark arguments when comparing revisions so regressions are meaningful.
Use at least one warmup run when measuring the default embedder, because first use includes
model loading.

CI regression guard is enabled for baseline config (`pool_size=1`, `batch_size=32`) and reports:
- `bad` (fails CI)
- `stable`
- `small improvement`
- `great improvement`

### Retrieval Quality Eval (long blog)

Evaluate whether correct chunks are retrieved for labeled queries on a long blog fixture:

```bash
python benchmarks/eval_retrieval.py --html-file benchmarks/blog.html --cases-file benchmarks/eval_cases.json --penalty 0.01
```

This reports:
- `macro_precision`: fraction of returned chunks that match expected phrases.
- `macro_recall`: fraction of expected phrases successfully retrieved.
- `macro_wrong_merge_rate`: selected chunks that mix expected and forbidden-topic cues.

Tune `penalty` and compare these metrics across revisions.

### Thread Pool Control

`HTMLIntentChunker.get_chunks(...)` and CLI support `pool_size` / `--pool-size`:

```bash
python -m domdistill file benchmarks/blog.html --query "concurrency" --top-k-chunks 5 --pool-size 4 --batch-size 25
```

Rules:
- With the default SentenceTransformer embedder, the high-level chunker batches embeddings
  across the whole document in one model call.
- For the default embedder, `batch_size` is passed to `SentenceTransformer.encode(...)`.
- For the default embedder, `pool_size` is only used when process-pool DP is explicitly
  enabled with `use_pool=True` / `--use-pool` and the section threshold is exceeded.
- With a custom `embedding_fn`, `pool_size>1` uses thread-based section parallelism.
- Custom `embedding_fn` works with thread pool; there is no pickling/serialization requirement.
- `--max-adjacent-chunks` is the primary merge-window option; `--max-merge-span` is a legacy alias.
- `--max-chunks-per-section` prevents candidate explosion on large sections.

Optional process-pool DP:

```bash
python -m domdistill file benchmarks/blog.html \
  --query "concurrency" \
  --top-k-chunks 5 \
  --pool-size 4 \
  --batch-size 32 \
  --use-pool \
  --concurrency-section-threshold 301
```

`--concurrency-section-threshold 0` keeps the current serial DP path after document-level
embedding. When `--use-pool` is set and the number of sections is greater than the threshold,
the chunker still batches embeddings once, then sends only precomputed section scores to a
process pool for DP evaluation. This is intended for pages with enough section-level DP work
to offset process serialization overhead.

Current optimization shape:

```text
split DOM into sections
build contiguous merged candidates per section
embed [query, all headings, all unique candidates] as one document-level batch
score candidates from cached embeddings
run DP per section over precomputed numeric scores, optionally via process pool
rank selected chunks using already-computed scores
```

This avoids the old bottleneck where query, heading, and merged chunks were embedded inside
the DP candidate loop. The DP still evaluates `n * (n + 1) / 2` contiguous candidates per
section, but embedding is no longer repeated per candidate.

Measured locally on `benchmarks/blog.html`:

```text
repeat_factor=3, batch_size=32, warmed:
  best -> about 0.58s

repeat_factor=20, batch_size=32, warmed:
  best -> about 0.52s
```

Benchmark numbers depend on CPU, torch runtime settings, and whether the model is already
loaded, so treat these as local reference numbers rather than fixed guarantees.

### Concurrency Tuning Guide

Start with the default path:

```bash
python -m domdistill file benchmarks/blog.html \
  --query "concurrency" \
  --top-k-chunks 5 \
  --batch-size 32
```

Use this when:
- you are using the built-in SentenceTransformer embedder.
- the page has many small or medium sections.
- you want the lowest overhead path.

Tune `batch_size` first. It controls the model batch size for the document-level
`SentenceTransformer.encode(...)` call. Larger batches can improve throughput, but they also
increase peak memory use. Try `32`, `64`, `128`, and compare warmed benchmark runs.

Use `pool_size` without `--use-pool` only for custom embedders:

```bash
python -m domdistill file page.html \
  --query "..." \
  --pool-size 4
```

This uses thread-based section parallelism when a custom `embedding_fn` is supplied from the
library API. It is useful when the custom embedder is slow per section or does I/O. It is not
the main speed knob for the default embedder.

Use process-pool DP only when DP itself is large enough:

```bash
python -m domdistill file page.html \
  --query "..." \
  --pool-size 4 \
  --use-pool \
  --concurrency-section-threshold 301
```

This keeps scoring batched in the parent process, then sends precomputed section scores to
worker processes for DP evaluation. It can help when:
- the document has more sections than `--concurrency-section-threshold`.
- sections are large enough to produce many contiguous merge candidates.
- profiling shows DP time is meaningful after embeddings are batched.

It can hurt when:
- sections are tiny.
- the page has many sections but only a few nodes per section.
- serialization of `chunks`, `merged_by_span`, and `score_by_chunk` costs more than the DP.

Practical defaults:
- `batch_size=32`: conservative starting point.
- `pool_size=1`: best default for built-in embeddings.
- `--concurrency-section-threshold 0`: disables process-pool DP.
- Try `--use-pool --pool-size 4 --concurrency-section-threshold 300` only after the default
  path is measured and DP is suspected to dominate.

Fastest small/medium HTML path from Python:

```python
from domdistill import HTMLIntentChunker

chunker = HTMLIntentChunker(
    html_content,
    penalty=0.01,
)

result = chunker.get_chunks(
    "what should I extract?",
    top_k_chunks=5,
    pool_size=1,
    batch_size=32,
    use_pool=False,
    concurrency_section_threshold=0,
)
```

This is usually the fastest path for small and medium HTML documents when using the built-in
embedder. It still batches scoring across all sections in one document-level
`SentenceTransformer.encode(...)` call, but it avoids thread/process scheduling and
serialization overhead for the DP phase. After embeddings are cached, section DP is mostly
Python dict lookups and float comparisons, so parallelizing it only helps when sections are
large enough to make DP itself expensive.
