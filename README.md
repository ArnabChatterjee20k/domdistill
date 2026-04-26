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
python -m domdistill file benchmarks/blog.html --query "how http servers work" --top-k-chunks 10
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
)
print(multi.top_sections[0].heading)
print(multi.top_chunks[0].content)
```

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
from domdistill.selection import get_chunks

section = sections[0]
chunks = [node.content for node in section.nodes if node.content.strip()]
score, selected, discarded = get_chunks(
    chunks=chunks,
    query="http server basics",
    heading=section.heading.content,
    penalty=0.01,
)
```

### 3) Custom embedder injection (for tests or custom models)

```python
import numpy as np
from domdistill.selection import get_chunks

def my_embedder(text: str) -> np.ndarray:
    # Plug in your own model/provider here.
    return np.asarray([float(len(text))], dtype=float)

score, selected, discarded = get_chunks(
    chunks=["a", "b", "c"],
    query="sample",
    heading="intro",
    penalty=0.01,
    embedding_fn=my_embedder,
)
```

### 4) Glue everything across all sections

```python
from domdistill.dom_split import split_dom
from domdistill.selection import get_chunks

results = []
for idx, section in enumerate(split_dom(html_content, splitter_tags=("h1", "h2"))):
    chunks = [node.content for node in section.nodes if node.content.strip()]
    if not chunks:
        continue
    score, selected, discarded = get_chunks(
        chunks=chunks,
        query="what should I extract?",
        heading=section.heading.content,
        penalty=0.01,
    )
    results.append(
        {
            "section_index": idx,
            "heading": section.heading.content,
            "score": score,
            "selected": selected,
            "discarded": discarded,
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
python benchmarks/bench_chunker_pool.py --html-file benchmarks/blog.html --repeat-factor 20 --warmup-runs 1 --iterations 5 --pool-sizes 1,2,4 --batch-sizes 32,128
```

Use fixed benchmark arguments when comparing revisions so regressions are meaningful.

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

### Process Pool Control

`HTMLIntentChunker.get_chunks(...)` and CLI support `pool_size` / `--pool-size`:

```bash
python -m domdistill file benchmarks/blog.html --query "concurrency" --top-k-chunks 5 --pool-size 4 --batch-size 25
```

Rules:
- `pool_size=1` runs serially.
- `pool_size>1` uses multiprocessing.
- If `embedding_fn` is provided, `pool_size` must stay `1` (otherwise an error is raised).
- `batch_size` controls how many chunk scoring items are submitted per process-pool task.
- Embedding-stage rescoring uses SentenceTransformer `encode_multi_process` (default embedder path).
- `pool_size` controls non-embedding task parallelism (section-level selection workers).

Cost tradeoff as `pool_size` increases:
- Each worker process loads its own embedding model instance (not shared in-memory across processes).
- Higher `pool_size` increases startup/warmup time and RAM usage.
- Throughput can improve on larger workloads, but small workloads may become slower due to process/model overhead.