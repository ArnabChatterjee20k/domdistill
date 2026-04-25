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
python -m domdistill file benchmarks/blog.html --query "how http servers work" --section-index 0
```

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

print(result.score)
print(result.selected_chunks)
print(result.discarded_chunks)
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
```

Use fixed benchmark arguments when comparing revisions so regressions are meaningful.