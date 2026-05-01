Intent driven chunking for a dom
We will take each nodes and will do combinations on each nodes
Example
```
nodes = [n1,n2,n3]
So a good intent chunk is which either answers the query or the local heading
So a good chunk is n1+n2+n3 or n1+n2 , n3 or n1+n3,n2, ... 2^3 combinations
Here comes the dp optimisation
```
Steps
1. Splitting of the dom based on h1, h2, h3, etc
2. Each will have nodes under it. They are the DP units/chunks
3. score(Chunk) = max(
    sim(Chunk, query),      # global intent
    sim(Chunk, heading)     # local intent
)
4. Lets add a penalty as well so that score doesn't deviate a lot
score(C) =
    max(sim_query, sim_heading)
    - λ * (length(C) ** 2)
5. total_score =
    sum(score(C_i)) - β * num_chunks

6. So a resultant chunk would have this structure
```
{
  "content": "...",
  "heading": "...",
  "sim_query": 0.82,
  "sim_heading": 0.91,
  "density": 0.76,
  "position": 0.3
}
```

dp[i] = best score till node i
dp[j] = max over i < j:
    dp[i] + score(nodes[i+1 → j]) - β

### Why not any already present solutions?
I am trying to a build generalized solution for info gather through web scraping. And for this I need some heuristic formulation as well. So this sematic distillation helps

### Why threadpool instead of the processpool?
Process pool was giving a high overhead even after batching. So pool with size 1 and batching was giving better performance as we needed to load model atleast once in each processes
Using thread pool is loading the model only once as the memory is shared among threads unlike process and spinning thread is cheap. Giving better benchmarks.
Even with prewarming the model with initializer inside process pool didn't solve it.
See the state now https://github.com/ArnabChatterjee20k/domdistill/blob/3ef3328d5ab78c84c7d1fad1e90f30c54a126cdf/domdistill/chunker.py

Though dp is present but still that is more embedding heavy. So defaulting thread pool for now only\

No overhead of GIL lock as libs like sentence transformers are precompiled(built outside of python)

### Optimization made

The main bottleneck was not the DP table itself, but where the embedding calls were placed.
The previous path did this inside the DP candidate loop:

```
for every section:
  for every start node i:
    for every end node j:
      merged = nodes[i..j]
      embed(query)
      embed(heading)
      embed(merged)
      score(merged)
```

For a section with `n` nodes, the DP considers `n * (n + 1) / 2` contiguous merged chunks.
The old implementation effectively made at least `3 * n * (n + 1) / 2` embedding calls per
section because query, heading, and merged chunk were embedded for each candidate. It also
rescored the selected chunks later for ranking, which added another embedding pass.

The optimized path now does:

```
for every section:
  build all contiguous merged candidates once
  embed([query, heading, *candidates]) in one batch
  store score(candidate) in a dict
  run DP using precomputed numeric scores
  return selected chunks with their already-computed scores
```

So the algorithm still uses the same DP recurrence and still evaluates the same candidate
space, but embedding is moved out of the inner DP loop. This changes the hot path from many
small model calls to one batched model call per section. The DP then becomes cheap numeric
lookup work:

```
dp[i] = best score from node i to the end

dp[i] = max(
  dp[i + 1] - discard_penalty,
  score(nodes[i..j]) + dp[j + 1] - chunk_penalty
)
```

The chunker also no longer runs a second scoring pass over `selected_chunks`. `select_chunks`
returns `selected_scores`, and ranking reuses those scores directly.

### Further optimization: document-level embedding batch

After the first optimization, the default embedder still did one `encode()` call per section.
That was much better than embedding inside the DP loop, but large pages can have hundreds of
small sections, so the model was still being called hundreds of times.

The current default path now prepares the whole document first:

```
for every section:
  build section candidates
  collect heading
  collect candidates

unique_texts = [query, *all_headings, *all_candidates]
embeddings = embed(unique_texts)  # one document-level batch

for every section:
  compute score(candidate) from cached embeddings
  run DP using precomputed scores
```

This keeps the same DP and scoring semantics, but changes the embedding shape again:

```
old initial path:
  many embedding calls inside every DP candidate

first optimization:
  one embedding batch per section

current optimization:
  one embedding batch per document
```

For the default SentenceTransformer embedder, `pool_size` now matters much less because the
main model work happens in one large batch. The model/PyTorch runtime can use CPU threads
internally. Python-level section fanout is still available for custom embedders, but it is no
longer the fastest path for the default model.

### Optional process-pool DP after scoring

> Idea is since the underlying pytorch already using the cpu cores by the batch so we can compute the scores in the batch format directly in a single thread and no thread overhead. DP overhead is very low as we have the precomputed values already

There is now an optional pool path for only the DP phase:

```
batch embeddings for the document
compute score_by_chunk per section

if use_pool and section_count > concurrency_section_threshold:
  send precomputed section scores to process pool
  run DP per section in worker processes
else:
  run DP serially in the parent process
```

Important: model scoring is still always batched before this step. Worker processes do not
load SentenceTransformer and do not recompute embeddings. They only receive `chunks`,
`merged_by_span`, and `score_by_chunk`.

`concurrency_section_threshold = 0` disables this path and keeps the current serial DP after
document-level batching. A threshold like `301` means process-pool DP is used only when the
document has more than 301 sections.

On the current benchmark fixture this is slower because each section is small, so process
serialization overhead is larger than the DP work. This path is mainly for pages with many
sections where each section also has enough candidates to make DP evaluation meaningful.

Measured impact on the benchmark page:

```
repeat_factor=3, batch_size=32

before:
  pool_size=1 -> 10.96s
  pool_size=2 -> 11.66s
  pool_size=4 -> 12.69s

after:
  pool_size=1 -> 5.81s
  pool_size=2 -> 1.21s
  pool_size=4 -> 0.95s

after document-level batch, warmed:
  best -> 0.58s
```

For a larger page:

```
repeat_factor=20, batch_size=32

pool_size=1 -> 18.11s
pool_size=2 -> 8.77s
pool_size=4 -> 6.46s
pool_size=8 -> 7.12s

after document-level batch, warmed:
  best -> 0.52s
```

In the per-section batching version, `pool_size=4` was best and `pool_size=8` was slower,
likely because the shared CPU model work started hitting scheduling and compute contention.
With document-level batching, pool-size differences are mostly noise for the default embedder,
because the hot path is no longer a section worker loop.

Time complexity:
O(n³)            ← string merging
+ O(n² * C)      ← scoring (dominant in practice)
+ O(n²)          ← DP

≈ O(n³ + n² * C)

Space complexity:
O(n²) ← merged_by_span
O(n²) ← score_by_chunk
O(n)  ← memo

≈ O(n²)

# Todos

### Change heuristics for this algo?
What if the local maxima is higher than the global maxima?
Example in the benchmarks blog.html, its all about http servers but I am running this
```bash
python -m domdistill file benchmarks/blog.html --query "thread" --top-k-chunks  1
```
Results
```
  "query": "thread",
  "top_chunks": [
    {
      "content": "An HTTP server is a system that listens for incoming HTTP requests from clients (usually browsers) and returns responses. These responses can be HTML pages, JSON data, images, or other resources.",
      "score": 0.8891703701946699,
      "heading": "What is an HTTP Server?",
      "section_index": 2
    }
  ]
```

Cause of the local maxima here -> the max is getting is too high

### Adding a separate discard tag type in the get_chunks
Currently tags like code, table, etc are added in the chunks of a section and not removed if they increase score a bit.
So we might want to strip them as a part of discarded chunk.
And also update the get_chunks to use get score on dicarded chunk
