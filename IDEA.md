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
```

For a larger page:

```
repeat_factor=20, batch_size=32

pool_size=1 -> 18.11s
pool_size=2 -> 8.77s
pool_size=4 -> 6.46s
pool_size=8 -> 7.12s
```

`pool_size=4` was best in this run. `pool_size=8` was slower, likely because the shared CPU
model work starts hitting scheduling and compute contention. So "use all cores" should still
be tuned by benchmark, not assumed to mean maximum worker count.

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
