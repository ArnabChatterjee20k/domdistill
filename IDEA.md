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


### What is the issue with this algo?
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