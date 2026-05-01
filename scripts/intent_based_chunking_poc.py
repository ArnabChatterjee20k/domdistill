from sentence_transformers import SentenceTransformer
import os
import numpy as np
from dataclasses import dataclass
from typing import List
from lxml import html


@dataclass
class Node:
    tag: str
    content: str


@dataclass
class SplittedDomNodes:
    heading: Node
    nodes: List[Node]


@dataclass
class Chunk:
    heading: str
    content: str
    sim_query: float
    sim_heading: float
    density: float
    position: float


def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    save_dir: str = "./models/embeddings",
    device: str = "cpu",
):
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_dir, "config.json")):
        print(f"Loading embedding model from local: {save_dir}")
        model = SentenceTransformer(save_dir, device=device)
    else:
        print(f"Downloading embedding model: {model_name}")
        model = SentenceTransformer(model_name, device=device)
        model.save(save_dir)
        print(f"Saved embedding model to: {save_dir}")

    return model


def get_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(text):
    model = load_embedding_model()
    return model.encode(text)


splitter = ["h1", "h2", "h3", "h4", "h5", "h6"]


def cache(fn):
    import pickle
    from pathlib import Path

    def wrapper(*args, **kwargs):
        path = Path("cache.pkl")
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        data = fn(*args, **kwargs)
        with open(path, "wb") as f:
            pickle.dump(data, f)
            return data

    return wrapper


@cache
def split_dom(html_content):
    tree = html.fromstring(html_content)
    dom_nodes: List[Node] = []

    for element in tree.iter():
        tag = getattr(element, "tag", "")
        if not isinstance(tag, str):
            continue

        # Use only direct text to avoid parent nodes duplicating all child text.
        text = " ".join(
            part.strip() for part in element.xpath("./text()") if part and part.strip()
        )
        if not text:
            continue

        dom_nodes.append(Node(tag=tag.lower(), content=text))

    sections: List[SplittedDomNodes] = []
    current_heading = Node(tag="root", content="root")
    current_nodes: List[Node] = []

    for node in dom_nodes:
        if node.tag in splitter:
            if current_nodes:
                sections.append(
                    SplittedDomNodes(heading=current_heading, nodes=current_nodes)
                )
            current_heading = node
            current_nodes = []
        else:
            current_nodes.append(node)

    if current_nodes:
        sections.append(SplittedDomNodes(heading=current_heading, nodes=current_nodes))

    return sections


from functools import cache as fn_cache


@fn_cache
def get_score_for_chunk(query, heading, chunk, penalty=0.0001):

    query_embedding = get_embedding(query)
    heading_embedding = get_embedding(heading)
    chunk_embedding = get_embedding(chunk)

    # getting similarity score for the chunk with respect to the query and local-heading
    query_simarity_score = get_cosine_similarity(query_embedding, chunk_embedding)
    heading_similarity_score = get_cosine_similarity(heading_embedding, chunk_embedding)

    # Use sub-linear length penalty so long chunks are discouraged but not wiped out.
    length_penalty = penalty * np.sqrt(max(len(chunk), 1))
    return max(query_simarity_score, heading_similarity_score) - length_penalty


def subarray(chunks, start):
    if not chunks or len(chunks) < start:
        return []

    return chunks[start:]


# the intelligent layer
def get_best_chunks_only(
    i=0, chunks: list[str] | None = None, query="", heading="", penalty=0.0001
) -> tuple[float, list[str]]:
    if chunks is None:
        chunks = []

    def helper(i):
        if i == len(chunks):
            return 0.0, []

        current_best = (-float("inf"), [])
        current_best_chunks = []
        for j in range(i, len(chunks)):
            current_best_chunks.append(chunks[j])
            current_best_chunks_merged = " ".join(current_best_chunks)
            current_chunk_score = get_score_for_chunk(
                query, heading, current_best_chunks_merged, penalty
            )

            rest_score, rest_segment = helper(j + 1)
            total_score_in_current_node = current_chunk_score + rest_score - penalty

            if total_score_in_current_node > current_best[0]:
                current_best = (
                    total_score_in_current_node,
                    [current_best_chunks_merged, *rest_segment],
                )
        return current_best

    return helper(i)


# returns best chunks merged in a way + orphan chunks as well
def get_chunks(
    i=0, chunks: list[str] | None = None, query="", heading="", penalty=0.0001
) -> tuple[float, list[str], list[str]]:
    if chunks is None:
        chunks = []
    memo: dict[int, tuple[float, list[str], list[str]]] = {}

    def helper(i) -> tuple[float, list[str], list[str]]:
        if i == len(chunks):
            return 0.0, [], []
        if i in memo:
            return memo[i]

        # score, selected, discarded
        current_best: tuple[float, list[str], list[str]] = (-float("inf"), [], [])

        # path1 -> dont include the node
        rest_score, rest_segment, rest_discarded = helper(i + 1)
        discard_penalty = penalty
        total_score_if_discard = rest_score - discard_penalty
        if total_score_if_discard > current_best[0]:
            current_best = (
                total_score_if_discard,
                rest_segment,
                [chunks[i], *rest_discarded],
            )

        # path2 -> include the node
        current_best_chunks = []
        for j in range(i, len(chunks)):
            current_best_chunks.append(chunks[j])
            current_best_chunks_merged = " ".join(current_best_chunks)
            current_chunk_score = get_score_for_chunk(
                query, heading, current_best_chunks_merged, penalty
            )

            rest_score, rest_segment, rest_discarded = helper(j + 1)
            total_score_in_current_node = current_chunk_score + rest_score - penalty

            if total_score_in_current_node > current_best[0]:
                current_best = (
                    total_score_in_current_node,
                    [current_best_chunks_merged, *rest_segment],
                    rest_discarded,
                )
        memo[i] = current_best
        return memo[i]

    best_score, selected_chunks, discarded_chunks = helper(i)
    return best_score, selected_chunks, discarded_chunks


def main():
    with open("blog.html", "r") as file:
        # html_content = file.read()
        # sections = split_dom(html_content)
        from pprint import pprint

        # if not sections:
        #     print("No sections found in document.")
        #     return

        # first_section = sections[0]
        # chunks = [node.content for node in first_section.nodes if node.content.strip()]

        # if not chunks:
        #     print("No chunks found in the first section.")
        #     return

        # score, selected_chunks, discarded_chunks = get_chunks(
        #     chunks=chunks,
        #     query="database",
        #     heading=first_section.heading.content,
        # )

        # print("get_chunks result for first normal section:")
        # pprint({
        #     "heading": first_section.heading.content,
        #     "score": score,
        #     "selected_chunks": selected_chunks,
        #     "discarded_chunks": discarded_chunks,
        # })

        print("\nsynthetic discard simulation:")
        synthetic_chunks = [
            "HTTP servers accept requests and return responses.",
            "lorem ipsum " * 1200,
            "Key takeaway: caching and HTTPS improve reliability and security.",
            "databases are a key part of the world",
            "i am arnab",
        ]
        sim_score, sim_selected, sim_discarded = get_chunks(
            chunks=synthetic_chunks,
            query="how http servers work",
            heading="Key Takeaways",
            penalty=0.05,
        )
        pprint(
            {
                "score": sim_score,
                "selected_count": len(sim_selected),
                "discarded_count": len(sim_discarded),
                "selected_chunks": sim_selected,
                "discarded_preview": [
                    chunk[:120] + "..." if len(chunk) > 120 else chunk
                    for chunk in sim_discarded
                ],
            }
        )


main()
