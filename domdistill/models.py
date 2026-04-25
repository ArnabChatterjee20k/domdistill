from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    tag: str
    content: str


@dataclass(frozen=True)
class SplittedDomNodes:
    heading: Node
    nodes: list[Node]


@dataclass(frozen=True)
class Chunk:
    heading: str
    content: str
    sim_query: float
    sim_heading: float
    density: float = 0.0
    position: float = 0.0