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


@dataclass(frozen=True)
class SectionFingerprint:
    heading: str
    hash: int


@dataclass(frozen=True)
class DocumentFingerprint:
    """
        produces like this
        {
            "document_hash": "...",
            "sections": [
                {
                    "heading": "Authentication",
                    "hash": "..."
                },
                {
                    "heading": "Storage",
                    "hash": "..."
                }
            ]
        }
        hash can be like simhash
        where document_hash is n-gram based tokenization
        and each section will have their own hash for content tracking over time
    """
    document_hash: int
    sections: list[SectionFingerprint]