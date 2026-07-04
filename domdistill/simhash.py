from __future__ import annotations

from collections import Counter

import xxhash

VECTOR_SIZE = 64


def get_simhash(content: str, n: int = 2) -> int:
    words = content.split()
    if len(words) < n:
        return 0

    tokens = [" ".join(words[left : left + n]) for left in range(len(words) - n + 1)]
    # using frequency of shingles as weights(TF weighting -> Term Frequency)
    shingles_weight = Counter(tokens)
    vector = [0] * VECTOR_SIZE

    for token, weight in shingles_weight.items():
        token_hash = xxhash.xxh64_intdigest(token.encode())
        for bit in range(VECTOR_SIZE):
            if (token_hash >> bit) & 1:
                vector[bit] += weight
            else:
                vector[bit] -= weight

    fingerprint = 0
    for bit, value in enumerate(vector):
        if value > 0:
            fingerprint |= 1 << bit

    return fingerprint


def get_hamming_distance(simhash1: int, simhash2: int) -> int:
    return (simhash1 ^ simhash2).bit_count()


def get_similarity(h1: int, h2: int) -> float:
    return 1 - (get_hamming_distance(h1, h2) / VECTOR_SIZE)
