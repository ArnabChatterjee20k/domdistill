from collections import Counter
import xxhash
VECTOR_SIZE = 64
def get_simhash(content: str, n:int = 2):
    words = list(filter(lambda c: len(c) > 0, content.split(" ")))
    tokens = []
    for right in range(n,len(words)+1):
        left = right-n
        tokens.append(" ".join(words[left:right]))
    
    # using frequency of shingles as weights(TF weighting -> Term Frequency)
    shingles_weight = Counter(tokens)
    vector = [0]*VECTOR_SIZE
    for token, weight in shingles_weight.items():
        token_hash = xxhash.xxh64(token).intdigest()
        for bit in range(VECTOR_SIZE):
            mask = 1 << bit
            if token_hash & mask:
                vector[bit] += weight
            else:
                vector[bit] -= weight
    
    fingerprint = 0
    for bit, value in enumerate(vector):
        if value > 0:
            fingerprint |= 1<<bit
    
    return fingerprint
tests = [
    ["a", "b", "c"],
    ["a", "b", "c", "d"],
    ["a", "b", "c", "d", "e"]
]

def get_hamming_distance(simhash1, simhash2):
    return (simhash1 ^ simhash2).bit_count()

def get_similarity(h1, h2):
    return 1 - (get_hamming_distance(h1, h2) / VECTOR_SIZE)