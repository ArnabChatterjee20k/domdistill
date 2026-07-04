from __future__ import annotations

import pytest

from domdistill.simhash import (
    VECTOR_SIZE,
    get_hamming_distance,
    get_similarity,
    get_simhash,
)


def test_get_simhash_is_deterministic():
    content = "the quick brown fox jumps over the lazy dog"
    assert get_simhash(content) == get_simhash(content)


def test_get_simhash_returns_zero_for_insufficient_tokens():
    assert get_simhash("") == 0
    assert get_simhash("single") == 0
    assert get_simhash("one two", n=3) == 0


def test_get_simhash_changes_when_content_changes():
    base = "alpha beta gamma delta epsilon"
    changed = "alpha beta gamma delta zeta"
    assert get_simhash(base) != get_simhash(changed)


def test_get_simhash_respects_ngram_size():
    content = "one two three four"
    assert get_simhash(content, n=2) != get_simhash(content, n=3)


def test_get_simhash_normalizes_whitespace():
    assert get_simhash("a  b   c") == get_simhash("a b c")


def test_get_hamming_distance_identical_hashes():
    fingerprint = get_simhash("hello world again")
    assert get_hamming_distance(fingerprint, fingerprint) == 0


def test_get_hamming_distance_is_symmetric():
    left = get_simhash("authentication tokens expire quickly")
    right = get_simhash("authentication tokens remain valid")
    assert get_hamming_distance(left, right) == get_hamming_distance(right, left)


def test_get_similarity_bounds():
    left = get_simhash("storage layer uses redis cache")
    right = get_simhash("storage layer uses redis cache")
    assert get_similarity(left, right) == pytest.approx(1.0)

    completely_different_left = get_simhash("alpha beta gamma delta")
    completely_different_right = get_simhash("wxyz lmnop qrst uvwx")
    assert 0.0 <= get_similarity(
        completely_different_left, completely_different_right
    ) <= 1.0


def test_similar_content_has_higher_similarity_than_unrelated_content():
    original = "user authentication requires secure password hashing"
    lightly_edited = "user authentication requires secure password storage"
    unrelated = "database indexes improve read query performance"

    original_hash = get_simhash(original)
    edited_hash = get_simhash(lightly_edited)
    unrelated_hash = get_simhash(unrelated)

    edited_similarity = get_similarity(original_hash, edited_hash)
    unrelated_similarity = get_similarity(original_hash, unrelated_hash)
    assert edited_similarity > unrelated_similarity


def test_get_similarity_uses_vector_size():
    assert get_similarity(0, (1 << VECTOR_SIZE) - 1) == pytest.approx(0.0)
