"""S4 -- Embed. A queue of its own apart from S3, because the provider's
chat and embedding quotas are separate things; one shared bucket would let
either starve the other.
"""

import io
import math

import numpy as np

from app.config import get_settings
from shared.llm import LLMProvider
from shared.rate_limiter import RateLimited, acquire_or_defer
from worker.steps.chunking import count_tokens

DIMENSIONS = get_settings().embed_dimensions


def contextualize(context: str, content: str) -> str:
    """What gets embedded -- and, in S5, stored as `contextualized`. One
    definition for both, or the stored text could drift away from the text
    its vector was actually computed from."""
    return f"{context}\n\n{content}".strip()


def vectors_to_npy(vectors: list[list[float]]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(vectors, dtype=np.float32))
    return buffer.getvalue()


def vectors_from_npy(data: bytes) -> list[list[float]]:
    return np.load(io.BytesIO(data)).tolist()


def assert_normalised(vectors: list[list[float]]) -> None:
    """Not cosmetic. This is the precondition for vector_ip_ops being
    correct."""
    for vector in vectors:
        norm = math.sqrt(sum(x * x for x in vector))
        assert abs(norm - 1.0) < 1e-4, f"vector is not L2-normalised (norm={norm})"


def embed_chunks(
    texts: list[str],
    provider: LLMProvider,
    batch_size: int = 100,
    done: list[list[float]] | None = None,
) -> list[list[float]]:
    """`done` is the vectors an earlier, deferred run already paid for -- the
    `partial` of its RateLimited, always a prefix of `texts` since batches
    go in order. Those texts are skipped, not embedded again."""
    vectors: list[list[float]] = list(done or [])

    try:
        for start in range(len(vectors), len(texts), batch_size):
            vectors.extend(_embed_batch(texts[start : start + batch_size], provider))
    except RateLimited as exc:
        # Whether the local limiter or the provider's own 429 said wait:
        # hand back the vectors already paid for, so the retry resumes.
        exc.partial = vectors
        raise
    return vectors


def _embed_batch(batch: list[str], provider: LLMProvider) -> list[list[float]]:
    # Counted, not guessed: len(text)//4 covers only ~48% of the real token
    # count of Vietnamese with diacritics, which would let real usage
    # overshoot the quota roughly twofold.
    tokens = sum(count_tokens(t) for t in batch)
    acquire_or_defer([("embed_rpm", 1), ("embed_tpm", tokens)])

    result = provider.embed(batch)

    assert len(result) == len(batch), f"got {len(result)} vectors for {len(batch)} texts"
    assert all(len(v) == DIMENSIONS for v in result), "wrong dimension"
    assert all(all(math.isfinite(x) for x in v) for v in result), "non-finite value"
    # Normalised here, not trusted: text-embedding-3-large shortened to 1536
    # dimensions returns norms like 1.00019. The assert after it is the
    # guard, catching what no division can fix -- an all-zero vector.
    result = [_unit(v) for v in result]
    assert_normalised(result)
    return result


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector
