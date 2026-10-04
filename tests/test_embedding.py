import math
from collections import Counter

import pytest

from shared.llm import StubProvider
from shared.rate_limiter import RateLimited
from worker.chunking import count_tokens
from worker.embedding import assert_normalised, contextualize, embed_chunks


def test_every_vector_has_the_configured_dimension():
    vectors = embed_chunks(["a", "b", "c"], provider=StubProvider())
    assert all(len(v) == 1536 for v in vectors)


def test_vectors_are_l2_normalised():
    """vector_ip_ops assumes unit vectors. Skip normalising and the index
    returns the WRONG neighbours with no error at all -- so the assert lives
    in the code, not only in this test."""
    for vector in embed_chunks(["xin chào"], provider=StubProvider()):
        assert abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < 1e-6


def test_assert_normalised_rejects_an_unnormalised_vector():
    with pytest.raises(AssertionError):
        assert_normalised([[3.0] + [0.0] * 1535])


def test_a_short_response_is_caught():
    class ShortStub(StubProvider):
        def embed(self, texts):
            return super().embed(texts)[:-1]

    with pytest.raises(AssertionError):
        embed_chunks(["a", "b"], provider=ShortStub())


def test_batching_respects_the_batch_size():
    class CountingStub(StubProvider):
        calls = 0

        def embed(self, texts):
            type(self).calls += 1
            return super().embed(texts)

    embed_chunks([f"t{i}" for i in range(250)], provider=CountingStub(), batch_size=100)
    assert CountingStub.calls == 3


def test_the_rate_limit_reservation_counts_real_tokens_not_characters(monkeypatch):
    """len(text)//4 is an English rule of thumb. Vietnamese with diacritics
    runs about two tokens per four characters' guess -- measured at 48% --
    so reserving by it lets real usage overshoot the quota roughly twofold,
    and OpenAI's 429s then surface as ordinary failures."""
    reserved = {}

    class RecordingBucket:
        def __init__(self, name):
            self.name = name

        def acquire(self, tokens: int = 1) -> tuple[bool, int]:
            reserved[self.name] = reserved.get(self.name, 0) + tokens
            return True, 0

    monkeypatch.setattr("shared.rate_limiter.get_bucket", RecordingBucket)
    texts = ["Doanh thu quý 3 năm 2024 đạt 41,7 tỷ đồng, tăng 12 phần trăm."] * 5

    embed_chunks(texts, provider=StubProvider())

    assert reserved["embed_tpm"] == sum(count_tokens(t) for t in texts)


def test_resuming_from_partials_embeds_each_text_once_and_always_finishes(monkeypatch):
    """Same livelock S3 had: with room for only two batches per attempt,
    restarting at batch one every time never reaches batch three. Resuming
    from the vectors each deferred attempt hands back, it finishes, with
    every text embedded exactly once and the vectors in input order."""
    texts = [f"text {i}" for i in range(50)]
    sent = Counter()

    class CountingStub(StubProvider):
        def embed(self, batch):
            sent.update(batch)
            return super().embed(batch)

    class RoomForTwoBatches:
        def __init__(self):
            self.acquires = 0

        def acquire(self, tokens: int = 1) -> tuple[bool, int]:
            self.acquires += 1  # two per batch: embed_rpm, then embed_tpm
            return (True, 0) if self.acquires <= 4 else (False, 500)

    provider = CountingStub()
    done: list[list[float]] = []
    for _attempt in range(10):
        bucket = RoomForTwoBatches()
        monkeypatch.setattr("shared.rate_limiter.get_bucket", lambda name, b=bucket: b)
        try:
            vectors = embed_chunks(texts, provider=provider, batch_size=10, done=done)
            break
        except RateLimited as exc:
            done = exc.partial
    else:
        pytest.fail("never finished in 10 attempts")

    assert set(sent.values()) == {1}, "a text was embedded more than once"
    assert vectors == StubProvider().embed(texts)


def test_contextualize_puts_the_context_first_and_drops_an_empty_one():
    """One definition, used by S4 for what gets embedded and by S5 for what
    gets stored -- two copies of this format would let the stored text drift
    away from the text its vector was computed from."""
    assert contextualize("About revenue.", "Q3 was 41.7bn.") == "About revenue.\n\nQ3 was 41.7bn."
    assert contextualize("", "Q3 was 41.7bn.") == "Q3 was 41.7bn."
