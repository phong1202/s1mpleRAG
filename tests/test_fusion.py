import random
import uuid
from dataclasses import replace

import pytest

from app.config import get_settings
from app.core.contracts import Candidate, DocRef, Query, Trace
from app.core.rerank_client import RerankError
from app.core.stages.fusion import make_fuse, reciprocal_rank_fusion


def _candidate(source: str, rank: int, child: uuid.UUID) -> Candidate:
    ref = DocRef(
        document_id=uuid.UUID(int=1),
        parent_id=uuid.UUID(int=2),
        child_id=child,
        page_number=1,
        filename="decree.pdf",
    )
    return Candidate(source=source, rank=rank, score=0.5, text="t", ref=ref)


A, B, C = (uuid.UUID(int=n) for n in (10, 11, 12))
# The table of the design, §5.3: A leads the vectors, C leads the keywords,
# B leads neither and is in both.
TABLE = [
    _candidate("vector", 1, A),
    _candidate("vector", 5, B),
    _candidate("bm25", 1, C),
    _candidate("bm25", 2, B),
]


def test_agreement_between_sources_beats_leading_one_list():
    fused, _ = reciprocal_rank_fusion(TABLE, k=60, top_k=10)

    assert fused[0].ref.child_id == B
    assert {c.ref.child_id for c in fused[1:]} == {A, C}


def test_fused_ranks_run_from_one_and_scores_are_the_rrf_sums():
    fused, _ = reciprocal_rank_fusion(TABLE, k=60, top_k=10)

    assert [c.rank for c in fused] == [1, 2, 3]
    assert fused[0].score == 1 / 65 + 1 / 62
    assert fused[1].score == fused[2].score == 1 / 61


def test_the_list_is_cut_to_top_k():
    many = [_candidate("vector", rank, uuid.UUID(int=100 + rank)) for rank in range(1, 16)]

    fused, _ = reciprocal_rank_fusion(many, k=60, top_k=10)

    assert len(fused) == 10


def test_one_source_alone_keeps_its_order():
    many = [_candidate("vector", rank, uuid.UUID(int=100 + rank)) for rank in range(1, 6)]

    fused, _ = reciprocal_rank_fusion(many, k=60, top_k=10)

    assert [c.ref.child_id for c in fused] == [c.ref.child_id for c in many]


def test_the_order_does_not_depend_on_the_input_order():
    """A and C tie. Whichever way the retrievers happened to be listed, the
    same question must give the same context."""
    shuffled = TABLE[:]
    random.Random(7).shuffle(shuffled)

    first, _ = reciprocal_rank_fusion(TABLE, k=60, top_k=10)
    second, _ = reciprocal_rank_fusion(shuffled, k=60, top_k=10)

    assert [c.ref.child_id for c in first] == [c.ref.child_id for c in second]


def test_the_summary_counts_where_the_kept_candidates_came_from():
    _, kept = reciprocal_rank_fusion(TABLE, k=60, top_k=10)

    assert kept == {"bm25+vector": 1, "vector": 1, "bm25": 1}


# --- R3: rerank after the fusion ---------------------------------------------


class _Reranker:
    """Scores each text by the number it ends with: the test decides the order."""

    def __init__(self, fail: bool = False, events: list | None = None) -> None:
        self.fail, self.calls, self.events = fail, [], events

    async def scores(self, query, documents):
        if self.events is not None:
            self.events.append("rerank")
        self.calls.append(list(documents))
        if self.fail:
            raise RerankError("ConnectError: reranker down")
        return [float(text.split()[-1]) for text in documents]


def _pool(n: int) -> list[Candidate]:
    # vector ranks 1..n; the text of rank r is "doc r"
    return [
        replace(_candidate("vector", rank, uuid.UUID(int=100 + rank)), text=f"doc {rank}")
        for rank in range(1, n + 1)
    ]


@pytest.fixture
def rerank_settings(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "retrieval_top_k", 10)
    monkeypatch.setattr(settings, "retrieval_rerank_candidates", 30)
    monkeypatch.setattr(settings, "retrieval_rerank_keep", 5)
    return settings


@pytest.mark.asyncio
async def test_the_reranker_sees_the_fused_pool_not_the_top_k(rerank_settings):
    """A chunk at RRF rank 20 is what the reranker exists to rescue. Cut to
    ten first and it never gets the chance."""
    reranker = _Reranker()

    await make_fuse(reranker)(Query(text="q"), _pool(40), Trace())

    assert len(reranker.calls[0]) == 30


@pytest.mark.asyncio
async def test_the_reranker_decides_the_order_and_keeps_rerank_keep(rerank_settings):
    trace = Trace()

    kept = await make_fuse(_Reranker())(Query(text="q"), _pool(40), trace)

    assert [c.text for c in kept] == ["doc 30", "doc 29", "doc 28", "doc 27", "doc 26"]
    assert [c.rank for c in kept] == [1, 2, 3, 4, 5]
    assert kept[0].score == 30.0
    detail = {n.node: n.detail for n in trace.nodes}["rerank_detail"]
    assert detail["sent"] == 30 and detail["fallback"] is False
    assert detail["from_rrf_ranks"] == [30, 29, 28, 27, 26]


@pytest.mark.asyncio
async def test_a_failed_rerank_keeps_the_fusion_order_and_says_why(rerank_settings):
    trace = Trace()

    kept = await make_fuse(_Reranker(fail=True))(Query(text="q"), _pool(40), trace)

    assert [c.text for c in kept] == [f"doc {r}" for r in range(1, 11)]
    detail = {n.node: n.detail for n in trace.nodes}["rerank_detail"]
    assert "reranker down" in detail["fallback"]


@pytest.mark.asyncio
async def test_the_connection_is_released_before_the_reranker_is_called(rerank_settings):
    events: list[str] = []

    async def release():
        events.append("release")

    await make_fuse(_Reranker(events=events), release=release)(Query(text="q"), _pool(5), Trace())

    assert events == ["release", "rerank"]


@pytest.mark.asyncio
async def test_without_a_reranker_the_stage_is_r2(rerank_settings):
    trace = Trace()

    kept = await make_fuse(None)(Query(text="q"), _pool(40), trace)

    assert [c.text for c in kept] == [f"doc {r}" for r in range(1, 11)]
    assert "rerank_detail" not in {n.node for n in trace.nodes}


def _hybrid_pool() -> list[Candidate]:
    """Two sources disagreeing: the fused order is neither list's order, so a
    fallback that returned the raw input instead of the fused list shows."""
    vector = [
        replace(_candidate("vector", r, uuid.UUID(int=200 + r)), text=f"v {r}")
        for r in range(1, 13)
    ]
    keyword = [
        replace(_candidate("bm25", r, uuid.UUID(int=212 - r)), text=f"k {r}") for r in range(1, 13)
    ]
    return vector + keyword


class _Broken:
    def __init__(self, error=None, scores=None):
        self.error, self.fixed = error, scores

    async def scores(self, query, documents):
        if self.error:
            raise self.error
        return self.fixed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reranker",
    [_Broken(RerankError("down")), _Broken(RuntimeError("anything")), _Broken(scores=[1.0])],
    ids=["rerank-error", "any-other-error", "too-few-scores"],
)
async def test_any_reranker_failure_keeps_the_fused_order_of_both_sources(
    rerank_settings, reranker
):
    """Before, an error the client did not wrap reached the pipeline, whose
    fallback for this stage is the no-op: the first ten of the raw input --
    the vector list -- with every keyword result dropped."""
    candidates = _hybrid_pool()
    expected, _ = reciprocal_rank_fusion(candidates, k=60, top_k=10)
    trace = Trace()

    kept = await make_fuse(reranker)(Query(text="q"), candidates, trace)

    assert [c.ref.child_id for c in kept] == [c.ref.child_id for c in expected]
    assert {c.source for c in kept} == {"vector", "bm25"}
    assert {n.node: n.detail for n in trace.nodes}["rerank_detail"]["fallback"]


@pytest.mark.asyncio
async def test_the_rerank_time_is_not_counted_twice(rerank_settings):
    """The fuse node already times the whole stage; a detail node with its own
    ms doubled query_traces.total_ms."""
    trace = Trace()

    await make_fuse(_Reranker())(Query(text="q"), _pool(5), trace)

    detail_node = next(n for n in trace.nodes if n.node == "rerank_detail")
    assert detail_node.ms == 0 and "took_ms" in detail_node.detail
