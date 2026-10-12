import random
import uuid

from app.core.contracts import Candidate, DocRef
from app.core.stages.fusion import reciprocal_rank_fusion


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
