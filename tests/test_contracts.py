# tests/test_contracts.py
"""These types are the contract R2 through R8 are written against. The
assertions here are about the three shapes that are easy to get wrong and
expensive to change later: the rank that RRF will read, the untrusted list
that must never merge with trusted passages, and the reference union."""

import uuid
from dataclasses import FrozenInstanceError

import pytest

from app.core.contracts import (
    Candidate,
    Context,
    DocRef,
    Filters,
    Passage,
    Query,
    Trace,
    WebPassage,
    WebRef,
)


def _doc_ref(page: int = 1) -> DocRef:
    return DocRef(
        document_id=uuid.uuid4(),
        parent_id=uuid.uuid4(),
        child_id=uuid.uuid4(),
        page_number=page,
        filename="decree.pdf",
    )


def test_candidate_records_its_rank_within_its_own_source():
    """R0 has one source and no use for rank. R2's RRF scores 1/(k + rank),
    not by score, because cosine and ts_rank are different scales. Recording
    it now is what keeps R2 from touching every retriever."""
    candidate = Candidate(source="vector", rank=1, score=0.82, text="x", ref=_doc_ref())

    assert candidate.rank == 1
    assert candidate.source == "vector"


def test_context_keeps_untrusted_passages_in_a_separate_list():
    """A flag on Passage can be forgotten. Two lists make merging them a type
    error rather than an oversight."""
    context = Context(
        passages=(Passage(ordinal=1, text="trusted", ref=_doc_ref()),),
        untrusted=(WebPassage(ordinal=2, text="web", ref=WebRef(url="https://x", title="X")),),
        token_count=12,
    )

    assert [p.text for p in context.passages] == ["trusted"]
    assert [p.text for p in context.untrusted] == ["web"]


def test_empty_context_is_usable_before_the_first_iteration():
    empty = Context.empty()

    assert empty.passages == () and empty.untrusted == () and empty.token_count == 0


def test_refining_a_query_keeps_history_and_filters():
    """R6 loops by refining the query. Losing the filters on the way round
    would silently widen the search on the second pass."""
    filters = Filters(category="LEGAL")
    original = Query(text="original", filters=filters)

    refined = original.refined("narrower question")

    assert refined.text == "narrower question"
    assert refined.filters == filters
    assert original.text == "original", "Query is frozen; refine returns a new one"


def test_contracts_are_frozen():
    candidate = Candidate(source="vector", rank=1, score=0.5, text="x", ref=_doc_ref())

    with pytest.raises(FrozenInstanceError):
        candidate.rank = 2


def test_trace_records_nodes_in_order():
    trace = Trace()
    trace.record("retrieve", ms=95, vector=50)
    trace.record("generate", ms=3200, citations=4)

    assert [n.node for n in trace.nodes] == ["retrieve", "generate"]
    assert trace.nodes[0].detail == {"vector": 50}
