import uuid

from app.core.context_builder import build_context
from app.core.contracts import Candidate, Context, DocRef, WebRef


def _candidate(rank: int, text: str, source: str = "vector", parent=None) -> Candidate:
    return Candidate(
        source=source,
        rank=rank,
        score=1.0 / rank,
        text=text,
        ref=DocRef(
            document_id=uuid.uuid4(),
            parent_id=parent or uuid.uuid4(),
            child_id=uuid.uuid4(),
            page_number=rank,
            filename="d.pdf",
        ),
    )


def test_context_accumulates_across_iterations():
    """R6 loops. Replacing the context instead of adding to it lets the loop
    oscillate between two insufficient sets and never converge."""
    first = build_context(Context.empty(), [_candidate(1, "alpha")], budget=8000)

    second = build_context(first, [_candidate(1, "beta")], budget=8000)

    assert [p.text for p in second.passages] == ["alpha", "beta"]


def test_the_same_parent_is_not_added_twice():
    """Iteration two re-retrieves much of iteration one. Without dedupe the
    budget fills with duplicates."""
    parent = uuid.uuid4()
    first = build_context(Context.empty(), [_candidate(1, "alpha", parent=parent)], 8000)

    second = build_context(first, [_candidate(1, "alpha", parent=parent)], 8000)

    assert len(second.passages) == 1


def test_ordinals_are_stable_and_one_based():
    """The generator prints these numbers and maps citations back through
    them. Renumbering on the second pass would repoint every citation."""
    first = build_context(Context.empty(), [_candidate(1, "alpha")], 8000)
    second = build_context(first, [_candidate(1, "beta")], 8000)

    assert [p.ordinal for p in second.passages] == [1, 2]


def test_the_budget_drops_whole_passages_never_truncates_one():
    """Half a paragraph is worse than no paragraph: the model answers from a
    sentence whose qualifying clause was cut off."""
    long_text = "word " * 400

    context = build_context(
        Context.empty(),
        [_candidate(1, long_text), _candidate(2, long_text), _candidate(3, long_text)],
        budget=900,
    )

    assert all(p.text == long_text for p in context.passages)
    assert context.token_count <= 900


def test_the_budget_stops_at_the_first_passage_that_does_not_fit():
    """A lower-ranked passage must not overtake a higher-ranked one just for
    being shorter: rank is a judgement about relevance, length is not.

    The three passages have deliberately different lengths. Given equal ones,
    whatever does not fit is followed only by things that also do not fit, and
    stopping and skipping become indistinguishable."""
    context = build_context(
        Context.empty(),
        [
            _candidate(1, "a " * 300),  # 301 tokens, fits
            _candidate(2, "b " * 300),  # 301 tokens, does not fit
            _candidate(3, "c " * 50),  # 51 tokens, would fit -- must still be dropped
        ],
        budget=500,
    )

    assert [p.ordinal for p in context.passages] == [1]


def test_a_duplicate_never_ends_the_loop():
    """A passage already held costs nothing, so it must not be what trips the
    budget. At R6 iteration two re-retrieves much of iteration one and those
    duplicates arrive first: charging for one would leave the second pass
    unable to add anything at all."""
    parent = uuid.uuid4()
    held = _candidate(1, "a " * 300, parent=parent)  # 301 tokens
    first = build_context(Context.empty(), [held], budget=400)

    second = build_context(first, [held, _candidate(2, "c " * 20)], budget=400)

    assert [p.text for p in second.passages] == ["a " * 300, "c " * 20]


def test_web_candidates_land_in_the_untrusted_list():
    """R8 is far off, but the split is structural. If web text could reach
    `passages`, the prompt builder would have no way to keep it out."""
    web = Candidate(
        source="web", rank=1, score=1.0, text="from the internet",
        ref=WebRef(url="https://example.test", title="Example"),
    )

    context = build_context(Context.empty(), [_candidate(1, "alpha"), web], 8000)

    assert [p.text for p in context.passages] == ["alpha"]
    assert [p.text for p in context.untrusted] == ["from the internet"]
