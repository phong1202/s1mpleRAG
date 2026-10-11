"""Retrieval-only by default: recall@10 and MRR need no chat call, only one
embedding per question, so the metric you check after every tuning change costs
next to nothing. `--full` runs the whole pipeline and adds citation accuracy
and mean iterations.

Set B (fabrication, the decision to go outside the corpus) is not scored yet:
both need a judge of the answer's text, which arrives with R7.
"""

import argparse
import asyncio
import unicodedata
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class LabelledQuestion:
    id: str
    kind: str
    question: str
    gold_sha256: str
    gold_page: int
    quote: str
    turns: tuple[str, ...] = ()


@dataclass(frozen=True)
class BehaviourQuestion:
    id: str
    kind: str
    question: str
    corpus_answers: bool


@dataclass(frozen=True)
class Dataset:
    set_a: list[LabelledQuestion]
    set_b: list[BehaviourQuestion]


def load_dataset(path: str) -> Dataset:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Dataset(
        set_a=[LabelledQuestion(**{**q, "turns": tuple(q.get("turns", ()))}) for q in raw["set_a"]],
        set_b=[BehaviourQuestion(**q) for q in raw["set_b"]],
    )


def recall_at_k(hits, gold: tuple[str, int], k: int) -> float:
    """Did the right page reach the context at all? This is the hard ceiling:
    a chunk that never arrives cannot be recovered by any prompt."""
    return float(any((h.sha256, h.page_number) == gold for h in hits[:k]))


def reciprocal_rank(hits, gold: tuple[str, int]) -> float:
    """Where did it land? Reranking does not change recall -- it brings nothing
    new in -- so this is the metric that shows whether it earned its keep."""
    for position, hit in enumerate(hits, start=1):
        if (hit.sha256, hit.page_number) == gold:
            return 1.0 / position
    return 0.0


@dataclass(frozen=True)
class Hit:
    sha256: str
    page_number: int


def _squash(text: str) -> str:
    """Case, Unicode form and whitespace are not part of a quote: OCR and PDF
    text extraction disagree on all three."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


async def unreachable_labels(session, questions: list[LabelledQuestion]) -> list[str]:
    """Ids of the labels no retrieval could ever hit: no child of that
    document, filed under that page, contains the quote."""
    from sqlalchemy import select

    from app.models import ChildChunk, Document

    missing = []
    for question in questions:
        rows = await session.execute(
            select(ChildChunk.content)
            .join(Document, Document.id == ChildChunk.document_id)
            .where(
                Document.sha256_hash == question.gold_sha256,
                ChildChunk.page_number == question.gold_page,
            )
        )
        quote = _squash(question.quote)
        if not any(quote in _squash(content) for (content,) in rows):
            missing.append(question.id)
    return missing


def by_kind(rows: list[tuple[str, float, float]]) -> dict[str, tuple[int, float, float]]:
    """(n, recall, MRR) for each kind of question, from (kind, recall, rank)
    rows. The overall figure cannot say which stage a miss belongs to."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    for kind, recall, rank in rows:
        grouped.setdefault(kind, []).append((recall, rank))
    return {
        kind: (
            len(scores),
            sum(r for r, _ in scores) / len(scores),
            sum(k for _, k in scores) / len(scores),
        )
        for kind, scores in grouped.items()
    }


def _as_query(question: LabelledQuestion):
    """The question as a user would have sent it: the last turn, with the
    turns before it as history. Until R4 reads the history, a follow-up is
    searched as its bare last turn -- and scores zero, which is the point."""
    from app.core.contracts import Query, Turn

    earlier = tuple(Turn(role="user", text=turn) for turn in question.turns[:-1])
    return Query(text=question.question, history=earlier)


async def retrieved(stages, question: LabelledQuestion) -> list:
    """What retrieval hands to the context for this question: the candidates
    leaving `fuse`. Measured there, not at one retriever, so the same ruler
    judges hybrid search (R2), reranking (R3) and rewriting (R4) unchanged."""
    from app.core.contracts import Trace

    trace = Trace()
    query = await stages.rewrite(_as_query(question), trace)
    plan = await stages.plan(query, trace)
    return await stages.fuse(query, await stages.retrieve(plan, trace), trace)


async def _check_labels() -> None:
    from app.utils.database import AsyncSessionLocal

    dataset = load_dataset("eval/dataset.yaml")
    async with AsyncSessionLocal() as session:
        missing = await unreachable_labels(session, dataset.set_a)
    if missing:
        # Measured anyway, each of these would score zero whatever retrieval did.
        raise SystemExit(f"unreachable labels, fix them first: {', '.join(missing)}")
    print(f"labels           {len(dataset.set_a)} reachable")


async def _main(full: bool) -> None:
    from sqlalchemy import select

    from app.config import get_settings
    from app.core.deps import build_stages
    from app.core.pipeline import answer as run_pipeline
    from app.models import Document
    from app.utils.database import AsyncSessionLocal
    from shared.llm import get_async_provider

    dataset = load_dataset("eval/dataset.yaml")
    settings = get_settings()
    if settings.llm_provider != "openai":
        # The corpus is embedded by OpenAI. A stub query vector is noise
        # against it, and the recall it yields looks exactly like a number.
        raise SystemExit("eval needs LLM_PROVIDER=openai: the corpus holds real embeddings")
    provider = get_async_provider()
    top_k = settings.retrieval_top_k

    recalls: list[float] = []
    ranks: list[float] = []
    scored: list[tuple[str, float, float]] = []
    missed: list[str] = []
    cited_right_page: list[float] = []
    iterations: list[int] = []

    async with AsyncSessionLocal() as session:
        # Labels are anchored to (sha256, page); candidates carry document_id.
        # One lookup bridges them without widening DocRef, which is frozen.
        rows = await session.execute(select(Document.id, Document.sha256_hash))
        sha_by_id = {row.id: row.sha256_hash for row in rows}

        stages = build_stages(session, provider, settings)

        for question in dataset.set_a:
            gold = (question.gold_sha256, question.gold_page)
            candidates = await retrieved(stages, question)
            hits = [Hit(sha_by_id[c.ref.document_id], c.ref.page_number) for c in candidates]
            recalls.append(recall_at_k(hits, gold, top_k))
            ranks.append(reciprocal_rank(hits, gold))
            scored.append((question.kind, recalls[-1], ranks[-1]))
            if not recalls[-1]:
                missed.append(question.id)

            if not full:
                continue

            result = await run_pipeline(
                _as_query(question),
                stages,
                max_iterations=settings.retrieval_max_iterations,
            )
            iterations.append(result.trace.iterations)
            cited_right_page.append(
                float(
                    any(
                        getattr(c, "page_number", None) == question.gold_page
                        and sha_by_id.get(getattr(c, "document_id", None)) == question.gold_sha256
                        for c in result.citations
                    )
                )
            )

    n = len(recalls)
    print(f"n                {n}")
    print(f"recall@{top_k:<10} {sum(recalls) / n:.3f}")
    print(f"MRR              {sum(ranks) / n:.3f}")
    for kind, (count, recall, rank) in by_kind(scored).items():
        print(f"  {kind:<14} n={count:<3} recall {recall:.3f}  MRR {rank:.3f}")
    print(f"missed           {', '.join(missed) or '-'}")
    if full:
        print(f"citation acc.    {sum(cited_right_page) / len(cited_right_page):.3f}")
        print(f"mean iterations  {sum(iterations) / len(iterations):.2f}")


async def _run(full: bool, check_only: bool) -> None:
    # One event loop for both: the engine's pooled connections belong to the
    # loop that opened them.
    await _check_labels()
    if not check_only:
        await _main(full)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also run generation")
    parser.add_argument(
        "--check-labels",
        action="store_true",
        help="verify every Set A quote sits on its gold page, then stop; calls no provider",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.full, args.check_labels))
