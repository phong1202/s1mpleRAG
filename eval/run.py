"""Retrieval-only by default: recall@10 and MRR need no chat call, only one
embedding per question, so the metric you check after every tuning change costs
next to nothing. `--full` runs the whole pipeline and adds citation accuracy
and mean iterations.

Set B (fabrication, the decision to go outside the corpus) is not scored yet:
both need a judge of the answer's text, which arrives with R7.
"""

import argparse
import asyncio
import collections
import json
import pathlib
import statistics
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


class CachedEmbeddings:
    """The provider, with question embeddings kept in a file between runs.

    The questions are frozen (spec §8.4), so their vectors are too: only the
    first run pays for them, and every later tuning step can be measured
    without a paid call. Keyed by model, dimensions and text, so a change to
    any of them misses the cache instead of reusing a vector from another
    model. Everything else goes to the provider unchanged.
    """

    def __init__(self, provider, path: pathlib.Path, model: str, dimensions: int) -> None:
        self._provider, self._path = provider, pathlib.Path(path)
        self._prefix = f"{model}|{dimensions}|"
        self._vectors: dict[str, list[float]] = (
            json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}
        )
        self.paid = 0

    async def embed_query(self, text: str) -> list[float]:
        key = self._prefix + text
        if key not in self._vectors:
            self._vectors[key] = await self._provider.embed_query(text)
            self.paid += 1
        return self._vectors[key]

    async def complete(self, messages, schema, reasoning_effort=None):
        return await self._provider.complete(messages, schema, reasoning_effort)

    def save(self) -> None:
        # Written aside and moved into place: a run killed mid-write must not
        # leave a half file that loses every vector already paid for.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        partial = self._path.with_suffix(".partial")
        partial.write_text(json.dumps(self._vectors), encoding="utf-8")
        partial.replace(self._path)


class CachedCompletions:
    """The provider, with model answers kept in a file between runs.

    R4 asks the model to rewrite every question. The questions and their
    history are frozen, so the answers can be too: keyed by model, schema and
    the exact messages sent, a change to the prompt misses the cache instead of
    reusing an answer to a different prompt.
    """

    def __init__(self, provider, path: pathlib.Path, model: str) -> None:
        self._provider, self._path, self._model = provider, pathlib.Path(path), model
        self._answers: dict[str, dict] = (
            json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}
        )
        self.paid = 0

    async def complete(self, messages, schema, reasoning_effort=None):
        # The effort is part of the key: the same prompt at another effort is
        # another answer.
        key = json.dumps(
            [self._model, reasoning_effort, schema.__name__, messages],
            ensure_ascii=False,
            sort_keys=True,
        )
        if key not in self._answers:
            answer = await self._provider.complete(messages, schema, reasoning_effort)
            self._answers[key] = answer.model_dump()
            self.paid += 1
        return schema.model_validate(self._answers[key])

    async def embed_query(self, text: str) -> list[float]:
        return await self._provider.embed_query(text)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        partial = self._path.with_suffix(".partial")
        partial.write_text(json.dumps(self._answers, ensure_ascii=False), encoding="utf-8")
        partial.replace(self._path)


async def outline_coverage(session, document_id, parent_ids) -> float:
    """The share of a document's top-level sections that the given parents
    belong to. A summary question's gold is one page (spec §8.4), which a
    plain search finds; what a summary needs is the document's parts (R5)."""
    from sqlalchemy import text

    rows = await session.execute(
        text("""
        SELECT DISTINCT split_part(heading_path, ' > ', 1) AS top,
               bool_or(id = ANY(CAST(:reached AS uuid[]))) OVER (
                   PARTITION BY split_part(heading_path, ' > ', 1)) AS reached
        FROM   parent_chunks
        WHERE  document_id = CAST(:document_id AS uuid) AND heading_path IS NOT NULL
        """),
        {"document_id": document_id, "reached": list(parent_ids)},
    )
    sections = {row.top: row.reached for row in rows}
    return sum(sections.values()) / len(sections) if sections else 0.0


def strategy_of(trace) -> str | None:
    """What the router (R5) chose; None without one."""
    for node in trace.nodes:
        if node.node == "route_detail":
            return node.detail.get("strategy")
    return None


def intent_of(trace) -> str | None:
    """What the rewriter (R4) took the question for; None without one."""
    for node in trace.nodes:
        if node.node == "rewrite_detail":
            return node.detail.get("intent")
    return None


def stages_line(settings) -> str:
    """Which rung of the ladder was measured. A recall figure recorded without
    it cannot be compared with the next one."""
    flags = {
        "bm25": settings.retrieval_bm25_enabled,
        "rerank": settings.retrieval_rerank_enabled,
        "intent": settings.retrieval_intent_enabled,
        "router": settings.retrieval_router_enabled,
        "reflect": settings.retrieval_reflect_enabled,
    }
    return "stages           " + " ".join(
        f"{name}={'on' if enabled else 'off'}" for name, enabled in flags.items()
    )


def _as_query(question: LabelledQuestion):
    """The question as a user would have sent it: the last turn, with the
    turns before it as history. Until R4 reads the history, a follow-up is
    searched as its bare last turn -- and scores zero, which is the point."""
    from app.core.contracts import Query, Turn

    earlier = tuple(Turn(role="user", text=turn) for turn in question.turns[:-1])
    return Query(text=question.question, history=earlier)


def rerank_outcome(trace) -> tuple[str | None, str | None]:
    """(the model that reranked, why it did not) from a question's trace;
    (None, None) when no rerank stage ran."""
    for node in trace.nodes:
        if node.node == "rerank_detail":
            fallback = node.detail.get("fallback")
            if fallback:
                return None, fallback
            return node.detail.get("model"), None
    return None, None


async def retrieved(stages, question: LabelledQuestion, trace=None, max_iterations: int = 1):
    """Everything up to the answer, through the pipeline's own loop: the
    candidates leaving `fuse` on the first round -- the ruler since R1, so
    hybrid search (R2), reranking (R3), rewriting (R4) and routing (R5) are
    judged unchanged -- and the context the rounds end with, which is what
    the gate (R6) changes."""
    from app.core.contracts import Trace
    from app.core.pipeline import gather

    trace = Trace() if trace is None else trace
    return await gather(_as_query(question), stages, max_iterations, trace)


async def _check_labels() -> None:
    from app.utils.database import AsyncSessionLocal

    dataset = load_dataset("eval/dataset.yaml")
    async with AsyncSessionLocal() as session:
        missing = await unreachable_labels(session, dataset.set_a)
    if missing:
        # Measured anyway, each of these would score zero whatever retrieval did.
        raise SystemExit(f"unreachable labels, fix them first: {', '.join(missing)}")
    print(f"labels           {len(dataset.set_a)} reachable")


async def _main(full: bool, embed_cache: str | None = None, llm_cache: str | None = None) -> None:
    from sqlalchemy import select

    from app.config import get_settings
    from app.core.contracts import Trace
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
    embeddings = completions = None
    if embed_cache:
        provider = embeddings = CachedEmbeddings(
            provider, embed_cache, settings.openai_embed_model, settings.embed_dimensions
        )
    if llm_cache:
        provider = completions = CachedCompletions(provider, llm_cache, settings.openai_chat_model)
    top_k = settings.retrieval_top_k

    recalls: list[float] = []
    ranks: list[float] = []
    scored: list[tuple[str, float, float]] = []
    missed: list[str] = []
    cited_right_page: list[float] = []
    iterations: list[int] = []
    rerank_models: set[str] = set()
    rerank_fallbacks: list[str] = []
    intents: dict[str, collections.Counter] = {
        "A": collections.Counter(),
        "B": collections.Counter(),
    }
    chitchat_in_a: list[str] = []
    rewrite_ms: list[int] = []
    context_recalls: list[float] = []
    context_ranks: list[float] = []
    rounds: list[int] = []
    gate_ms: list[int] = []
    short: list[str] = []
    coverage: dict[str, float] = {}
    strategies: dict[str, collections.Counter] = {
        "A": collections.Counter(),
        "B": collections.Counter(),
    }
    browsed: dict[str, bool] = {}

    try:
        async with AsyncSessionLocal() as session:
            # Labels are anchored to (sha256, page); candidates carry document_id.
            # One lookup bridges them without widening DocRef, which is frozen.
            rows = await session.execute(select(Document.id, Document.sha256_hash))
            sha_by_id = {row.id: row.sha256_hash for row in rows}

            stages = build_stages(session, provider, settings)

            for question in dataset.set_a:
                gold = (question.gold_sha256, question.gold_page)
                trace = Trace()
                gathered = await retrieved(
                    stages, question, trace, settings.retrieval_max_iterations
                )
                candidates = gathered.first_round
                # R6's ruler: the pages in the context the rounds ended with,
                # in the order the model reads them.
                held = [
                    Hit(sha_by_id[p.ref.document_id], p.ref.page_number)
                    for p in gathered.context.passages
                ]
                context_recalls.append(recall_at_k(held, gold, len(held)))
                context_ranks.append(reciprocal_rank(held, gold))
                rounds.append(gathered.iterations)
                gate_ms += [n.ms for n in trace.nodes if n.node == "gate"]
                for node in trace.nodes:
                    if node.node == "gate_detail" and not node.detail["sufficient"]:
                        short.append(
                            f"{question.id}: missing {node.detail['missing']!r}"
                            f" -> searched {node.detail['refined_query']!r}"
                        )
                if strategy := strategy_of(trace):
                    strategies["A"][strategy] += 1
                intent = intent_of(trace)
                if intent:
                    rewrite_ms += [n.ms for n in trace.nodes if n.node == "rewrite"]
                    intents["A"][intent] += 1
                    if intent == "CHITCHAT":
                        chitchat_in_a.append(question.id)
                served, fallback = rerank_outcome(trace)
                if served:
                    rerank_models.add(served)
                if fallback:
                    rerank_fallbacks.append(f"{question.id}: {fallback}")
                if question.kind == "summarize":
                    gold_id = next(i for i, sha in sha_by_id.items() if sha == question.gold_sha256)
                    reached = [c.ref.parent_id for c in candidates if c.ref.document_id == gold_id]
                    coverage[question.id] = await outline_coverage(session, gold_id, reached)
                if candidates and all(c.source == "browse" for c in candidates):
                    picked = sha_by_id.get(candidates[0].ref.document_id)
                    browsed[question.id] = picked == question.gold_sha256
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
                            and sha_by_id.get(getattr(c, "document_id", None))
                            == question.gold_sha256
                            for c in result.citations
                        )
                    )
                )

            if settings.retrieval_intent_enabled:
                # Set B is not scored yet (it needs a judge, R7), but what the
                # rewriter makes of it is already worth seeing.
                from app.core.contracts import Query

                for question in dataset.set_b:
                    trace = Trace()
                    rewritten = await stages.rewrite(Query(text=question.question), trace)
                    await stages.plan(rewritten, trace)
                    intents["B"][intent_of(trace) or "none"] += 1
                    if strategy := strategy_of(trace):
                        strategies["B"][strategy] += 1

    finally:
        # Also when a question fails mid-run: what was paid for is kept.
        for cache in (embeddings, completions):
            if cache is not None:
                cache.save()
    if rerank_fallbacks:
        # These would be R2 figures under an R3 label.
        raise SystemExit(
            f"{len(rerank_fallbacks)} of {len(dataset.set_a)} questions were not reranked; "
            "no figures printed:\n  " + "\n  ".join(rerank_fallbacks)
        )
    n = len(recalls)
    print(stages_line(settings))
    if settings.retrieval_rerank_enabled:
        # From the traces, not the settings: what actually reranked.
        served = ", ".join(sorted(rerank_models)) or "none"
        print(f"reranker         {served} at {settings.reranker_url}")
    if embeddings is not None:
        print(f"paid embeddings  {embeddings.paid}")
    if completions is not None:
        print(f"paid completions {completions.paid}")
    for name, counter in intents.items():
        if counter:
            shown = ", ".join(f"{intent} {count}" for intent, count in counter.most_common())
            print(f"intents, set {name}   {shown}")
    if rewrite_ms and completions is not None and completions.paid == 0:
        print("rewrite          answered from the cache: time not measured")
    elif rewrite_ms:
        median = statistics.median(rewrite_ms)
        print(f"rewrite          median {median:.0f} ms, max {max(rewrite_ms)} ms")
    for name, counter in strategies.items():
        if counter:
            shown = ", ".join(f"{s} {count}" for s, count in counter.most_common())
            print(f"strategies, {name}   {shown}")
    if chitchat_in_a:
        # Small talk skips retrieval: a real question taken for it is lost.
        print(f"WARNING          taken for small talk: {', '.join(chitchat_in_a)}")
    print(f"n                {n}")
    print(f"recall@{top_k:<10} {sum(recalls) / n:.3f}")
    print(f"MRR              {sum(ranks) / n:.3f}")
    for kind, (count, recall, rank) in by_kind(scored).items():
        print(f"  {kind:<14} n={count:<3} recall {recall:.3f}  MRR {rank:.3f}")
    print(f"missed           {', '.join(missed) or '-'}")
    print(
        f"context          recall {sum(context_recalls) / n:.3f}  MRR {sum(context_ranks) / n:.3f}"
        f"  (all rounds; mean rounds {sum(rounds) / n:.2f})"
    )
    if gate_ms and (completions is None or completions.paid):
        print(f"gate             median {statistics.median(gate_ms):.0f} ms, max {max(gate_ms)} ms")
    for line in short:
        print(f"  not enough     {line}")
    if coverage:
        mean = sum(coverage.values()) / len(coverage)
        shown = ", ".join(f"{qid} {share:.2f}" for qid, share in coverage.items())
        print(f"outline reached  {mean:.3f}  ({shown})")
    if browsed:
        right = sum(browsed.values())
        wrong = [qid for qid, ok in browsed.items() if not ok]
        print(f"browsed          {right}/{len(browsed)} on the right document {wrong or ''}")
    if full:
        print(f"citation acc.    {sum(cited_right_page) / len(cited_right_page):.3f}")
        print(f"mean iterations  {sum(iterations) / len(iterations):.2f}")


async def _run(
    full: bool, check_only: bool, embed_cache: str | None, llm_cache: str | None
) -> None:
    # One event loop for both: the engine's pooled connections belong to the
    # loop that opened them.
    await _check_labels()
    if not check_only:
        await _main(full, embed_cache, llm_cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also run generation")
    parser.add_argument(
        "--embed-cache",
        metavar="PATH",
        help="keep question embeddings in this file; later runs pay for none (data/ is gitignored)",
    )
    parser.add_argument(
        "--llm-cache",
        metavar="PATH",
        help="keep model answers (the R4 rewrite) in this file; later runs pay for none",
    )
    parser.add_argument(
        "--check-labels",
        action="store_true",
        help="verify every Set A quote sits on its gold page, then stop; calls no provider",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.full, args.check_labels, args.embed_cache, args.llm_cache))
