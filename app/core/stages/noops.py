"""The five seams R0 leaves empty. Each one is also the fallback its real
implementation degrades to when it fails, which is why they are named rather
than inlined."""

from app.config import get_settings
from app.core.context_builder import build_context as _build_context
from app.core.contracts import Candidate, Context, GateVerdict, Plan, Query, Trace


async def rewrite(query: Query, trace: Trace) -> Query:
    """R4 replaces this."""
    return query


async def plan(query: Query, trace: Trace) -> Plan:
    """R5 replaces this. One search over the question as written -- or none,
    when the rewriter (R4) found small talk: "cảm ơn" asks for no
    information, and searching it only puts unrelated pages in the prompt."""
    if query.intent == "CHITCHAT":
        return Plan(strategy="NO_RETRIEVAL", queries=(), filters=query.filters)
    return Plan(strategy="VECTOR_ONLY", queries=query.search_texts, filters=query.filters)


async def fuse(query: Query, candidates: list[Candidate], trace: Trace) -> list[Candidate]:
    """R2 (RRF) and R3 (rerank) replace this.

    The depth cut is real work, not a no-op: the retriever returns over_fetch
    deep, and only the top retrieval_top_k are worth expanding into parents.
    Every version of this stage ends with the same cut -- except for an
    outline from the browse tool (R5), which is kept whole.
    """
    if candidates and all(c.source == "browse" for c in candidates):
        return candidates
    return candidates[: get_settings().retrieval_top_k]


def build_context_stage(previous: Context, candidates: list[Candidate], trace: Trace) -> Context:
    """The budget covers everything held so far (spec §6.3). Each round after
    the first gets room of its own: round one alone filled 7,712 of 8,000
    tokens, and the gate's further searches (R6) could add nothing."""
    settings = get_settings()
    budget = settings.context_token_budget + previous.rounds * settings.context_round_extra
    return _build_context(previous, candidates, budget)


async def gate(query: Query, context: Context, trace: Trace) -> GateVerdict:
    """R6 replaces this. Always sufficient means the loop runs once."""
    return GateVerdict(sufficient=True)


async def reflect(query: Query, answer, context: Context, trace: Trace):
    """R7 replaces this."""
    return answer


async def web(query: Query, trace: Trace) -> list[Candidate]:
    """What a failed web search (R8) leaves: the corpus alone."""
    return []
