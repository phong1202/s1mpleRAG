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
    """R5 replaces this. One vector search over the question as written."""
    return Plan(strategy="VECTOR_ONLY", queries=query.search_texts, filters=query.filters)


async def fuse(query: Query, candidates: list[Candidate], trace: Trace) -> list[Candidate]:
    """R2 (RRF) and R3 (rerank) replace this.

    The depth cut is real work, not a no-op: the retriever returns over_fetch
    deep, and only the top retrieval_top_k are worth expanding into parents.
    Every version of this stage ends with the same cut.
    """
    return candidates[: get_settings().retrieval_top_k]


def build_context_stage(previous: Context, candidates: list[Candidate], trace: Trace) -> Context:
    return _build_context(previous, candidates, get_settings().context_token_budget)


async def gate(query: Query, context: Context, trace: Trace) -> GateVerdict:
    """R6 replaces this. Always sufficient means the loop runs once."""
    return GateVerdict(sufficient=True)


async def reflect(answer, context: Context, trace: Trace):
    """R7 replaces this."""
    return answer
