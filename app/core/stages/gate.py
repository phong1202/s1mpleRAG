"""R6: does the context answer the main part of the question?

The first node that decides what the pipeline does next on its own: enough,
and the loop ends; not enough, and it names what is missing and the search
that would find it, and the pipeline searches again (spec §6.3). The context
accumulates across rounds (decision 12).

The calibration clause is the point of the prompt. A model trained towards
thoroughness always finds some detail missing, and without the clause every
question costs three rounds for nothing.
"""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from app.core.contracts import Context, GateVerdict, Query, Trace
from shared.llm import AsyncLLMProvider

_SYSTEM = (
    "You decide whether the sources below are enough to answer the question. "
    "You do not answer it.\n"
    "Answer sufficient when the sources answer the MAIN part of the question, even if secondary "
    "details are missing. Answer insufficient only when what is missing would make the answer "
    "wrong or misleading.\n"
    "When insufficient: missing says in one sentence exactly what is absent, and refined_query is "
    "one search question, in the language of the question, that would find it. Use only words "
    "from the question and the sources: never add a concept, fact or number that is not there.\n"
    "When sufficient: missing and refined_query are empty."
)


class GateResult(BaseModel):
    # No defaults: structured outputs in strict mode require every field.
    sufficient: bool
    missing: str
    refined_query: str


def build_messages(query: Query, context: Context) -> list[dict]:
    lines = [f"[QUESTION] {query.text}", "", "[SOURCES]"]
    for passage in context.passages:
        label = [passage.ref.filename, passage.heading, f"page {passage.ref.page_number}"]
        shown = " | ".join(part for part in label if part)
        lines.append(f"[{passage.ordinal}] ({shown}) {passage.text}")
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def make_gate(
    provider: AsyncLLMProvider, reasoning_effort: str | None = None
) -> Callable[[Query, Context, Trace], Awaitable[GateVerdict]]:
    async def gate(query: Query, context: Context, trace: Trace) -> GateVerdict:
        if query.intent in ("CHITCHAT", "SUMMARIZE_DOC"):
            # Small talk searched nothing, and a summary holds the whole
            # outline: a second search adds to neither.
            verdict, asked = GateVerdict(sufficient=True), False
        elif not context.passages:
            # Nothing to judge, and nothing to lose by searching again.
            verdict = GateVerdict(
                sufficient=False, missing="nothing found", refined_query=query.text
            )
            asked = False
        else:
            # A failure here is the pipeline's to catch: the no-op fallback
            # says "sufficient" and the question is answered from what is held.
            result = await provider.complete(
                build_messages(query, context), GateResult, reasoning_effort=reasoning_effort
            )
            verdict = GateVerdict(
                sufficient=result.sufficient,
                missing=None if result.sufficient else (result.missing.strip() or None),
                refined_query=None
                if result.sufficient
                else (result.refined_query.strip() or query.text),
            )
            asked = True
        trace.record(
            "gate_detail",
            ms=0,
            sufficient=verdict.sufficient,
            missing=verdict.missing,
            refined_query=verdict.refined_query,
            passages=len(context.passages),
            asked_model=asked,
        )
        return verdict

    return gate
