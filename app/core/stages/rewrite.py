"""R4: the question as it would read with no conversation before it.

Vector and keyword search have no memory. "Thế nếu người ngồi sau không đội
thì sao?" names neither helmets nor motorbikes; searched as written it finds
whatever else mentions a passenger. One call turns it into the question it
stands for, using the conversation the server kept.

The rewriter may only rephrase. A standalone question built from words that
are neither the user's nor the conversation's searches for something nobody
asked (spec §6.1: over-broadening, hallucinated context).
"""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from app.core.contracts import Intent, Query, Trace
from shared.llm import AsyncLLMProvider

# Turns of history the rewriter reads. A follow-up refers to the last
# exchange or two; more only adds tokens and older topics to confuse it with.
HISTORY_TURNS = 6
# An earlier answer is context for a pronoun, not something to re-read whole.
_TURN_CHARS = 600
_MAX_SEARCHES = 3

_SYSTEM = (
    "You prepare the user's latest message for a search engine over a library of documents. "
    "Return three fields.\n"
    "standalone: the latest message rewritten so that it can be understood without the "
    "conversation. Replace every pronoun and reference to earlier turns with what it refers to. "
    "Use only words from the conversation and the message: never add a concept, fact, number or "
    "detail that is not there. If the message already stands alone, return it unchanged. Keep "
    "the language of the message.\n"
    "sub_queries: empty, unless the message asks about two or more separate things; then one "
    "standalone question per thing, at most three.\n"
    "intent: CHITCHAT for greetings, thanks and small talk that ask for no information; "
    "SUMMARIZE_DOC when it asks to summarise or outline a document, or what a document covers "
    "or is about; METADATA when it asks about "
    "the library itself, such as which documents it holds; OUT_OF_SCOPE when it asks for "
    "information no document library would hold, such as today's weather or an exchange rate; "
    "otherwise SEARCH_CORPUS. When unsure, SEARCH_CORPUS.\n"
    "document_hint: when the message, read with the conversation, is about one particular "
    "document -- named by its title, number or subject -- that name as the user or the "
    'conversation wrote it, such as "Luật Đường bộ" or "Nghị định 168"; otherwise empty.'
)


class RewriteResult(BaseModel):
    # No defaults and no length limits: structured outputs in strict mode
    # require every field and reject some constraints. The limits are applied
    # in code below.
    intent: Intent
    standalone: str
    sub_queries: list[str]
    document_hint: str


def build_messages(query: Query) -> list[dict]:
    lines = []
    if query.history:
        lines.append("[CONVERSATION]")
        for turn in query.history[-HISTORY_TURNS:]:
            text = turn.text if len(turn.text) <= _TURN_CHARS else turn.text[:_TURN_CHARS] + " …"
            lines.append(f"{turn.role}: {text}")
        lines.append("")
    lines.append(f"[LATEST MESSAGE] {query.text}")
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def make_rewrite(
    provider: AsyncLLMProvider, reasoning_effort: str | None = None
) -> Callable[[Query, Trace], Awaitable[Query]]:
    async def rewrite(query: Query, trace: Trace) -> Query:
        # A failure here is the pipeline's to catch: it falls back to the no-op
        # and the question is searched as the user wrote it.
        result = await provider.complete(
            build_messages(query), RewriteResult, reasoning_effort=reasoning_effort
        )
        standalone = result.standalone.strip() or query.text
        searches = tuple(s.strip() for s in result.sub_queries if s.strip())[:_MAX_SEARCHES]
        document_hint = result.document_hint.strip() or None
        trace.record(
            "rewrite_detail",
            ms=0,
            intent=result.intent,
            standalone=standalone,
            sub_queries=list(searches),
            document_hint=document_hint,
        )
        return Query(
            text=standalone,
            conversation_id=query.conversation_id,
            filters=query.filters,
            history=query.history,
            # One search is the question itself; only two or more are a split.
            sub_queries=searches if len(searches) > 1 else (),
            intent=result.intent,
            document_hint=document_hint,
        )

    return rewrite
