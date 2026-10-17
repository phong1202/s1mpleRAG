"""R5: which tool answers the question.

Spec §6.2 gives the router a model call of its own. This one reads what the
rewriter (R4) already returned -- the intent, and the one document the
question names -- because with the strategies there are, nothing else would
change the choice, and a second call costs every question about a second
(measured at R4). If a strategy ever needs a judgement the intent cannot
carry, that is the time for the call.

The choice and its reason go into the trace: a router that has quietly become
a constant is visible only in the distribution of its choices (spec §6.2).
"""

from app.core.contracts import Plan, Query, Trace


async def route(query: Query, trace: Trace) -> Plan:
    if query.intent == "CHITCHAT":
        strategy, reason = "NO_RETRIEVAL", "small talk asks for no information"
    elif query.intent == "SUMMARIZE_DOC" and (query.document_hint or query.filters.document_id):
        strategy, reason = "BROWSE_DOC", "asks about one whole document"
    elif query.intent == "SUMMARIZE_DOC":
        strategy, reason = "HYBRID", "asks for a summary but names no document"
    else:
        strategy, reason = "HYBRID", "a question for the corpus"
    trace.record(
        "route_detail",
        ms=0,
        strategy=strategy,
        document_hint=query.document_hint,
        reason=reason,
    )
    return Plan(
        strategy=strategy,
        queries=() if strategy == "NO_RETRIEVAL" else query.search_texts,
        filters=query.filters,
        # Kept for a fallback: when no document matches the name, the
        # browse plan is searched like any other question.
        document_hint=query.document_hint,
    )
