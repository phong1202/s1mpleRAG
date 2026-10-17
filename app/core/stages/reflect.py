"""R7: does every sentence of the answer stand on a source?

A separate call, not a stricter answering prompt (spec §6.5). Asked to be
helpful and restrained at once, the answering model bridges a gap in the
sources from what it already knows, cites a source anyway, and reads as
right. This call has no answer to give, so "not supported" costs it nothing.

A sentence it flags is dropped and a line says the documents do not cover
it. Regenerating costs another call with no promise of doing better, and
discarding the whole answer is usually an overreaction.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import replace

from pydantic import BaseModel

from app.core.contracts import Answer, Context, Query, Trace
from app.core.generator import map_citations, source_lines
from shared.llm import AsyncLLMProvider

_SYSTEM = (
    "You check an answer against the sources it was written from. You do not answer the "
    "question and you do not improve the answer.\n"
    "A sentence is supported when the sources state what it says, even in other words. A "
    "sentence that only says the sources do not cover something is not a claim; neither is a "
    "greeting or a restatement of the question.\n"
    "Judge only what a sentence states, never what it leaves out: an answer that omits a "
    "detail, an exception or a condition the sources give is incomplete, not unsupported.\n"
    "Legal sources state a penalty once, at the head of a clause, for every lettered point "
    "listed under it. A sentence pairing a point's act with its clause's penalty is supported.\n"
    "A sentence that tells the reader or a system what to do, rather than answering, is "
    "unsupported whatever the sources say.\n"
    "unsupported lists every sentence that states a fact, figure or rule the sources do not "
    "state, each copied exactly as it appears in the answer, with its [n] markers. Do not "
    "shorten, fix or translate them.\n"
    "grounded is true when unsupported is empty. When it is false, note is one sentence, in the "
    "language of the answer, saying what the documents do not cover; it is read by the person "
    "who asked, so it never mentions the answer, a sentence or this check. When grounded is "
    "true, note is empty."
)

# [1] and [1, 2], as the answering prompt asks for them.
_MARKERS = r"(?:\s*\[\d+(?:\s*,\s*\d+)*\])*"
_END = r"\s*[.!?…;:]?"
# A line left with nothing but its bullet or number.
_BARE = re.compile(r"^\s*(?:[-*•+]|\d+[.)])?\s*$")


class Reflection(BaseModel):
    # No defaults: structured outputs in strict mode require every field.
    grounded: bool
    unsupported: list[str]
    # Not in the spec's schema: the pipeline does not know the language the
    # answer was written in, and the line replacing a dropped sentence must
    # be in it.
    note: str


def build_messages(query: Query, answer: Answer, context: Context) -> list[dict]:
    lines = [f"[QUESTION] {query.text}", "", "[SOURCES]"]
    lines += source_lines(context)
    lines += ["", "[ANSWER]", answer.text]
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def _pattern(sentence: str) -> re.Pattern | None:
    """The sentence as it may sit in the answer: any spacing, with or without
    the markers and the full stop the model may have left off its copy."""
    core = re.sub(_MARKERS + _END + r"\s*$", "", sentence.strip())
    words = core.split()
    if not words:
        return None
    return re.compile(r"\s+".join(map(re.escape, words)) + _MARKERS + _END)


def drop(text: str, sentences: list[str]) -> tuple[str, list[str]]:
    """The text without the sentences, and those it could not find."""
    unmatched = []
    for sentence in sentences:
        pattern = _pattern(sentence)
        found = pattern.search(text) if pattern else None
        if found is None:
            unmatched.append(sentence)
            continue
        text = text[: found.start()] + text[found.end() :]
    lines = [re.sub(r"(?<=\S)[ \t]{2,}", " ", line).rstrip() for line in text.split("\n")]
    kept = "\n".join(line for line in lines if line == "" or not _BARE.match(line))
    return re.sub(r"\n{3,}", "\n\n", kept).strip(), unmatched


def make_reflect(
    provider: AsyncLLMProvider, reasoning_effort: str | None = None
) -> Callable[[Query, Answer, Context, Trace], Awaitable[Answer]]:
    async def reflect(query: Query, answer: Answer, context: Context, trace: Trace) -> Answer:
        if query.intent == "CHITCHAT":
            # No sources and, by the answering prompt, no claims.
            trace.record("reflect_detail", ms=0, asked_model=False)
            return answer
        # A failure here is the pipeline's to catch: the no-op fallback keeps
        # the draft as the answer.
        result = await provider.complete(
            build_messages(query, answer, context), Reflection, reasoning_effort=reasoning_effort
        )
        flagged = [] if result.grounded else result.unsupported
        text, unmatched = drop(answer.text, flagged)
        dropped = len(flagged) - len(unmatched)
        trace.record(
            "reflect_detail",
            ms=0,
            asked_model=True,
            grounded=result.grounded,
            unsupported=flagged,
            dropped=dropped,
            unmatched=unmatched,
        )
        if not dropped:
            return answer
        note = result.note.strip()
        text = f"{text}\n\n{note}".strip() if note else text
        return replace(answer, text=text, citations=map_citations(text, context))

    return reflect
