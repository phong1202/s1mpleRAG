"""The judge `--full` reads answers with (spec §8.2: fabrication rate, and
whether Set B is declined or answered).

Not reflection grading itself: a different prompt, at a higher reasoning
effort. It is still the same model, because the provider serves one, so it
can share reflection's blind spots. What it flags is printed for a person to
read; the judge only says where to look.
"""

import re

from pydantic import BaseModel

from app.core.contracts import Context
from app.core.generator import source_line

JUDGE_REASONING_EFFORT = "high"

_SYSTEM = (
    "You grade an answer that was written from the sources below. You did not write it.\n"
    "unsupported lists every statement in the answer of a fact, figure or rule that no source "
    "states, copied from the answer. A statement is supported when a source states it, even in "
    "other words. Saying the sources do not cover something is not a claim.\n"
    "declined is true when the answer as a whole says the documents do not answer the question "
    "and gives no answer to it; false when it answers the question, even in part."
)


class Judgement(BaseModel):
    # No defaults: structured outputs in strict mode require every field.
    declined: bool
    unsupported: list[str]


def build_messages(question: str, answer: str, context: Context) -> list[dict]:
    lines = [f"[QUESTION] {question}", "", "[SOURCES]"]
    lines += [source_line(passage) for passage in context.passages]
    lines += ["", "[ANSWER]", answer]
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


async def judge(provider, question: str, answer: str, context: Context) -> Judgement:
    return await provider.complete(
        build_messages(question, answer, context),
        Judgement,
        reasoning_effort=JUDGE_REASONING_EFFORT,
    )


def fabrication_rate(judged: list[Judgement]) -> float:
    """The share of answers stating anything no source states."""
    return sum(bool(j.unsupported) for j in judged) / len(judged)


def decided_right(kind: str, judged: Judgement) -> bool:
    """Set B: a question outside the corpus is declined; one the corpus
    answers, however much it sounds like it needs the web, is answered."""
    return judged.declined if kind == "outside" else not judged.declined


def _bare(sentence: str) -> str:
    without = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", sentence)
    return " ".join(without.casefold().split()).strip(" .!?…;:")


def same_sentence(a: str, b: str) -> bool:
    """Two copies of one sentence: markers, case, spacing and the full stop
    are what a model changes when it copies, so they do not count."""
    a, b = _bare(a), _bare(b)
    return bool(a and b) and (a in b or b in a)


def unconfirmed(dropped: list[str], judged: Judgement) -> list[str]:
    """Sentences reflection dropped that the judge, reading the draft, did not
    flag: each one is either an over-drop or a judge's miss."""
    return [s for s in dropped if not any(same_sentence(s, f) for f in judged.unsupported)]
