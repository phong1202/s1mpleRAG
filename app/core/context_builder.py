"""Turns ranked candidates into a token-budgeted prompt context.

Stable from R0 to R8. The one thing it must never do is merge trusted and
untrusted passages, and the one thing it must always do is accumulate rather
than replace: R6's loop depends on both.
"""

import tiktoken

from app.core.contracts import Candidate, Context, DocRef, Passage, WebPassage

# Resolved at import. In the image this reads TIKTOKEN_CACHE_DIR, which the
# Dockerfile fills at build time; otherwise the first call downloads a 1.7 MB
# BPE table and caches it under the system temp directory.
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_ENCODER.encode(text))


def build_context(previous: Context, candidates: list[Candidate], budget: int) -> Context:
    passages = list(previous.passages)
    untrusted = list(previous.untrusted)
    seen_parents = {p.ref.parent_id for p in passages}
    seen_urls = {p.ref.url for p in untrusted}
    ordinal = len(passages) + len(untrusted)
    used = previous.token_count

    for candidate in candidates:
        ref = candidate.ref
        seen = (
            ref.parent_id in seen_parents
            if isinstance(ref, DocRef)
            else ref.url in seen_urls
        )
        if seen:
            # Checked before the budget, because a passage already held costs
            # nothing and so must not be what ends the loop. At R6 iteration two
            # re-retrieves much of iteration one and those duplicates arrive
            # first, which would otherwise leave the second pass adding nothing.
            continue

        cost = _count(candidate.text)
        if used + cost > budget:
            # Stop, rather than skip ahead to something shorter. Truncating
            # mid-passage produces a sentence missing its qualifying clause,
            # which reads as fact; and letting a shorter, lower-ranked passage
            # overtake a longer one substitutes "fits" for "is relevant", a
            # judgement this function has no information to make.
            break

        ordinal += 1
        if isinstance(ref, DocRef):
            seen_parents.add(ref.parent_id)
            passages.append(Passage(ordinal=ordinal, text=candidate.text, ref=ref))
        else:
            seen_urls.add(ref.url)
            untrusted.append(WebPassage(ordinal=ordinal, text=candidate.text, ref=ref))
        used += cost

    return Context(passages=tuple(passages), untrusted=tuple(untrusted), token_count=used)
