"""Does reflection (R7) catch what it exists for?

The eval set's drafts stated nothing the sources do not, so `--full` could
only show what reflection drops wrongly. Here one fabricated sentence is
planted in each Set A draft (`eval/injections.yaml`) and reflection reads the
result: caught is the planted sentence dropped, collateral is anything else
dropped. Reflection runs whatever RETRIEVAL_REFLECT_ENABLED says, at
REFLECT_REASONING_EFFORT.

    python -m eval.inject --embed-cache data/eval-embeddings.json --llm-cache data/eval-llm.json
"""

import argparse
import asyncio
import collections
from dataclasses import dataclass

import yaml

from eval.judge import same_sentence


@dataclass(frozen=True)
class Injection:
    id: str
    kind: str
    sentence: str


def load_injections(path: str) -> list[Injection]:
    with open(path, encoding="utf-8") as handle:
        return [Injection(**row) for row in yaml.safe_load(handle)]


def plant(text: str, sentence: str) -> str:
    """At the end of the first paragraph: inside the answer, where a model's
    own fabrication sits, not tacked on after it."""
    head, sep, rest = text.partition("\n\n")
    return f"{head.rstrip()} {sentence}{sep}{rest}"


def outcome(planted: str, dropped: list[str]) -> tuple[bool, list[str]]:
    """(caught, collateral)."""
    caught = any(same_sentence(planted, d) for d in dropped)
    return caught, [d for d in dropped if not same_sentence(planted, d)]


async def _main(embed_cache: str | None, llm_cache: str | None) -> None:
    from app.config import get_settings
    from app.core.contracts import Answer, Trace
    from app.core.deps import build_stages
    from app.core.generator import map_citations
    from app.core.stages.reflect import make_reflect
    from app.utils.database import AsyncSessionLocal
    from eval.run import (
        CachedCompletions,
        CachedEmbeddings,
        dropped_by_reflection,
        load_dataset,
        retrieved,
    )
    from shared.llm import get_async_provider

    settings = get_settings()
    if settings.llm_provider != "openai":
        raise SystemExit("eval needs LLM_PROVIDER=openai: the corpus holds real embeddings")
    provider = get_async_provider()
    embeddings = completions = None
    if embed_cache:
        provider = embeddings = CachedEmbeddings(
            provider, embed_cache, settings.openai_embed_model, settings.embed_dimensions
        )
    if llm_cache:
        provider = completions = CachedCompletions(provider, llm_cache, settings.openai_chat_model)
    reflect = make_reflect(provider, reasoning_effort=settings.reflect_reasoning_effort)
    questions = {q.id: q for q in load_dataset("eval/dataset.yaml").set_a}
    caught: dict[str, list[bool]] = collections.defaultdict(list)
    collateral: list[str] = []
    missed: list[str] = []
    try:
        async with AsyncSessionLocal() as session:
            stages = build_stages(session, provider, settings)
            for planted in load_injections("eval/injections.yaml"):
                trace = Trace()
                gathered = await retrieved(
                    stages, questions[planted.id], trace, settings.retrieval_max_iterations
                )
                text, _ = await stages.generate(gathered.asked, gathered.context, trace)
                draft_text = plant(text, planted.sentence)
                draft = Answer(
                    text=draft_text,
                    citations=map_citations(draft_text, gathered.context),
                    trace=trace,
                )
                await reflect(gathered.asked, draft, gathered.context, trace)
                hit, others = outcome(planted.sentence, list(dropped_by_reflection(trace)))
                caught[planted.kind].append(hit)
                if not hit:
                    missed.append(f"{planted.id} ({planted.kind}): {planted.sentence}")
                collateral += [f"{planted.id}: {sentence}" for sentence in others]
    finally:
        for cache in (embeddings, completions):
            if cache is not None:
                cache.save()

    total = [hit for hits in caught.values() for hit in hits]
    print(f"reflect effort   {settings.reflect_reasoning_effort}")
    if completions is not None:
        print(f"paid completions {completions.paid}")
    print(f"caught           {sum(total)}/{len(total)}")
    for kind, hits in sorted(caught.items()):
        print(f"  {kind:<14} {sum(hits)}/{len(hits)}")
    for line in missed:
        print(f"  missed         {line}")
    print(f"collateral       {len(collateral)}")
    for line in collateral:
        print(f"  dropped        {line}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-cache", help="file keeping question embeddings between runs")
    parser.add_argument("--llm-cache", help="file keeping model answers between runs")
    args = parser.parse_args()
    asyncio.run(_main(args.embed_cache, args.llm_cache))


if __name__ == "__main__":
    main()
