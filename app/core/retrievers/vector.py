import asyncio
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.contracts import Candidate, DocRef, Plan, Source
from app.repositories.chunk_repository import ChildHit, ChunkRepository
from shared.llm import AsyncLLMProvider


class VectorRetriever:
    source: Source = "vector"

    def __init__(self, session: AsyncSession, provider: AsyncLLMProvider) -> None:
        self._repository = ChunkRepository(session)
        self._provider = provider

    async def search(self, plan: Plan) -> list[Candidate]:
        over_fetch = get_settings().retrieval_over_fetch
        vectors = await asyncio.gather(
            *(self._provider.embed_query(q) for q in plan.queries)
        )

        best: dict[uuid.UUID, ChildHit] = {}
        for vector in vectors:
            # Sequential where the embeddings above are concurrent: these share
            # one AsyncSession, and one connection cannot run two statements at
            # once. Gathering them raises "another operation is in progress".
            hits = await self._repository.search(
                vector=vector, over_fetch=over_fetch, filters=plan.filters
            )
            for hit in hits:
                # Sub-queries overlap by design. Keeping each child once, at its
                # best distance, is what stops one chunk from scoring twice in
                # R2's fusion merely because it matched two phrasings of the
                # same question.
                current = best.get(hit.child_id)
                if current is None or hit.distance < current.distance:
                    best[hit.child_id] = hit

        # Deduplicate first, rank second. Ranking first would leave gaps where
        # duplicates were removed, and R2 divides by (k + rank).
        ordered = sorted(best.values(), key=lambda hit: hit.distance)
        return [
            Candidate(
                source="vector",
                rank=position,
                # <#> is negated inner product; report plain similarity, so a
                # later sort by score cannot silently invert the ordering.
                score=-hit.distance,
                text=hit.child_content.strip(),
                ref=DocRef(
                    document_id=hit.document_id,
                    parent_id=hit.parent_id,
                    child_id=hit.child_id,
                    page_number=hit.page_number,
                    filename=hit.filename,
                ),
            )
            for position, hit in enumerate(ordered, start=1)
        ]
