import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.contracts import Candidate, DocRef, Plan, Source
from app.repositories.chunk_repository import ChunkRepository, KeywordHit


class BM25Retriever:
    """Keyword search beside the vectors. The name is the role, not the
    scorer: Postgres ranks with ts_rank, which has no IDF. See the SQL in
    ChunkRepository for why that is enough here."""

    source: Source = "bm25"

    def __init__(self, session: AsyncSession) -> None:
        self._repository = ChunkRepository(session)

    async def search(self, plan: Plan) -> list[Candidate]:
        over_fetch = get_settings().retrieval_over_fetch

        best: dict[uuid.UUID, KeywordHit] = {}
        for query in plan.queries:
            hits = await self._repository.keyword_search(
                query=query, over_fetch=over_fetch, filters=plan.filters
            )
            for hit in hits:
                # One child, one vote from this source, however many
                # sub-queries found it.
                current = best.get(hit.child_id)
                if current is None or hit.score > current.score:
                    best[hit.child_id] = hit

        # Deduplicate first, rank second: the fusion divides by (k + rank),
        # and a list with gaps would understate everything after the gap.
        ordered = sorted(best.values(), key=lambda hit: (-hit.score, str(hit.child_id)))
        return [
            Candidate(
                source="bm25",
                rank=position,
                score=hit.score,
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
