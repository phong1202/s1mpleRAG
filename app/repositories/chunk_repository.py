"""R0 retrieval. Read-only: this module never writes."""

import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.contracts import Filters

# The query vector is bound as text and cast server-side. Binding it straight
# into CAST(:qvec AS vector) makes Postgres infer the parameter as `vector`,
# and the driver has no codec for that type -- the cast through text is what
# keeps the parameter a plain string.
_SEARCH = text("""
WITH candidates AS (
  SELECT c.id, c.parent_id, c.page_number, c.content,
         c.embedding <#> CAST(CAST(:qvec AS text) AS vector) AS distance
  FROM   child_chunks c
  WHERE  (CAST(:category AS text) IS NULL OR c.category = CAST(:category AS text))
    AND  (CAST(:document_id AS uuid) IS NULL OR c.document_id = CAST(:document_id AS uuid))
  -- ASC: <#> is NEGATIVE inner product, so nearest sorts first
  ORDER  BY c.embedding <#> CAST(CAST(:qvec AS text) AS vector)
  LIMIT  :over_fetch
),
best_per_parent AS (
  SELECT DISTINCT ON (parent_id)
         parent_id, distance, page_number, id AS child_id, content AS child_content
  FROM   candidates
  ORDER  BY parent_id, distance          -- parent_id MUST lead, or an arbitrary
)                                        -- child wins its group with no error
SELECT p.id  AS parent_id,
       p.content AS parent_content,
       b.child_id, b.child_content, b.page_number, b.distance,
       d.id AS document_id, d.filename
FROM   best_per_parent b
JOIN   parent_chunks p ON p.id = b.parent_id
JOIN   documents      d ON d.id = p.document_id
ORDER  BY b.distance
LIMIT  :top_k
""")


@dataclass(frozen=True)
class ParentHit:
    parent_id: uuid.UUID
    parent_content: str
    child_id: uuid.UUID
    child_content: str
    page_number: int
    distance: float
    document_id: uuid.UUID
    filename: str


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search(
        self, vector: list[float], over_fetch: int, top_k: int, filters: Filters
    ) -> list[ParentHit]:
        # pgvector's ef_search defaults to 40. Over-fetching 50 out of a queue
        # 40 wide degrades the tail silently, so the session sets it
        # explicitly. set_config rather than SET LOCAL: SET is a utility
        # statement and takes no bind parameters. The third argument is the
        # LOCAL flag -- it lasts until the end of this transaction.
        await self._session.execute(
            text("SELECT set_config('hnsw.ef_search', :ef, true)"),
            {"ef": str(get_settings().retrieval_ef_search)},
        )
        rows = await self._session.execute(
            _SEARCH,
            {
                "qvec": str(vector),
                "category": filters.category,
                "document_id": filters.document_id,
                "over_fetch": over_fetch,
                "top_k": top_k,
            },
        )
        return [ParentHit(**row) for row in rows.mappings()]
