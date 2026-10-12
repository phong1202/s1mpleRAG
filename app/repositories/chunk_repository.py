"""Retrieval SQL. Read-only: this module never writes.

`search` returns CHILDREN, not parents. The collapse to parents is a separate
step (app/core/parent_expander.py) that runs after fusion and reranking,
because a cross-encoder scoring a 700-token parent has to find the one
relevant sentence buried in six irrelevant ones, while the same model scoring
the 150-token child reads nothing but the relevant text.
"""

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
#
# The ordering and the LIMIT stay inside a CTE over child_chunks alone: adding
# the documents join above them can cost the HNSW index scan.
#
# The cast sits in a scalar subquery so it runs once. Written inline, it is
# folded to a constant only while Postgres builds a custom plan; from the
# sixth execution on a connection it switches to a generic plan and re-parses
# the 1536-number string for every row it compares.
_SEARCH = text("""
WITH candidates AS (
  SELECT c.id, c.parent_id, c.document_id, c.page_number, c.content,
         c.embedding <#> (SELECT CAST(CAST(:qvec AS text) AS vector)) AS distance
  FROM   child_chunks c
  WHERE  (CAST(:category AS text) IS NULL OR c.category = CAST(:category AS text))
    AND  (CAST(:document_id AS uuid) IS NULL OR c.document_id = CAST(:document_id AS uuid))
  -- ASC: <#> is NEGATIVE inner product, so nearest sorts first
  ORDER  BY c.embedding <#> (SELECT CAST(CAST(:qvec AS text) AS vector))
  LIMIT  :over_fetch
)
SELECT c.id AS child_id, c.parent_id, c.document_id, c.page_number,
       c.content AS child_content, c.distance, d.filename
FROM   candidates c
JOIN   documents d ON d.id = c.document_id
ORDER  BY c.distance
""")

# Keyword search, and what Postgres does not give: it has no BM25. ts_rank
# counts term frequency with no IDF, so a word found in most of the corpus
# weighs as much as a code found in three chunks. That is tolerable because
# the fusion reads this list's order, never its scores.
#
# The question's words are joined with OR. websearch_to_tsquery joins them
# with AND, and a question asked in ordinary words never has every one of them
# in one chunk: measured on the real corpus, 19 of 20 eval questions came back
# empty that way, and 15 of 20 found their page in the top ten this way.
#
# The words come from to_tsvector, so they are normalised exactly as the index
# was. `tsv` is built with the english config for English chunks and simple
# for the rest, hence two queries, each matched against the rows built its
# way. The two OR branches are spelled out so the GIN index serves both.
#
# A question with no word in it makes string_agg return NULL: nothing
# matches, no error. The trailing c.id keeps equal scores in one order.
_KEYWORD = text("""
WITH q AS (
  SELECT (SELECT string_agg(quote_literal(lex), ' | ')::tsquery
          FROM   unnest(tsvector_to_array(to_tsvector('simple', :q))) AS lex) AS plain,
         (SELECT string_agg(quote_literal(lex), ' | ')::tsquery
          FROM   unnest(tsvector_to_array(to_tsvector('english', :q))) AS lex) AS english
), candidates AS (
  SELECT c.id, c.parent_id, c.document_id, c.page_number, c.content,
         -- 1: divide by 1 + log(length), or long chunks win on length alone
         ts_rank(c.tsv, CASE WHEN c.language = 'en' THEN q.english ELSE q.plain END, 1) AS score
  FROM   child_chunks c, q
  WHERE  ((c.language = 'en' AND c.tsv @@ q.english)
       OR (c.language IS DISTINCT FROM 'en' AND c.tsv @@ q.plain))
    AND  (CAST(:category AS text) IS NULL OR c.category = CAST(:category AS text))
    AND  (CAST(:document_id AS uuid) IS NULL OR c.document_id = CAST(:document_id AS uuid))
  ORDER  BY score DESC, c.id
  LIMIT  :over_fetch
)
SELECT c.id AS child_id, c.parent_id, c.document_id, c.page_number,
       c.content AS child_content, c.score, d.filename
FROM   candidates c
JOIN   documents d ON d.id = c.document_id
ORDER  BY c.score DESC, c.id
""")

_PARENTS = text("""
SELECT id, content, heading_path FROM parent_chunks WHERE id = ANY(CAST(:ids AS uuid[]))
""")


@dataclass(frozen=True)
class ChildHit:
    child_id: uuid.UUID
    parent_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    child_content: str
    distance: float
    filename: str


@dataclass(frozen=True)
class KeywordHit:
    child_id: uuid.UUID
    parent_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    child_content: str
    score: float
    filename: str


@dataclass(frozen=True)
class ParentText:
    content: str
    # Where the parent sits in its document. A parent is a slice of an article
    # and often does not name it; this is the only place that does.
    heading_path: str | None


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search(
        self, vector: list[float], over_fetch: int, filters: Filters
    ) -> list[ChildHit]:
        # pgvector's ef_search defaults to 40. Over-fetching 50 out of a queue
        # 40 wide degrades the tail silently, so the session sets it
        # explicitly. set_config rather than SET LOCAL: SET is a utility
        # statement and takes no bind parameters. The third argument is the
        # LOCAL flag -- it lasts until the end of this transaction.
        await self._session.execute(
            text("SELECT set_config('hnsw.ef_search', :ef, true)"),
            {"ef": str(get_settings().retrieval_ef_search)},
        )
        # A filtered search through the index is otherwise filtered AFTER the
        # index has returned its nearest ef_search rows: a category or document
        # matching few of them comes back short, or empty, with no error.
        # strict_order keeps scanning until the LIMIT is met, in exact order.
        await self._session.execute(
            text("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
        )
        rows = await self._session.execute(
            _SEARCH,
            {
                "qvec": str(vector),
                "category": filters.category,
                "document_id": filters.document_id,
                "over_fetch": over_fetch,
            },
        )
        return [ChildHit(**row) for row in rows.mappings()]

    async def keyword_search(
        self, query: str, over_fetch: int, filters: Filters
    ) -> list[KeywordHit]:
        rows = await self._session.execute(
            _KEYWORD,
            {
                "q": query,
                "category": filters.category,
                "document_id": filters.document_id,
                "over_fetch": over_fetch,
            },
        )
        return [KeywordHit(**row) for row in rows.mappings()]

    async def fetch_parents(self, parent_ids: list[uuid.UUID]) -> dict[uuid.UUID, ParentText]:
        """One statement for the whole set. Fetching per parent inside the
        expansion loop is the classic N+1, and it hides well because every
        individual query is fast."""
        if not parent_ids:
            return {}
        rows = await self._session.execute(_PARENTS, {"ids": parent_ids})
        return {row.id: ParentText(row.content, row.heading_path) for row in rows}
