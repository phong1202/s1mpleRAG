import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.contracts import DocCitation, Filters, Query
from app.core.deps import build_stages
from app.core.pipeline import answer as run_pipeline
from app.schemas.query import CitationOut, QueryRequest, QueryResponse
from shared.llm import AsyncLLMProvider


class QueryService:
    def __init__(self, session: AsyncSession, provider: AsyncLLMProvider) -> None:
        # The provider arrives from outside rather than from get_async_provider()
        # here: a test cannot override what a function fetches for itself, and
        # every stage from R4 on is an LLM call that tests must be able to script.
        self._session = session
        self._provider = provider

    async def ask(self, payload: QueryRequest) -> QueryResponse:
        settings = get_settings()
        started = time.perf_counter()

        result = await run_pipeline(
            Query(
                text=payload.question,
                filters=Filters(category=payload.category, document_id=payload.document_id),
            ),
            build_stages(self._session, self._provider, settings),
            max_iterations=settings.retrieval_max_iterations,
        )

        return QueryResponse(
            answer=result.text,
            citations=[_serialise(c) for c in result.citations],
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _serialise(citation) -> CitationOut:
    if isinstance(citation, DocCitation):
        return CitationOut(
            ordinal=citation.ordinal, kind="document", document_id=citation.document_id,
            filename=citation.filename, page_number=citation.page_number, quote=citation.quote,
        )
    return CitationOut(
        ordinal=citation.ordinal, kind="web", url=citation.url, title=citation.title
    )
