import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.contracts import DocCitation, Filters, Query, Trace
from app.core.deps import build_stages
from app.core.pipeline import answer as run_pipeline
from app.core.stages.rewrite import HISTORY_TURNS
from app.exceptions.base import AppException
from app.exceptions.error_codes import ErrorCode
from app.models import QueryTrace
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.query import CitationOut, QueryRequest, QueryResponse
from shared.llm import AsyncLLMProvider

logger = logging.getLogger(__name__)


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

        conversations = ConversationRepository(self._session)
        if payload.conversation_id is None:
            conversation = await conversations.create()
        else:
            conversation = await conversations.get(payload.conversation_id)
            if conversation is None:
                raise AppException(ErrorCode.CONVERSATION_NOT_FOUND)
        # Read whether or not the rewriter is on: cheap, and it keeps the
        # conversation whole for when it is.
        history = await conversations.recent_turns(conversation.id, limit=HISTORY_TURNS)

        trace = Trace()
        try:
            result = await run_pipeline(
                Query(
                    text=payload.question,
                    conversation_id=conversation.id,
                    filters=Filters(category=payload.category, document_id=payload.document_id),
                    history=history,
                ),
                build_stages(self._session, self._provider, settings),
                max_iterations=settings.retrieval_max_iterations,
                trace=trace,
            )
        except Exception:
            # The request is lost; where it got to need not be. The row is
            # committed here because get_session rolls back on the way out,
            # and a table of successes only cannot say what broke.
            try:
                await self._session.rollback()
                # A conversation this request created may have been rolled
                # back with it; one the client named still exists.
                await _persist_trace(
                    self._session,
                    payload.question,
                    trace,
                    citation_count=0,
                    conversation_id=payload.conversation_id,
                )
                await self._session.commit()
            except Exception:
                # Never let the bookkeeping replace the error that matters.
                logger.exception("failed to persist the trace of a failed query")
            raise

        # Only an answer the user receives enters the history the next
        # question is read in.
        await conversations.add_exchange(conversation.id, payload.question, result.text)
        await _persist_trace(
            self._session, payload.question, trace, len(result.citations), conversation.id
        )

        return QueryResponse(
            answer=result.text,
            citations=[_serialise(c) for c in result.citations],
            latency_ms=int((time.perf_counter() - started) * 1000),
            conversation_id=conversation.id,
        )


def _serialise(citation) -> CitationOut:
    if isinstance(citation, DocCitation):
        return CitationOut(
            ordinal=citation.ordinal,
            kind="document",
            document_id=citation.document_id,
            filename=citation.filename,
            page_number=citation.page_number,
            quote=citation.quote,
        )
    return CitationOut(ordinal=citation.ordinal, kind="web", url=citation.url, title=citation.title)


async def _persist_trace(
    session,
    question: str,
    trace: Trace,
    citation_count: int,
    conversation_id=None,
) -> None:
    """Best effort by design: a broken trace table must not take down the
    feature it exists to observe."""
    try:
        # A savepoint, not a bare flush: a failed INSERT aborts the transaction
        # it ran in, and the commit that ends the request would fail after it.
        # Leaving the block flushes; an error rolls back to the savepoint only.
        async with session.begin_nested():
            session.add(
                QueryTrace(
                    question=question,
                    conversation_id=conversation_id,
                    # What was searched for, when the rewriter changed it.
                    standalone=next(
                        (
                            n.detail.get("standalone")
                            for n in trace.nodes
                            if n.node == "rewrite_detail"
                        ),
                        None,
                    ),
                    iterations=trace.iterations,
                    total_ms=trace.total_ms,
                    citation_count=citation_count,
                    nodes=[{"node": n.node, "ms": n.ms, "detail": n.detail} for n in trace.nodes],
                )
            )
    except Exception:
        logger.exception("failed to persist query trace")
