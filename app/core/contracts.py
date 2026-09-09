# app/core/contracts.py
"""The read path's vocabulary. Written at R0 and unchanged through R8: every
later stage adds a function, never a field.

Three shapes look premature here and are not. `Candidate.rank` exists although
R0 has one source, because R2's RRF reads rank and adding it later means
touching every retriever. `Context.untrusted` is a separate list rather than a
flag, because a flag can be forgotten and a list cannot be silently merged.
`Ref` is a union although R0 only builds `DocRef`, because widening a type that
half the modules already destructure is the expensive kind of change.
"""

import uuid
from dataclasses import dataclass, field
from typing import Literal

Source = Literal["vector", "bm25", "browse", "web"]
Strategy = Literal["VECTOR_ONLY", "HYBRID", "BROWSE_DOC", "NO_RETRIEVAL"]


@dataclass(frozen=True)
class DocRef:
    """A reference into the corpus. Verifiable: one SQL statement confirms the
    page really contains the quoted text."""

    document_id: uuid.UUID
    parent_id: uuid.UUID
    child_id: uuid.UUID
    page_number: int
    filename: str


@dataclass(frozen=True)
class WebRef:
    """A reference outside the corpus. Not verifiable, and the page may say
    something different tomorrow. Never shares a type with DocRef."""

    url: str
    title: str


Ref = DocRef | WebRef


@dataclass(frozen=True)
class Filters:
    category: str | None = None
    document_id: uuid.UUID | None = None
    language: str | None = None


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class Query:
    text: str
    conversation_id: uuid.UUID | None = None
    filters: Filters = field(default_factory=Filters)
    history: tuple[Turn, ...] = ()
    sub_queries: tuple[str, ...] = ()

    def refined(self, text: str) -> "Query":
        """R6 loops by refining. Filters and history must survive the trip, or
        the second pass quietly searches a wider corpus than the first."""
        return Query(
            text=text,
            conversation_id=self.conversation_id,
            filters=self.filters,
            history=self.history,
            sub_queries=(),
        )

    @property
    def search_texts(self) -> tuple[str, ...]:
        return self.sub_queries or (self.text,)


@dataclass(frozen=True)
class Candidate:
    source: Source
    rank: int
    score: float
    text: str
    ref: Ref


@dataclass(frozen=True)
class Passage:
    ordinal: int
    text: str
    ref: DocRef


@dataclass(frozen=True)
class WebPassage:
    ordinal: int
    text: str
    ref: WebRef


@dataclass(frozen=True)
class Context:
    passages: tuple[Passage, ...] = ()
    untrusted: tuple[WebPassage, ...] = ()
    token_count: int = 0

    @classmethod
    def empty(cls) -> "Context":
        return cls()


@dataclass(frozen=True)
class DocCitation:
    ordinal: int
    document_id: uuid.UUID
    filename: str
    page_number: int
    quote: str


@dataclass(frozen=True)
class WebCitation:
    ordinal: int
    url: str
    title: str


Citation = DocCitation | WebCitation


@dataclass(frozen=True)
class TraceNode:
    node: str
    ms: int
    detail: dict


@dataclass
class Trace:
    """Mutable on purpose: every stage appends to the same trace as the request
    moves through it. R1 persists this."""

    nodes: list[TraceNode] = field(default_factory=list)
    iterations: int = 0

    def record(self, node: str, ms: int, **detail) -> None:
        self.nodes.append(TraceNode(node=node, ms=ms, detail=detail))

    @property
    def total_ms(self) -> int:
        return sum(n.ms for n in self.nodes)


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    trace: Trace


@dataclass(frozen=True)
class Plan:
    strategy: Strategy
    queries: tuple[str, ...]
    filters: Filters
    document_hint: str | None = None


@dataclass(frozen=True)
class GateVerdict:
    sufficient: bool
    missing: str | None = None
    refined_query: str | None = None
