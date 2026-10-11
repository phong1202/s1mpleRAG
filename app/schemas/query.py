import uuid

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    category: str | None = None
    document_id: uuid.UUID | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class CitationOut(BaseModel):
    """The DocCitation | WebCitation union, flattened for JSON.

    `kind` is what a client reads to know whether the reference can be checked:
    a document and page can, a URL cannot.
    """

    ordinal: int
    kind: str
    document_id: uuid.UUID | None = None
    filename: str | None = None
    page_number: int | None = None
    quote: str | None = None
    url: str | None = None
    title: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    latency_ms: int
