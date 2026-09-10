from typing import Protocol

from app.core.contracts import Candidate, Plan, Source


class Retriever(Protocol):
    """Adding a tool at R2, R5 or R8 means adding a class here and registering
    it. The pipeline calls every enabled retriever concurrently and never knows
    which ones exist.

    `source` is an attribute rather than a method because the pipeline needs to
    label a retriever before calling it -- for the trace, and so fusion knows
    how many lists it is merging.
    """

    source: Source

    async def search(self, plan: Plan) -> list[Candidate]: ...
