# Phase 2 · R0 + R1 — Seams, Basic RAG, and the Measuring Stick · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer a question from the corpus with citations that name a document and a page — and be able to prove, with a number, whether any later change made retrieval better.

**Architecture:** One `for` loop in `app/core/pipeline.py` with eight seams. Three seams are real at R0 (retrieve, build context, generate); five are no-ops (rewrite, plan, fuse, gate, reflect) that later stages replace without restructuring anything. The loop bound is a config constant that R6 will raise from `1` to `3`. R1 adds a trace row per query and a labelled question set, which is what every stage from R2 onward argues against.

**Tech Stack:** FastAPI (async) · SQLAlchemy 2 async + `asyncpg` · Postgres 16 + pgvector (HNSW, `vector_ip_ops`) · OpenAI `text-embedding-3-large` (shortened to 1536 dimensions) / `gpt-6-luna` behind a Protocol · pytest · uv · ruff

**Spec:** [`docs/retrieval-design.md`](../retrieval-design.md) — the settled design. Where this plan and the spec disagree, **the spec wins**.

---

## In step with `main` — 2026-10-10

Phase 1 is merged into `main` (PR #1). On 2026-10-10 `phase-2` was rebased onto `main`: the branch's 13 commits now sit on the 46 commits `main` had gained. This section records what that rebase changes in the plan. Where it contradicts a task written before that date, **this section wins**.

### Status

Updated 2026-10-10, after the first execution pass. Nothing is committed; each task has a snapshot of its own awaiting review.

| Task | Status |
| --- | --- |
| 1 → 9 (R0) | Committed on `phase-2`. Summarised under R0. |
| 9b, 9c | Done. |
| 9d | **Waiting**: it calls OpenAI (3 embeddings + 3 chat calls) and needs a go-ahead. |
| 10, 11 | Done. |
| 12 | Steps 1 → 7 done: runner, tests, 28 drafted items, 20 labels reported reachable. **Waiting**: a person reviews `eval/dataset.yaml` (Step 8), then the baseline is measured with OpenAI (Step 9). |
| 13 | README rewritten and the spec's facts corrected (Steps 1 → 3, 5). **Left**: running the README's commands with the stack down (Step 4), and the baseline in spec §8. |
| 14 | Done: the recommendation is not to use DeepSeek. |

The suite now: 318 tests green (`--ignore=tests/test_ocr_service.py`), `ruff check .` and `ruff format --check .` clean.

### Conflicts resolved in the rebase

- **`Dockerfile`, `pyproject.toml`** (the context builder commit). `main` already had `tiktoken` and already baked the BPE table into `/opt/tiktoken-cache`. `main`'s versions were kept; that commit now adds only `app/core/context_builder.py` and its tests. The Dockerfile snippet in Task 6 Step 1 no longer applies.
- **`app/controllers/__init__.py`** (the `POST /query` commit). Both `ingestion_controller` and `query_controller` are registered.

The local branch has now diverged from `origin/phase-2`. Pushing it takes `git push --force-with-lease`, and only when told to.

### What `main` changed that this plan must follow

| Matter | On `main` | Consequence here |
| --- | --- | --- |
| Models | `text-embedding-3-large` shortened to 1536 dimensions; chat is `gpt-6-luna` with `OPENAI_REASONING_EFFORT=low` | The Tech Stack line is corrected. `AsyncOpenAIProvider` uses the right model and dimensions but does not pass `reasoning_effort` → Task 9c |
| The sync provider was hardened | Strict structured outputs, a clear error when the key is missing | The async one still calls JSON mode and never shows the model the schema → Task 9c |
| CI (`.github/workflows/ci.yml`) | A PR into `main` runs `ruff check`, `ruff format --check`, pytest and the `web/` build | 14 files of this branch are unformatted → Task 9b. "Suite green" now includes formatting |
| Tests need real services | Postgres, RabbitMQ, Redis, MinIO | `docker compose up -d --wait db rabbitmq redis minio` before running the suite |
| Part of the schema belongs to the database | The HNSW index, the generated column `child_chunks.tsv`, two indexes and a check constraint on `documents` are deliberately not mapped on the models | `alembic --autogenerate` proposes to **drop** all of them → Task 10 Step 4 |
| A real corpus | 10 documents, about 2,960 child chunks, Vietnamese traffic law; the source PDFs are in `data/` (not committed) | The e-invoice examples in Task 12 are shape only; real labels are made against this corpus |
| The dev machine's `.env` | `LLM_PROVIDER=openai` | Every real run (DoD 1, Task 12 Step 6) spends on OpenAI → a new line in Global Constraints |
| Test fixtures | `client` in `tests/conftest.py` does not replace the read path's provider | Task 11's tests are rewritten on `query_client` |
| FE (`web/`) | Has a chat panel and a PDF viewer; its `SourceReference` type is still marked provisional | See below |

### Outside this plan

- **Wiring the FE to `POST /query`.** A frontend session plans and does it separately, from [`docs/fe-query-handoff.md`](../fe-query-handoff.md). This plan does not touch `web/`.
- **`locations` for citations** (`docs/system-design.md` §5.3: the format is fixed and marked "built in Phase 2"). It takes S1 keeping the bboxes, S2 attaching them to children, a migration, and **re-ingesting the corpus**. It needs a plan of its own. Until then the FE falls back to `page_number`, which R0 already returns.
- **The read path does not go through the worker's rate limiter** (`rl:openai:*`). A 429 from OpenAI in `retrieve` or `generate` surfaces as a 500. Accepted for R0/R1; revisit when R4 → R7 add LLM calls to every request.
- **`/query` has no authentication**, like every other endpoint (a risk already recorded in the Phase 1 plan).

---

## Global Constraints

Every task implicitly carries these. Values copied verbatim from the spec.

- **Python** `>=3.12`; dependencies via `uv add` / `uv add --dev`, never by hand-editing `pyproject.toml`.
- **Branch:** all of R0 and R1 land on branch `phase-2`.
- **Git:** do NOT run `git commit`, `git push`, or `git merge`. End each task by stopping and reporting; wait for an explicit instruction.
- **Branch point: `phase-2` sits on `main`** since the rebase of 2026-10-10, and `main` carries all of Phase 1. **Every test in this plan seeds its own rows**, so they require the schema and the running services, not an ingested corpus. Only Definition of Done item 1 and Task 12 Step 6 need a really ingested corpus. Rebase again whenever `main` advances.
- **The suite must be green at the end of EVERY task.** `uv run pytest -q --ignore=tests/test_ocr_service.py`, `uv run ruff check .` and `uv run ruff format --check .` all clean: those are the three things CI runs on a PR into `main`. The suite needs Postgres, RabbitMQ, Redis and MinIO running (`docker compose up -d --wait db rabbitmq redis minio`). `tests/test_ocr_service.py` needs `chandra` and a GPU, and has nothing to do with the read path.
- **The API side is async.** Everything under `app/core/` is `async def`. The worker's synchronous `LLMProvider` is not touched; R0 adds a second Protocol beside it.
- **`ef_search` is set explicitly per session** and must be `>= 2 x RETRIEVAL_OVER_FETCH`. pgvector's default is 40; asking for 50 neighbours from a queue of 40 silently degrades the tail.
- **`<#>` is negative inner product.** `ORDER BY ... ASC` is nearest-first. `DESC` reverses the result set and raises nothing.
- **`DISTINCT ON (parent_id)` requires `parent_id` to lead the `ORDER BY`.** Otherwise Postgres keeps an arbitrary child per parent, with no error.
- **`Candidate.rank` is the rank within its own source**, recorded even at R0 where there is one source. R2's RRF reads it.
- **`Context.untrusted` is a separate list, never a flag on a passage.**
- **`ref` is `DocRef | WebRef` from the first line of code**, even though R0 only ever builds `DocRef`.
- **UUIDs never enter a prompt.** Sources are numbered `[1]`, `[2]`; code maps the number back.
- **Failure policy:** any stage that fails to parse or violates its schema falls back to its own no-op. A capability is lost; the request is not.
- **Stage flags default to off**, and `RETRIEVAL_MAX_ITERATIONS` defaults to `1`. With every flag off the system must behave exactly as R0 — and a test must prove it.
- **Every LLM call in tests goes through `AsyncStubProvider`.** The suite runs with no API key and costs nothing.
- **Calling the real OpenAI is a deliberate act.** The dev machine's `.env` sets `LLM_PROVIDER=openai`, so `POST /query` on the running API and `python -m eval.run` both cost money. This plan needs the real provider in exactly two places: Definition of Done item 1 and Task 12 Step 6. Stop and ask before running either.

---

## The R0 / R1 split

| | Tasks | Deliverable |
| --- | --- | --- |
| **R0** | 1 → 9 *(done)* | `POST /query` returns an answer with citations; all eight seams exist; five are no-ops |
| **R0** | 9b, 9c, 9d | The branch passes `main`'s CI; the real provider returns what the schema asks for; a real question comes back citing the right page |
| **R1** | 10 → 12 | Every query writes a trace row; `eval/run.py` prints `recall@10` and `MRR` against a labelled set |
| **Close** | 13 | `README.md` and the spec describe the system as built |
| **Separate** | 14 | A recommendation on DeepSeek, per LLM call |

R0 is done when a question about a real ingested PDF comes back with a citation naming the right page. R1 is done when you have a **baseline number written down** — the thing R2 through R8 have to beat.

---

## File structure

```
app/core/                          ★ NEW — the read path, API-side only
  contracts.py                     Query, Candidate, Context, Answer, Trace, refs (Task 1)
  pipeline.py                      the loop and its eight seams (Task 8)
  deps.py                          assembles the stage callables from settings flags (Task 8)
  retrievers/
    base.py                        Retriever Protocol (Task 5)
    vector.py                      R0 — embed, search, rank children (Task 5)
  stages/
    noops.py                       the five no-op seams (Task 8)
  parent_expander.py               R0 — collapse children to parents (Task 4b)
  context_builder.py               accumulate, dedupe, token budget, untrusted split (Task 6)
  generator.py                     numbered sources, citation mapping (Task 7)
app/repositories/
  chunk_repository.py              ★ NEW — child-level search + parent fetch (Task 4)
app/schemas/query.py               ★ NEW — request/response bodies (Task 9)
app/services/query_service.py      ★ NEW — orchestrates app/core/ (Task 9)
app/controllers/query_controller.py ★ NEW — POST /query (Task 9)
app/models/query_trace.py          ★ NEW — R1 (Task 10)
shared/llm.py                      MODIFY — add AsyncLLMProvider beside the sync one (Task 2)
app/config.py                      MODIFY — retrieval settings and stage flags (Task 3)
README.md                          MODIFY — rewritten for the system as built (Task 13)
eval/
  dataset.yaml                     ★ NEW — 20 + 8 labelled items (Task 12)
  run.py                           ★ NEW — retrieval-only by default (Task 12)
alembic/versions/                  ★ ONE new revision (Task 10)
tests/
  test_contracts.py                Task 1
  test_async_llm_provider.py       Task 2
  helpers.py                       shared seed data (Task 4)
  test_chunk_repository.py         Task 4
  test_parent_expander.py          Task 4b
  test_vector_retriever.py         Task 5
  test_context_builder.py          Task 6
  test_generator.py                Task 7
  test_pipeline.py                 Task 8  ← the most important test in this plan
  test_query_api.py                Task 9
  test_query_trace.py              Task 10, 11
  test_eval_harness.py             Task 12
```

**`app/core/` never imports from `worker/`, and `worker/` never imports from `app/core/`.** The two paths share `app/models`, `app/config` and `shared/` and nothing else — the same boundary Phase 1 established.

---

# R0 — Seams and Basic RAG

## Tasks 1 → 9: done

The first nine tasks are committed on `phase-2`. Their code and tests are in git and are not repeated here. The step-by-step plan for those nine is still in history: `git log -p -- docs/plans/phase-2-r0-r1.en.md`.

| Task | Deliverable | Commit |
| --- | --- | --- |
| 1 | `app/core/contracts.py` | `feat(core): add the read path contracts` |
| 2 | `AsyncLLMProvider`, `AsyncStubProvider`, `AsyncOpenAIProvider` in `shared/llm.py` | `feat(shared): add the async llm provider for the read path` |
| 3 | The `RETRIEVAL_*` settings and stage flags | `feat(config): add retrieval settings and stage flags` |
| 4, 5 | `ChunkRepository` and `VectorRetriever`: search and rank at child level | `feat(core): add the parent-collapsing retrieval query`, `feat(core): retrieve and rank children instead of parents` |
| 4b | `expand_parents` | `feat(core): add the parent expansion stage` |
| 6 | `build_context` | `feat(core): add the context builder` |
| 7 | `generate`, `map_citations` | `feat(core): add the answer generator and citation mapping` |
| 8 | `Stages`, `answer`, the five no-ops, `build_stages` | `feat(core): add the retrieval pipeline and its no-op seams` |
| 9 | `POST /query` | `feat(api): add POST /query` |

**Interfaces the remaining tasks use.** Signatures are taken from the code as it stands, not from the old plan:

- `Answer(text, citations, trace)`; `Trace.nodes` is a list of `TraceNode(node, ms, detail)`, with `Trace.iterations` and `Trace.total_ms`
- `DocCitation(ordinal, document_id, filename, page_number, quote)`; `Candidate.ref` is a `DocRef(document_id, parent_id, child_id, page_number, filename)`
- `VectorRetriever(session, provider).search(plan) -> list[Candidate]`, `retrieval_over_fetch` deep, not cut to `top_k`
- `build_stages(session, provider, settings) -> Stages` and `answer(query, stages, max_iterations) -> Answer` in `app/core/pipeline.py`
- `QueryService(session, provider).ask(payload) -> QueryResponse`; the provider arrives through `Depends(get_async_provider)`
- `AsyncStubProvider.script(schema, *responses)`; `complete()` raises `KeyError` for a schema nobody scripted
- In tests: `tests/helpers.seed_chunks(session)` seeds 4 parents with 3 children each (child `i` is on page `i // 3 + 1`); the `query_client` fixture yields `(client, provider)`

---

## Task 9b: Format the read path's 14 files

CI on `main` runs `ruff format --check .` on every PR. The first nine tasks were committed before CI existed and without the pre-commit hook (it lives in `.git/hooks` and does not travel with a clone), so 14 files wrap at column 88 while the project sets `line-length = 100`. No line of logic changes.

**Files:**
- Modify: `app/config.py`, `app/core/context_builder.py`, `app/core/generator.py`, `app/core/pipeline.py`, `app/core/retrievers/vector.py`, `app/services/query_service.py`
- Modify: `tests/test_async_llm_provider.py`, `tests/test_chunk_repository.py`, `tests/test_context_builder.py`, `tests/test_generator.py`, `tests/test_parent_expander.py`, `tests/test_pipeline.py`, `tests/test_query_api.py`, `tests/test_vector_retriever.py`

- [x] **Step 1: Verify it is red**

```bash
uv run ruff format --check .
```

Expected: `14 files would be reformatted`.

- [x] **Step 2: Format**

```bash
uv run ruff format .
```

- [x] **Step 3: Check the diff** — `git diff --stat` lists exactly the 14 files above; skim the diff, it is only lines joined and split.

- [x] **Step 4: Run CI's three checks, then stop**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest -q --ignore=tests/test_ocr_service.py
```

Expected: clean, 304 tests PASS. Reinstall the pre-commit hook from the README's "Linting" section so this does not recur.

Proposed commit message: `style: format the read path to the project line length`

---

## Task 9c: Bring `AsyncOpenAIProvider` level with `main`'s sync provider

`complete()` calls JSON mode (`response_format={"type": "json_object"}`) and never shows the model the schema. Against the real OpenAI that breaks twice. JSON mode requires the prompt to mention "JSON", otherwise the API answers 400, and the generator's prompt does not. And if it got through, the model would not know to return an `answer` key, so `schema.model_validate` would fail. No test catches it, because every test goes through `AsyncStubProvider` and this branch has never called the real provider.

`main` solved exactly this for `enrich()` with strict structured outputs. The read path follows, through the SDK's `chat.completions.parse`: it builds the strict JSON schema from the Pydantic model itself, so every schema of R4 → R7 takes the same road and nobody writes a schema by hand. Two smaller gaps go with it: `reasoning_effort` is not passed (the API default is `medium`, and reasoning tokens bill as output), and a missing key shows up on the first request instead of at construction.

**Files:**
- Modify: `shared/llm.py` (class `AsyncOpenAIProvider`)
- Modify: `tests/test_async_llm_provider.py`

**Interfaces:**
- Consumes: `Settings.openai_api_key`, `Settings.openai_chat_model`, `Settings.openai_reasoning_effort`
- Produces: `AsyncOpenAIProvider.complete(messages, schema) -> T`, signature unchanged; `RuntimeError` when the key is missing, and when the model returns no object

- [x] **Step 1: Write the failing tests**

```python
# tests/test_async_llm_provider.py — append


def _async_openai_returning(monkeypatch, parsed=None, requests=None):
    """An AsyncOpenAIProvider whose client is a stand-in that records each
    request and answers with `parsed`, the way the SDK's parse() would. The
    async twin of `_openai_returning` in test_llm_provider.py."""
    from types import SimpleNamespace

    from app.config import get_settings
    from shared.llm import AsyncOpenAIProvider

    async def parse(**kwargs):
        if requests is not None:
            requests.append(kwargs)
        message = SimpleNamespace(parsed=parsed)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    monkeypatch.setattr(get_settings(), "openai_api_key", "sk-test")
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **_: client)
    return AsyncOpenAIProvider()


async def test_complete_asks_for_schema_bound_output_on_the_configured_model(monkeypatch):
    """JSON mode never showed the model the schema, and the API refuses it
    unless a message happens to say "json". Structured outputs bind the reply
    to the schema -- the choice enrich() already made on the write path."""
    from app.config import get_settings

    requests: list[dict] = []
    provider = _async_openai_returning(monkeypatch, Verdict(sufficient=True), requests)

    verdict = await provider.complete([{"role": "user", "content": "?"}], Verdict)

    (request,) = requests
    settings = get_settings()
    assert verdict.sufficient is True
    assert request["response_format"] is Verdict
    assert request["model"] == settings.openai_chat_model
    assert request["reasoning_effort"] == settings.openai_reasoning_effort


async def test_a_reply_with_no_object_raises_instead_of_returning_none(monkeypatch):
    """A refusal, or a reply cut off before the object closed, parses to None.
    Returned as it is, it would surface two calls later as an AttributeError
    with no hint of the cause."""
    provider = _async_openai_returning(monkeypatch, parsed=None)

    with pytest.raises(RuntimeError, match="Verdict"):
        await provider.complete([{"role": "user", "content": "?"}], Verdict)


async def test_openai_without_a_key_fails_at_construction(monkeypatch):
    from app.config import get_settings
    from shared.llm import AsyncOpenAIProvider

    monkeypatch.setattr(get_settings(), "openai_api_key", None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        AsyncOpenAIProvider()
```

- [x] **Step 2: Run them to verify they fail**

```bash
uv run pytest -q tests/test_async_llm_provider.py
```

Expected: 3 FAIL, 5 PASS. The first two with `AttributeError: ... has no attribute 'create'`; the third with `openai.OpenAIError: Missing credentials`, which is the SDK's error and not yet ours.

- [x] **Step 3: Fix the provider**

```python
# shared/llm.py — class AsyncOpenAIProvider; embed_query is unchanged


class AsyncOpenAIProvider:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._chat_model = settings.openai_chat_model
        self._reasoning_effort = settings.openai_reasoning_effort
        self._embed_model = settings.openai_embed_model
        self._dimensions = settings.embed_dimensions

    async def complete(self, messages: list[dict], schema: type[T]) -> T:
        # Structured outputs, strict -- the choice enrich() made above. JSON
        # mode never showed the model the schema, and the API refuses it
        # unless a message happens to say "json".
        response = await self._client.chat.completions.parse(
            model=self._chat_model,
            reasoning_effort=self._reasoning_effort,
            response_format=schema,
            messages=messages,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            # A refusal, or a reply cut off before the object closed.
            raise RuntimeError(f"the model returned no {schema.__name__}")
        return parsed
```

`json` is still used by the synchronous half of the file; do not remove that import.

- [x] **Step 4: Run the tests** — PASS, 8 tests in this file.

- [x] **Step 5: Full suite, then stop**

This task does not call OpenAI. The first real call is Definition of Done item 1.

Proposed commit message: `fix(llm): bind read-path completions to their schema`

---

## Task 9d: The first real question

The nine tasks of R0 have never run against real embeddings, a real model or a real corpus. Do it now, before R1 builds the measuring stick on top: a read-path fault that only shows at Task 12 means the baseline measures a broken system.

**Files:** none. This task only runs and records. If it turns up a fault, the fix is a task of its own, with its own test.

- [x] **Step 1: Ask before running.** This task calls OpenAI: 3 embeddings and 3 chat calls (Global Constraints).

- [x] **Step 2: Bring the stack up on this branch's code**

```bash
docker compose up -d --wait db rabbitmq redis minio
docker compose up -d --wait --force-recreate api
curl -s localhost:8000/openapi.json | grep -c '"/query"'
docker compose exec api sh -c 'echo $LLM_PROVIDER'
```

Expected: `1` (the API has `/query`), then `openai`. The source is bind-mounted and this branch adds no dependency, so recreating the container is enough; the image needs no rebuild.

- [x] **Step 3: Ask three questions**

```bash
ask() {
  curl -s -D - localhost:8000/query -H 'content-type: application/json' \
    -d "$(python3 -c 'import json, sys; print(json.dumps({"question": sys.argv[1]}))' "$1")"
}

ask "Người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu tiền?"
ask "Phương tiện thủy nội địa phải đáp ứng những điều kiện gì để được hoạt động?"
ask "Tỷ giá USD hôm nay là bao nhiêu?"
```

The first two have answers in the corpus, in two different documents. The third does not.

- [x] **Step 4: Check each answer**

| Check | Passes when |
| --- | --- |
| HTTP | 200, the `{code, message, data}` envelope |
| Language | The answer is in Vietnamese |
| Citations | Every claim in the answer carries an `[n]`, and every `[n]` has an entry in `citations` |
| Right page | Open the PDF in `data/` (or the FE viewer) at each citation's `filename` + `page_number`: the passage used really is on that page |
| The outside question | It says plainly that the sources do not answer; no number is invented |
| Latency | Note each `latency_ms` |

- [x] **Step 5: Record the outcome, then stop**

Report one line per question: the question, the document and page cited, right or wrong, `latency_ms`. For a failure, add the `X-Request-ID` header and `docker compose logs api` around that request, then stop: nothing is fixed inside this task.

A pass here is also Definition of Done item 1.

**Outcome (2026-10-10):** all three requests returned 200 through the real provider, and all three wrote their `query_traces` row.

| Question | Citations | Verdict | `latency_ms` |
| --- | --- | --- | --- |
| What is the fine for a car driver ignoring a traffic light? | `nd-168-2024…` p. 25 | **Wrong.** It answered "6 to 8 million" and attributed it to cars, but page 25 is Article 8, penalties for drivers of **special-purpose machinery**. The citation is true to the page; the answer is wrong for the question | 6,635 |
| What must an inland waterway vessel satisfy to operate? | `luat-giao-thong-duong-thuy…` pp. 17 and 41 | Right: the content of Article 24, every class of vessel | 5,566 |
| Today's USD exchange rate? | none | Right: says plainly the sources do not answer | 2,310 |

The read path runs end to end. The fault in the first answer is one of quality, not machinery: three articles of the decree use the same wording for cars, motorbikes and special-purpose machinery, retrieval picked the wrong article, and the prompt gives the model only `(page N)` and the passage text, with no `heading_path` or document name, so the model cannot tell which kind of vehicle a passage is about. Not fixed in this task.

---

# R1 — The Measuring Stick

## Task 10: `query_traces` table

**Files:**
- Create: `app/models/query_trace.py`
- Modify: `app/models/__init__.py`
- Create: `alembic/versions/<hash>_query_traces.py` (CLI-generated)
- Create: `tests/test_query_trace.py`

**Interfaces:**
- Produces: `QueryTrace(id, question, standalone, strategy, iterations, total_ms, total_tokens, citation_count, nodes: JSONB, created_at)`

- [x] **Step 1: Write the failing test**

```python
# tests/test_query_trace.py
import uuid

import pytest
from sqlalchemy import select

from app.models import QueryTrace

pytestmark = pytest.mark.asyncio


async def test_a_trace_row_stores_the_per_node_detail(db_session):
    """Without the per-node breakdown, a poor answer can only be described as
    'it searched badly'. With it you can see the gate asked for a second pass
    and the rerank dropped the right parent at position 11."""
    trace = QueryTrace(
        question="hoa don dien tu",
        iterations=2,
        total_ms=6773,
        citation_count=4,
        nodes=[
            {"node": "retrieve", "ms": 95, "detail": {"vector": 50}},
            {"node": "gate", "ms": 520, "detail": {"sufficient": False}},
        ],
    )
    db_session.add(trace)
    await db_session.flush()

    stored = (await db_session.execute(select(QueryTrace))).scalar_one()

    assert stored.nodes[1]["detail"]["sufficient"] is False
    assert isinstance(stored.id, uuid.UUID)


async def test_optional_columns_stay_null_until_their_stage_exists(db_session):
    """`standalone` arrives at R4 and `strategy` at R5. R1 must not require
    them, or R1 cannot ship before them."""
    trace = QueryTrace(question="q", iterations=1, total_ms=1, citation_count=0, nodes=[])
    db_session.add(trace)
    await db_session.flush()

    assert trace.standalone is None and trace.strategy is None
```

- [x] **Step 2: Run it to verify it fails** — `ImportError: cannot import name 'QueryTrace'`

- [x] **Step 3: Write the model**

```python
# app/models/query_trace.py
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class QueryTrace(Base):
    """One row per answered question. This is what R2 through R8 argue from:
    without it, a regression can only be described, never located."""

    __tablename__ = "query_traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # Written from R4 onward; null before the rewriter exists.
    standalone: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Written from R5 onward. The distribution of this column is the only way
    # to catch a router that has quietly become a constant.
    strategy: Mapped[str | None] = mapped_column(Text, nullable=True)
    iterations: Mapped[int] = mapped_column(Integer, nullable=False)
    total_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    citation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    nodes: Mapped[list] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
```

- [x] **Step 4: Generate the migration, then cut it down to one table**

```bash
uv run alembic revision --autogenerate -m "query traces"
```

**Do not `upgrade` yet.** On `main`'s schema, autogenerate sees what the database owns and the models deliberately do not map, and proposes to drop all of it:

- `ix_child_chunks_embedding_hnsw` — the index every R0 query relies on
- `ix_child_chunks_tsv` and the column `child_chunks.tsv` — what R2's BM25 stands on
- `ix_documents_active_status`, `ix_documents_title_fts`
- `ck_documents_status`

No test would catch them slipping through: without the HNSW index a query still returns the right rows, only slower. Open the generated file and delete every op that is not about `query_traces`. What remains: `down_revision = "4af5845c2429"`, an `upgrade()` that is exactly one `op.create_table("query_traces", ...)` with the model's ten columns, and a `downgrade()` that is exactly one `op.drop_table("query_traces")`.

Only then apply it, and confirm the other three are still there:

```bash
uv run alembic upgrade head
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d child_chunks"' \
  | grep -E 'ix_child_chunks_embedding_hnsw|ix_child_chunks_tsv| tsv '
```

Expected: three lines — the `tsv` column, the `hnsw` index and the `gin (tsv)` index.

`uv run alembic check` will still report the "removed" items above. That is `main`'s standing state, not this task's business; it must simply not mention `query_traces`.

- [x] **Step 5: Run the tests, then stop** — PASS, 2 tests.

Proposed commit message: `feat(db): add the query trace table`

---

## Task 11: Persist the trace on every query

**Files:**
- Modify: `app/services/query_service.py`
- Modify: `tests/test_query_trace.py`
- Modify: `tests/conftest.py`, `tests/test_query_api.py` (moving fixtures)

**Interfaces:**
- Consumes: `QueryTrace`, `Answer.trace`
- Produces: a `query_traces` row per `POST /query`

- [x] **Step 0: Move the read path's two fixtures into `tests/conftest.py`**

`query_client` and `seeded_corpus` live in `tests/test_query_api.py`, so no other test file can see them. The `client` fixture of `conftest.py` is no substitute: it does not override `get_async_provider`, so `POST /query` reaches an unscripted `AsyncStubProvider`, raises `KeyError`, and the test is red for the wrong reason.

Cut both fixtures to the end of `tests/conftest.py`, along with the imports they need (`ASGITransport`, `AsyncClient`, `create_app`, `get_session`, `AsyncStubProvider`, `get_async_provider`, `seed_chunks`), then drop the now-unused imports from `tests/test_query_api.py`.

```bash
uv run ruff check . && uv run pytest -q tests/test_query_api.py
```

Expected: clean, 4 tests PASS as before.

- [x] **Step 1: Write the failing test**

```python
# tests/test_query_trace.py — append. Two imports join the ones at the top of
# the file: `from app.core.generator import Draft`, and `func` beside `select`.


async def test_every_query_writes_exactly_one_trace_row(query_client, db_session, seeded_corpus):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Lap hoa don dieu chinh [1]."))

    response = await client.post("/query", json={"question": "hoa don dien tu"})

    rows = (await db_session.execute(select(QueryTrace))).scalars().all()

    assert response.status_code == 200
    assert len(rows) == 1
    # build_stages records the per-source counts as a node of their own, ahead
    # of the stage that produced them.
    recorded = [n["node"] for n in rows[0].nodes]
    assert recorded[:4] == ["rewrite", "plan", "retrieve_detail", "retrieve"]
    assert rows[0].iterations == 1


async def test_a_failed_trace_write_does_not_fail_the_answer(
    query_client, monkeypatch, seeded_corpus
):
    """Observability must never be able to take down the feature it observes."""
    client, provider = query_client
    provider.script(Draft, Draft(answer="Lap hoa don dieu chinh [1]."))

    def broken(*args, **kwargs):
        raise RuntimeError("trace table is gone")

    # Broken from the inside: _persist_trace imports QueryTrace when it runs.
    # Replacing _persist_trace itself would remove the very try/except this
    # test is about, and the request would fail for a reason of the test's own.
    monkeypatch.setattr("app.models.QueryTrace", broken)

    response = await client.post("/query", json={"question": "hoa don"})

    assert response.status_code == 200


async def test_a_trace_the_database_rejects_leaves_the_session_usable(
    query_client, db_session, monkeypatch, seeded_corpus
):
    """A failed INSERT aborts the transaction it ran in. Without a savepoint the
    answer is computed and then lost to the commit that ends the request."""
    client, provider = query_client
    provider.script(Draft, Draft(answer="Lap hoa don dieu chinh [1]."))

    def questionless(**columns):
        return QueryTrace(**{**columns, "question": None})  # violates NOT NULL

    monkeypatch.setattr("app.models.QueryTrace", questionless)

    response = await client.post("/query", json={"question": "hoa don"})

    assert response.status_code == 200
    # The session still answers: with a bare flush() this raises
    # PendingRollbackError, which is what the request's own commit would hit.
    count = await db_session.execute(select(func.count()).select_from(QueryTrace))
    assert count.scalar_one() == 0
```

- [x] **Step 2: Run them to verify the failure** — the first test: no rows written. The other two pass already, because there is nothing to break yet; they only mean something after Step 3, and the third is the one that goes red if someone turns the savepoint back into a bare `flush()`.

- [x] **Step 3: Wire it in**

```python
# app/services/query_service.py — inside QueryService.ask, before returning

        await _persist_trace(self._session, payload.question, result)

# and at module level

import logging

logger = logging.getLogger(__name__)


async def _persist_trace(session, question: str, result) -> None:
    """Best effort by design: a broken trace table must not take down the
    feature it exists to observe."""
    from app.models import QueryTrace

    try:
        # A savepoint, not a bare flush: a failed INSERT aborts the transaction
        # it ran in, and the commit that ends the request would fail after it.
        # Leaving the block flushes; an error rolls back to the savepoint only.
        async with session.begin_nested():
            session.add(
                QueryTrace(
                    question=question,
                    iterations=result.trace.iterations,
                    total_ms=result.trace.total_ms,
                    citation_count=len(result.citations),
                    nodes=[
                        {"node": n.node, "ms": n.ms, "detail": n.detail} for n in result.trace.nodes
                    ],
                )
            )
    except Exception:
        logger.exception("failed to persist query trace")
```

- [x] **Step 4: Run the tests** — PASS, 5 tests in this file.

- [x] **Step 5: Full suite, then stop**

Proposed commit message: `feat(api): persist a trace row for every query`

---

## Task 12: `eval/dataset.yaml` and `eval/run.py`

The baseline number. Every stage from R2 onward has to beat what this prints today.

**Files:**
- Create: `eval/__init__.py`, `eval/dataset.yaml`, `eval/run.py`
- Create: `tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `ChunkRepository`, `AsyncLLMProvider`, `build_stages`
- Produces:
  - `load_dataset(path) -> Dataset` with `set_a: list[LabelledQuestion]`, `set_b: list[BehaviourQuestion]`
  - `recall_at_k(hits, gold, k) -> float`, `reciprocal_rank(hits, gold) -> float`
  - `python -m eval.run` → prints `recall@10`, `MRR`, `n`; `--full` adds citation accuracy and mean iterations
  - `unreachable_labels(session, questions) -> list[str]`: ids of the labels no retrieval could ever hit
  - `python -m eval.run --check-labels` → checks the labels and stops; calls no provider

- [x] **Step 1: Add the dependency**

```bash
uv add --dev pyyaml
```

- [x] **Step 2: Write the failing test**

```python
# tests/test_eval_harness.py
"""The metrics are three lines of arithmetic and every one of them is easy to
get subtly wrong, so they are unit tested away from the database."""

from eval.run import load_dataset, recall_at_k, reciprocal_rank


class _Hit:
    def __init__(self, sha256: str, page: int):
        self.sha256 = sha256
        self.page_number = page


def test_recall_counts_a_hit_when_the_right_page_appears_anywhere_in_top_k():
    hits = [_Hit("aaa", 1), _Hit("aaa", 7), _Hit("bbb", 12)]

    assert recall_at_k(hits, gold=("aaa", 7), k=10) == 1.0
    assert recall_at_k(hits, gold=("aaa", 9), k=10) == 0.0


def test_recall_respects_k():
    """A gold chunk at position 11 is not in the context and must not count."""
    hits = [_Hit("x", i) for i in range(1, 12)]

    assert recall_at_k(hits, gold=("x", 11), k=10) == 0.0


def test_reciprocal_rank_rewards_position():
    hits = [_Hit("a", 1), _Hit("a", 2), _Hit("a", 3)]

    assert reciprocal_rank(hits, gold=("a", 1)) == 1.0
    assert reciprocal_rank(hits, gold=("a", 2)) == 0.5
    assert reciprocal_rank(hits, gold=("a", 99)) == 0.0


def test_the_dataset_is_anchored_to_sha256_and_page_not_to_ids():
    """Chunk ids are regenerated by every re-chunk. Labels anchored to them die
    the first time the chunking strategy changes -- which is the first knob
    anyone turns."""
    dataset = load_dataset("eval/dataset.yaml")

    for question in dataset.set_a:
        assert len(question.gold_sha256) == 64
        assert question.gold_page >= 1
        assert question.quote, "a quote makes a wrong label visible on sight"


def test_set_a_covers_every_stage_that_will_be_measured_against_it():
    """A stage with no question exercising its failure mode cannot be shown to
    work. These counts are the ones argued for in the spec."""
    dataset = load_dataset("eval/dataset.yaml")
    kinds = [q.kind for q in dataset.set_a]

    assert len(dataset.set_a) == 20
    assert kinds.count("baseline") == 8
    assert kinds.count("code") == 4  # R2 -- vectors miss exact strings
    assert kinds.count("followup") == 4  # R4 -- scores zero until rewriting
    assert kinds.count("summarize") == 4  # R5 -- vector search cannot do this
    assert len(dataset.set_b) == 8
```

The four tests above are pure arithmetic. The fifth needs the database, because it checks the very condition that makes a label mean something:

```python
# tests/test_eval_harness.py — append. The imports at the top of the file become:
#   import pytest
#
#   from eval.run import (
#       LabelledQuestion,
#       load_dataset,
#       recall_at_k,
#       reciprocal_rank,
#       unreachable_labels,
#   )
#   from tests.helpers import seed_chunks


@pytest.mark.asyncio
async def test_a_label_is_reachable_only_if_its_quote_sits_on_its_page(db_session):
    """Recall is scored on (sha256, page). A label whose quote lives in a chunk
    filed under another page can never be hit, and caps recall below 1.0 for a
    reason that has nothing to do with retrieval."""
    document = await seed_chunks(db_session)  # children 0-2 on page 1, 3-5 on page 2

    def label(id_: str, page: int, quote: str) -> LabelledQuestion:
        return LabelledQuestion(
            id=id_,
            kind="baseline",
            question="?",
            gold_sha256=document.sha256_hash,
            gold_page=page,
            quote=quote,
        )

    questions = [
        label("right", 2, "Child   4"),  # case and spacing are not part of a quote
        label("wrong-page", 1, "child 4"),
        label("no-such-text", 2, "khong co"),
    ]

    assert await unreachable_labels(db_session, questions) == ["wrong-page", "no-such-text"]
```

- [x] **Step 3: Run it to verify it fails** — module does not exist.

- [x] **Step 4: Write the dataset skeleton**

Twenty entries in Set A and eight in Set B. The four below are the shape; the
remaining twenty-four are drafted in Step 6. A label is two claims: the quote
is on this page, and this page answers this question. A machine can check the
first; the second still needs a person to read it and agree.

```yaml
# eval/dataset.yaml
# Anchored to (sha256, page): survives every re-chunking. Never to chunk ids.
set_a:
  - id: a01
    kind: baseline
    question: "Quy trinh phat hanh hoa don dien tu gom nhung buoc nao?"
    gold_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
    gold_page: 12
    quote: "nguoi ban lap hoa don dien tu theo dinh dang chuan du lieu"

  - id: a09
    kind: code
    question: "Mau 04/SS-HDDT dung khi nao?"
    gold_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
    gold_page: 31
    quote: "thong bao voi co quan thue theo Mau 04/SS-HDDT"

  - id: a13
    kind: followup
    turns:
      - "Quy trinh phat hanh hoa don dien tu la gi?"
      - "Con khi bi sai thong tin thi sao?"
    question: "Con khi bi sai thong tin thi sao?"
    gold_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
    gold_page: 18
    quote: "truong hop phat hien sai sot, nguoi ban lap hoa don dieu chinh"

  - id: a17
    kind: summarize
    question: "Tom tat Nghi dinh 123/2020/ND-CP"
    gold_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
    gold_page: 1
    quote: "Nghi dinh nay quy dinh ve hoa don, chung tu"

set_b:
  - id: b01
    kind: outside
    question: "Ty gia USD hom nay la bao nhieu?"
    corpus_answers: false

  - id: b06
    kind: web-tempting
    question: "Quy dinh moi nhat ve hoa don dien tu la gi?"
    corpus_answers: true
```

> **The zeroed `gold_sha256` values are placeholders for the labelling pass and
> the test in Step 2 will fail until they are replaced with real hashes.** That
> failure is deliberate: it is what stops the harness being declared done with
> an empty dataset.

> **The four sample items ask about e-invoices; the real corpus does not.** They
> were written before a corpus existed and are shape only. The corpus on `main`
> is ten documents of traffic law (road, inland waterway, maritime, aviation)
> plus a three-page Japanese file; the source PDFs are in `data/`. All 28 items
> must ask about that corpus. To get the hashes and page counts for labelling:

```bash
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
  "select sha256_hash, filename, page_count from documents where completed_at is not null order by filename"'
```

> `gold_page` is the 1-based page number, the same thing `child_chunks.page_number`
> stores, and scoring matches on exactly that number. A quote inside a child
> that starts on the previous page carries the previous page's number: such a
> label can never be hit. `--check-labels` in Step 5 exists to catch just that.

- [x] **Step 5: Write the runner**

```python
# eval/run.py
"""Retrieval-only by default: recall@10 and MRR need no chat call, only one
embedding per question, so the metric you check after every tuning change costs
next to nothing. `--full` runs the
whole pipeline and is only needed for citation accuracy and fabrication rate.
"""

import argparse
import asyncio
import unicodedata
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class LabelledQuestion:
    id: str
    kind: str
    question: str
    gold_sha256: str
    gold_page: int
    quote: str
    turns: tuple[str, ...] = ()


@dataclass(frozen=True)
class BehaviourQuestion:
    id: str
    kind: str
    question: str
    corpus_answers: bool


@dataclass(frozen=True)
class Dataset:
    set_a: list[LabelledQuestion]
    set_b: list[BehaviourQuestion]


def load_dataset(path: str) -> Dataset:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Dataset(
        set_a=[LabelledQuestion(**{**q, "turns": tuple(q.get("turns", ()))}) for q in raw["set_a"]],
        set_b=[BehaviourQuestion(**q) for q in raw["set_b"]],
    )


def recall_at_k(hits, gold: tuple[str, int], k: int) -> float:
    """Did the right page reach the context at all? This is the hard ceiling:
    a chunk that never arrives cannot be recovered by any prompt."""
    return float(any((h.sha256, h.page_number) == gold for h in hits[:k]))


def reciprocal_rank(hits, gold: tuple[str, int]) -> float:
    """Where did it land? Reranking does not change recall -- it brings nothing
    new in -- so this is the metric that shows whether it earned its keep."""
    for position, hit in enumerate(hits, start=1):
        if (hit.sha256, hit.page_number) == gold:
            return 1.0 / position
    return 0.0


@dataclass(frozen=True)
class Hit:
    sha256: str
    page_number: int


def _squash(text: str) -> str:
    """Case, Unicode form and whitespace are not part of a quote: OCR and PDF
    text extraction disagree on all three."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


async def unreachable_labels(session, questions: list[LabelledQuestion]) -> list[str]:
    """Ids of the labels no retrieval could ever hit: no child of that
    document, filed under that page, contains the quote."""
    from sqlalchemy import select

    from app.models import ChildChunk, Document

    missing = []
    for question in questions:
        rows = await session.execute(
            select(ChildChunk.content)
            .join(Document, Document.id == ChildChunk.document_id)
            .where(
                Document.sha256_hash == question.gold_sha256,
                ChildChunk.page_number == question.gold_page,
            )
        )
        quote = _squash(question.quote)
        if not any(quote in _squash(content) for (content,) in rows):
            missing.append(question.id)
    return missing


async def _check_labels() -> None:
    from app.utils.database import AsyncSessionLocal

    dataset = load_dataset("eval/dataset.yaml")
    async with AsyncSessionLocal() as session:
        missing = await unreachable_labels(session, dataset.set_a)
    if missing:
        # Measured anyway, each of these would score zero whatever retrieval did.
        raise SystemExit(f"unreachable labels, fix them first: {', '.join(missing)}")
    print(f"labels           {len(dataset.set_a)} reachable")


async def _main(full: bool) -> None:
    from sqlalchemy import select

    from app.config import get_settings
    from app.core.contracts import Filters, Plan, Query
    from app.core.deps import build_stages
    from app.core.pipeline import answer as run_pipeline
    from app.core.retrievers.vector import VectorRetriever
    from app.models import Document
    from app.utils.database import AsyncSessionLocal
    from shared.llm import get_async_provider

    dataset = load_dataset("eval/dataset.yaml")
    settings = get_settings()
    if settings.llm_provider != "openai":
        # The corpus is embedded by OpenAI. A stub query vector is noise
        # against it, and the recall it yields looks exactly like a number.
        raise SystemExit("eval needs LLM_PROVIDER=openai: the corpus holds real embeddings")
    provider = get_async_provider()
    top_k = settings.retrieval_top_k

    recalls: list[float] = []
    ranks: list[float] = []
    cited_right_page: list[float] = []
    iterations: list[int] = []

    async with AsyncSessionLocal() as session:
        # Labels are anchored to (sha256, page); candidates carry document_id.
        # One lookup bridges them without widening DocRef, which is frozen.
        rows = await session.execute(select(Document.id, Document.sha256_hash))
        sha_by_id = {row.id: row.sha256_hash for row in rows}

        retriever = VectorRetriever(session, provider)

        for question in dataset.set_a:
            gold = (question.gold_sha256, question.gold_page)
            candidates = await retriever.search(
                Plan(strategy="VECTOR_ONLY", queries=(question.question,), filters=Filters())
            )
            hits = [Hit(sha_by_id[c.ref.document_id], c.ref.page_number) for c in candidates]
            recalls.append(recall_at_k(hits, gold, top_k))
            ranks.append(reciprocal_rank(hits, gold))

            if not full:
                continue

            result = await run_pipeline(
                Query(text=question.question),
                build_stages(session, provider, settings),
                max_iterations=settings.retrieval_max_iterations,
            )
            iterations.append(result.trace.iterations)
            cited_right_page.append(
                float(
                    any(
                        getattr(c, "page_number", None) == question.gold_page
                        and sha_by_id.get(getattr(c, "document_id", None)) == question.gold_sha256
                        for c in result.citations
                    )
                )
            )

    n = len(recalls)
    print(f"n                {n}")
    print(f"recall@{top_k:<10} {sum(recalls) / n:.3f}")
    print(f"MRR              {sum(ranks) / n:.3f}")
    if full:
        print(f"citation acc.    {sum(cited_right_page) / len(cited_right_page):.3f}")
        print(f"mean iterations  {sum(iterations) / len(iterations):.2f}")


async def _run(full: bool, check_only: bool) -> None:
    # One event loop for both: the engine's pooled connections belong to the
    # loop that opened them.
    await _check_labels()
    if not check_only:
        await _main(full)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also run generation")
    parser.add_argument(
        "--check-labels",
        action="store_true",
        help="verify every Set A quote sits on its gold page, then stop; calls no provider",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.full, args.check_labels))
```

- [x] **Step 6: Draft the 28 items from the corpus**

The agent drafts them by reading `child_chunks.content` over SQL. No OpenAI call and **no retrieval run**: picking questions out of search results makes the system's own output its answer key.

| Rule | Why |
| --- | --- |
| The exact count of each `kind` the Step 2 test demands: 8 `baseline`, 4 `code`, 4 `followup`, 4 `summarize`; Set B: 5 `outside`, 3 `web-tempting` | Each later stage needs questions that probe its own weakness |
| The 8 `baseline` questions span at least 6 documents | A set that asks about one decree measures that decree, not the corpus |
| Questions are worded as a user would ask, not copied from the chunk | A question that repeats the chunk verbatim makes vector search look better than it is |
| `quote` is a verbatim span of about 15 words at most, inside **one** child, free of Markdown or HTML markup | `--check-labels` matches within a single child; a span straddling two would be reported unreachable |
| `code`: the question contains a document number, an article number or a form code | This is what vectors miss and R2 has to rescue |
| `followup`: has `turns`; `question` is the last turn and cannot stand alone | Scoring zero until R4 is by design |
| `summarize`: `gold_page` is the page carrying the document's scope article | Vector search cannot do this until R5 |

- [x] **Step 7: Let the machine check the labels**

```bash
uv run python -m eval.run --check-labels
```

Expected: `labels           20 reachable`. This calls no provider. For every label it reports, fix the `quote` or the `gold_page` until it is clean.

- [ ] **Step 8: Stop and wait for a person to review the set**

The machine has shown the quote is on that page. It cannot show that the page **answers** the question, nor whether the question resembles what a real user would ask. The reviewer reads all 28 items, edits or replaces them, then reruns Step 7. After this review the set is **frozen** (spec §8.4): adding or dropping one question voids every before/after comparison.

- [ ] **Step 9: Record the baseline**

```bash
uv run python -m eval.run
```

This command **calls OpenAI**: 20 embeddings, one per Set A question, because the corpus is embedded with `text-embedding-3-large` and a stub vector against it is noise. `--full` adds 20 chat calls. Ask before running it (Global Constraints). With `LLM_PROVIDER=stub` the runner refuses to run rather than print a meaningless number.

**Write the printed `recall@10` and `MRR` into `docs/retrieval-design.md` §8 as the
R1 baseline.** A number that is not written down is not a baseline.

- [ ] **Step 10: Full suite, then stop**

Proposed commit message: `feat(eval): add the retrieval evaluation harness`

---

## Task 13: Make the README and the spec describe the system as built

`README.md` on `main` is still the scaffold's: the name `rag-beginner`, the sentence "There is no RAG logic here yet", a table of CRUD endpoints that no longer exist, and a "Deliberate gaps" section saying `embedding` is always `NULL`. Phase 1 merged without it being touched. This task rewrites it to match the code after R1, and comes last so it is not written twice. `docs/retrieval-design.md` also carries a few facts from before Phase 1; they are corrected in the same pass, because this plan declares that the spec wins where the two disagree.

**Files:**
- Modify: `README.md`
- Modify: `docs/retrieval-design.md`, `docs/system-design.md`

**Constraints:**
- Keep the two section names `## Linting` and `## Testing`: `.github/workflows/ci.yml` points at them by name.
- The README stays in English.
- The README is a way in, not a copy of the design: each section summarises, then links to `docs/system-design.md` or `docs/retrieval-design.md`.
- Copy no value out of `.env`. The configuration table takes names and defaults from `.env.example` and `app/config.py`.

- [x] **Step 1: Take the facts from the code, not from memory**

```bash
grep -n '@router\.' app/controllers/*.py                  # endpoints
grep -v '^#' .env.example | grep -v '^$' | cut -d= -f1    # configuration variables
grep -n '^  [a-z-]*:$' docker-compose.yml                 # services
```

- [x] **Step 2: Rewrite section by section**

| Section | What to do |
| --- | --- |
| Title and opening | `rag-beginner` becomes `s1mpleRAG`. Two or three sentences: ingest PDFs, answer questions with citations down to the page; an API, three workers, OCR and a frontend. Drop "There is no RAG logic here yet". |
| *(new)* Architecture | The five stages S1 → S5 and the read path, a short paragraph each. A table of the `docker-compose.yml` services and what each is for: `db`, `rabbitmq`, `redis`, `minio`, `minio-init`, `chandra`, `api`, `worker-ocr`, `worker-cpu`, `worker-llm`. Link to both design documents and `docs/diagrams/`. |
| Layout | Add `app/core/`, `worker/`, `shared/`, `web/`, `eval/`, `tests/`, `docs/`, `alembic/`. Keep the sentence about controller → service → repository → model. |
| *(new)* Requirements | Docker and Compose, `uv`, `pnpm` for `web/`. A 16 GB NVIDIA GPU for `chandra`, not shared with anything else. On WSL the RAM limit has to be raised (link to the host RAM section of `docs/system-design.md`). |
| Running locally (uv) | Keep the `uv` paragraphs. Correct: the API needs RabbitMQ, Redis and MinIO as well as `db`. `LLM_PROVIDER=stub` is the default; `openai` costs money. Add how to run `web/` (`pnpm install`, `pnpm dev`). |
| Running with Docker | `docker compose up -d --build` now brings up `chandra` too: a model download, and a GPU. Add the way to run without a GPU: leave out `chandra` but still run `worker-ocr` with `--no-deps`, since every document starts on the `ocr` queue. Note that changing `task_routes` needs an `api` restart, since the publisher is what decides the queue. |
| Response envelope | Keep. Change the examples to real payloads: `id` is a UUID, the 404 comes from `/documents/{id}/status`. |
| Endpoints | Replace the whole table with the one below. |
| Linting | Keep as is. Add one sentence: CI runs these same two commands on every PR into `main`. |
| Testing | Change the start-up command to the four services, as CI does. Add: `tests/test_ocr_service.py` needs `chandra` and a GPU; the `real_llm` marker needs a key and costs money; every other test uses the stub. |
| Migrations | Keep. Add the warning from Task 10 Step 4: autogenerate proposes to drop the HNSW index, the `tsv` column and the indexes and constraint the database owns; those ops must be deleted from the generated file. |
| Configuration | Replace the table with every variable of `.env.example`, grouped: database, app, RabbitMQ, Redis, OCR, MinIO, LLM, limits, rate limits, retrieval. Keep the two paragraphs explaining `DB_*` and `DATABASE_URL`. |
| *(new)* Evaluation | `uv run python -m eval.run` and `--full`; needs `LLM_PROVIDER=openai`; the baseline is in `docs/retrieval-design.md` §8. |
| Deliberate gaps | Replace entirely with the real ones: no authentication; Redis and RabbitMQ have no volume; `locations` is not produced yet; every `RETRIEVAL_*_ENABLED` flag is off because R2 → R8 are not built; the frontend is not wired to `POST /query`. |

The endpoint table, from the code as it stands:

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | Liveness. |
| POST | `/documents/upload-url` | A presigned PUT for one PDF; 409 if a file with that SHA-256 already exists. |
| POST | `/documents` | 202; registers the uploaded file and sends it down the pipeline. |
| GET | `/documents` | Paginated list, with live progress. |
| GET | `/documents/{id}/status` | One document's status, metadata and progress; 404 if absent. |
| GET | `/documents/{id}/file-url` | A short-lived URL (300 seconds) for the PDF viewer to fetch the file. |
| DELETE | `/documents/{id}` | 200 with `data: null`; 404 if absent. |
| POST | `/query` | An answer with citations. |

- [x] **Step 3: Check the README against the code**

```bash
# every endpoint the code declares is in the README, and nothing else is
grep -o '`/[a-z{}_/-]*`' README.md | sort -u
grep -n '@router\.' app/controllers/*.py

# every variable of .env.example is in the configuration table
for v in $(grep -v '^#' .env.example | grep -v '^$' | cut -d= -f1); do
  grep -q "\`$v\`" README.md || echo "missing: $v"
done

# no leftovers of the scaffold
grep -n -i 'rag-beginner\|no RAG logic\|never populated\|PATCH' README.md
```

Expected: the two endpoint lists agree; no `missing:` line; the last grep prints nothing.

- [x] **Step 4: Run what the README tells the reader to run** — every command block under "Running", "Linting", "Testing" and "Migrations" is run once, on a fresh clone or on the dev machine with the stack down. Where a command does not work, fix the README, not the code.

- [x] **Step 5: Correct the spec's stale facts**

Facts only. No argument is rewritten.

| File, place | Says | Becomes |
| --- | --- | --- |
| `retrieval-design.md` §2 (decision 7), §5.4, §9 | the reranker follows the "same pattern as `docling`", `docling_service/` | `chandra` — the OCR container that exists; Docling was replaced in Phase 1 |
| `retrieval-design.md` §6.1 | "`text-embedding-3-small` aligns Vietnamese and English weakly" | The model is now `text-embedding-3-large`. Add one sentence saying so: the case for bilingual sub-queries rests on a weakness of `-small` and has to be re-measured on the eval set before R4 is built. Do **not** delete the argument |
| `retrieval-design.md` §8 | no baseline yet | Written at Task 12 Step 9; confirm it is there |
| `retrieval-design.md` §11 | "What Phase 1 must deliver", in the future tense, pointing at `78bfb53` | Recast as delivered: all five items are on `main`. Keep the table, since it still says which columns R2 and R5 stand on |
| `retrieval-design.md` §12, the §2 decision 3 row | "`text-embedding-3-small` … Model unchanged" | Phase 1 moved to `-large` shortened to 1536 dimensions (MIRACL 54.9 vs 44.0); the corpus is Vietnamese plus one Japanese file |
| `retrieval-design.md` §14 | splitting `shared/llm.py` "after Phase 1 merges" | The condition is met; the split is still unscheduled |
| `system-design.md`, "Phase 2 — Retrieval (deferred)" | "deferred", lists `retriever.py` | R0 exists on `phase-2`; point to `retrieval-design.md` and this plan |

```bash
grep -n -i 'docling\|embedding-3-small\|78bfb53' docs/retrieval-design.md
```

Expected: only lines that tell history (the §12 table recounting the old decision), none describing the present.

- [ ] **Step 6: Stop and report**

Proposed commit message: `docs: describe the system as built`

---

## Task 14: Assess DeepSeek for the LLM calls

Added 2026-10-10 on request. The question: should the system use DeepSeek, and for which calls. This task is research and a recommendation; **no code changes**. If the answer is yes, adding the provider is a plan of its own. This task is not part of the R0 + R1 Definition of Done.

**Files:** no code. The outcome is written under "Outcome" just below.

The system calls an LLM in five places, and each asks something different:

| Call | Where | What it needs from a provider |
| --- | --- | --- |
| S3 enrich | worker, `OpenAIProvider.enrich` | Strict structured output with a closed enum; Vietnamese; thousands of chunks per document; clear rate limits |
| S4 embed | worker, `OpenAIProvider.embed` | An embeddings API; changing model or dimension means re-ingesting the whole corpus |
| Query embedding | API, `AsyncOpenAIProvider.embed_query` | The same model as S4 |
| Answer generation | API, `AsyncOpenAIProvider.complete` | A reply bound to a Pydantic schema; low latency; Vietnamese |
| Decision nodes R4 → R7 | API, not built | Several small schema-bound calls per question: price and latency per call decide it |

- [x] **Step 1: Get today's facts.** Models, context windows, prices (cache hit, cache miss, output), whether there is an embeddings API, how far structured output goes (strict `json_schema` or JSON mode only), OpenAI SDK compatibility, rate limits, data policy. Every number carries the URL of the page it was read from; what cannot be checked is marked "not verified".

- [x] **Step 2: Get the current prices of the two models in use** (`OPENAI_CHAT_MODEL` and `text-embedding-3-large`) to compare against.

- [x] **Step 3: Estimate the cost on each provider**, with the arithmetic: enriching one 600-chunk document; 1,000 questions with generation only; 1,000 questions once four decision nodes exist.

- [x] **Step 4: Write a recommendation per call, then stop.** Use it, keep OpenAI, or not possible; with the risks and what `shared/llm.py` would have to change.

**Outcome** (researched 2026-10-10 from both vendors' official docs and price lists; no API was called to test):

**Recommendation: do not move to DeepSeek, for any of the five calls.**

| Call | Verdict | Main reason |
| --- | --- | --- |
| S3 enrich | Keep OpenAI | 1.3 to 2.6 times dearer on this workload; DeepSeek's Chat Completions accepts only `text` or `json_object`, not the strict `json_schema` that `_ENRICHED_SCHEMA` uses |
| S4 embed, query embedding | Not possible | DeepSeek's API reference lists no embeddings endpoint |
| Answer generation | Keep OpenAI | 1.4 to 2.8 times dearer; Vietnamese office hours fall in DeepSeek's peak window; data is processed and stored in China |
| Decision nodes R4 → R7 | Keep OpenAI | The same price gap; `chat.completions.parse()` (Task 9c) does not work against DeepSeek |

Prices per 1M tokens, from each vendor's pricing page:

| Model | Input | Output |
| --- | --- | --- |
| `gpt-6-luna` (in use) | $0.10 | $0.50 |
| `deepseek-flash`, peak | $0.30 | $1.20 |
| `deepseek-flash`, off-peak | $0.15 | $0.60 |
| `deepseek-v4-pro`, peak | $1.32 | $3.96 |

Estimates, caching not counted:

| Job | `gpt-6-luna` | `deepseek-flash` peak | `deepseek-flash` off-peak |
| --- | --- | --- | --- |
| Enrich one 600-chunk document (99K in, 45K out) | $0.032 | $0.084 | $0.042 |
| 1,000 questions, generation only (3.0M in, 0.30M out) | $0.45 | $1.26 | $0.63 |
| 1,000 questions with four decision nodes (6.2M in, 0.62M out) | $0.93 | $2.60 | $1.30 |

DeepSeek's output price is above `gpt-6-luna`'s even off-peak, so it only comes out cheaper when most of the input is a cache hit.

The one thing worth trying is answer quality: on Vals' English legal benchmarks `deepseek-flash` beats `gpt-6-luna`. That is not evidence for Vietnamese; to find out, measure with `eval.run --full` after R1.

Not verified: Vietnamese quality (no benchmark found); latency at low reasoning effort; retention and training policy for data sent through the API (DeepSeek's two terms documents do not say); whether DeepSeek's Responses API really enforces a schema.

Sources: `api-docs.deepseek.com` (pages `quick_start/pricing`, `api/create-chat-completion`, `guides/json_mode`, `quick_start/rate_limit`), `developers.openai.com/api/docs/pricing`, `vals.ai`.

---

## Definition of Done — R0 + R1

1. `POST /query` answers a question about a real ingested PDF, with a citation naming a document and a page that actually contains the quoted text (Task 9d).
2. All eight seams exist; five are no-ops; `tests/test_pipeline.py` proves the loop iterates, re-enters at `plan`, uses the refined query, and accumulates context.
3. A stage that raises degrades to its no-op and the request still succeeds.
4. Every stage flag is off and `RETRIEVAL_MAX_ITERATIONS` is `1` in `.env.example`.
5. `hnsw.ef_search` is set per session and is at least twice `RETRIEVAL_OVER_FETCH`.
6. No prompt contains a UUID; a phantom citation is dropped and logged.
7. Every query writes exactly one `query_traces` row, and a broken trace write does not fail the answer.
8. `eval/dataset.yaml` holds 20 labelled Set A items and 8 Set B items, anchored to `(sha256, page)`; `--check-labels` reports all 20 reachable, and a person has reviewed the set.
9. `uv run python -m eval.run` prints `recall@10` and `MRR`, and **the numbers are written into the spec**.
10. Full suite green; `ruff check .` and `ruff format --check .` clean; CI green on the PR into `main`.
11. `worker/` imports nothing from `app/core/`, and `app/core/` imports nothing from `worker/`.
12. `AsyncOpenAIProvider` asks for schema-bound structured output, and a real question through OpenAI comes back as a valid `Draft` (which is what item 1 quietly requires).
13. The `query_traces` migration drops nothing: the HNSW index, the `tsv` column and its index are all still there after `alembic upgrade head`.
14. `README.md` describes the system as built: no `rag-beginner` left, the endpoint table matches the code, the configuration table matches `.env.example`. `docs/retrieval-design.md` no longer describes the present with `docling` or `text-embedding-3-small`.
