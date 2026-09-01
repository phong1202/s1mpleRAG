# RAG Ingestion & Retrieval — Final System Design

**Rev 3 · build-ready · extends the existing `rag-beginner` FastAPI scaffold**

Rev 2 (`ingestion-architecture.md`) argued the generic case for a staged pipeline. This document is
the concrete build design: every decision settled, every boundary named, sized for local Docker
Compose while keeping the scale-up path a config change rather than a rewrite.

---

## 1. Scope and shape

Two independent paths through one codebase:

| Path | Runs in | Trigger | Code lives in |
|------|---------|---------|---------------|
| **Write — ingestion** | Celery workers (sync) | PDF upload | `worker/` |
| **Read — retrieval** | FastAPI (async) | User question | `app/core/` |

They share the ORM models, the config, and the embedding client. Nothing else. The API never imports
Celery task code; the worker never imports FastAPI.

**In scope:** PDF upload → parse → chunk → contextualize → embed → persist, and question → retrieve →
generate. **Out of scope for this revision:** reranking, multi-tenancy, auth, streaming responses,
evaluation harness.

---

## 2. Decisions ledger

Everything settled across the review. This table is the contract — if the implementation disagrees
with a row here, the implementation is wrong.

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Base | Extend `rag-beginner` | FastAPI + async SQLAlchemy + pgvector already built and tested |
| 2 | `app/core/` | RAG **read path only** | API-side retrieval/generation; ingestion is worker code |
| 3 | Embedding model | `text-embedding-3-small` @ 1536 | 6.5× cheaper than `3-large` for ~2 MTEB points |
| 4 | HNSW | Plain `CREATE INDEX` | Bulk-load path is for backfills; irrelevant at this scale |
| 5 | Deployment | Local Docker Compose | Study/test scale |
| 6 | LLM provider | OpenAI (`gpt-4o-mini`) | Single provider; caching caveat accepted (§12) |
| 7 | Docling | Independent container | Isolates VRAM/RAM from worker concurrency |
| 8 | Object storage | MinIO container | S3-API compatible — swap to S3 by URL later |
| 9 | Broker | **RabbitMQ** | Real AMQP acks; no `visibility_timeout` guessing |
| 10 | Rate limiter | **Redis** (only job) | RabbitMQ can't do atomic token buckets |
| 11 | Result backend | **None** — Postgres is authoritative | `documents.status` is durable and queryable |
| 12 | Backpressure | Requeue with delay + jitter | Sleeping workers look healthy while idle |
| 13 | Worker pools | 3 containers | 5-pool split is a `task_routes` change later |
| 14 | Worker DB | **Sync** SQLAlchemy (`psycopg`) | No event-loop binding; plain `def` tasks |
| 15 | API DB | **Async** (`asyncpg`) — unchanged | Existing scaffold |
| 16 | Upload | Presigned PUT to MinIO | Keeps large transfers off the API request path |

---

## 3. Container topology

```
┌─────────────┐        ┌──────────────┐
│   client    │───────▶│  api :8000   │  FastAPI, async
└─────────────┘        └──────┬───────┘
       │                      │ publish
       │ presigned PUT        ▼
       │              ┌──────────────┐
       │              │ rabbitmq     │  broker :5672 / UI :15672
       │              └──────┬───────┘
       ▼                     │ consume
┌─────────────┐              ├────────────────────┐
│ minio :9000 │              ▼                    ▼
│  UI :9001   │      ┌──────────────┐     ┌──────────────┐
└──────┬──────┘      │ worker-cpu   │     │ worker-llm   │
       │             │ -Q cpu -c 8  │     │ -Q llm -c 20 │
       │             │ S2, S5       │     │ S3, S4       │
       │             └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │                    │  S1 over HTTP      │ token bucket
       │             ┌──────▼───────┐     ┌──────▼───────┐
       └────────────▶│ docling      │     │ redis :6379  │
                     │ :8100        │     │ rate limiter │
                     └──────────────┘     └──────────────┘
                                │
                         ┌──────▼──────────────────┐
                         │ db :5433                │
                         │ postgres 16 + pgvector  │
                         └─────────────────────────┘
```

**Eight containers.** RabbitMQ (~200MB) and Docling (~1–2GB with models) are the heavy ones. If your
machine strains, fold `docling` into `worker-cpu` for local runs via a compose override — keep it
separate in the file you'd deploy.

| Service | Image / build | Ports | Notes |
|---------|---------------|-------|-------|
| `api` | build `.` | 8000 | FastAPI, async, existing |
| `worker-cpu` | build `.` | — | `celery -A worker.celery_app worker -Q cpu -c 8` |
| `worker-llm` | build `.` | — | `celery -A worker.celery_app worker -Q llm -c 20` |
| `docling` | build `./docling_service` | 8100 | FastAPI wrapper, models loaded at startup |
| `rabbitmq` | `rabbitmq:3-management` | 5672, 15672 | Management UI for queue inspection |
| `redis` | `redis:7-alpine` | 6379 | Token buckets only |
| `db` | `pgvector/pgvector:pg16` | 5433→5432 | Existing; 5432 taken by another container |
| `minio` | `minio/minio` | 9000, 9001 | Buckets: `raw`, `staging` |

---

## 4. Repository layout

```
rag-beginner/
├── main.py                          # unchanged — uvicorn only
├── app/                             # ── FastAPI (async) ──────────────
│   ├── app.py                       # + mounts ingestion & query routers
│   ├── config.py                    # + OpenAI, MinIO, RabbitMQ, Redis settings
│   ├── core/                        # ★ NEW — RAG read path, API only
│   │   ├── retriever.py             #   embed query → vector search → parent expansion
│   │   ├── context_builder.py       #   assemble parents into a prompt, token-budgeted
│   │   ├── generator.py             #   OpenAI chat call + citation assembly
│   │   └── contracts.py             #   Query, RetrievedParent, Answer, Citation
│   ├── controllers/
│   │   ├── health_controller.py     # unchanged
│   │   ├── ingestion_controller.py  # ★ upload-url, register, status
│   │   └── query_controller.py      # ★ POST /query
│   ├── services/
│   │   ├── ingestion_service.py     # ★ dedup, register, publish to broker
│   │   └── query_service.py         # ★ orchestrates app/core/
│   ├── repositories/
│   │   ├── document_repository.py   # reworked for the new documents shape
│   │   └── chunk_repository.py      # ★ retrieval queries (read-only)
│   ├── models/                      # ── SHARED with worker ───────────
│   │   ├── base.py
│   │   ├── document.py              # reworked: file-level entity
│   │   ├── parent_chunk.py          # ★
│   │   └── child_chunk.py           # ★
│   ├── schemas/                     # response envelope, unchanged pattern
│   ├── utils/                       # logging, async database — unchanged
│   ├── exceptions/                  # + ingestion-specific ErrorCodes
│   └── middleware/                  # unchanged
│
├── shared/                          # ★ NEW — imported by BOTH sides
│   ├── embeddings.py                #   OpenAI embedding client (query + chunks)
│   ├── storage.py                   #   MinIO/S3 client
│   └── rate_limiter.py              #   Redis token bucket
│
├── worker/                          # ── Celery (sync) ────────────────
│   ├── celery_app.py                #   app, task_routes, retry defaults
│   ├── stages.py                    #   @task wrappers; the S1→S5 chain
│   ├── parsing.py                   #   PyMuPDF + docling client + detection
│   ├── chunking.py                  #   sanitize + hierarchical chunking
│   ├── enrichment.py                #   metadata + contextualizer
│   ├── embedding.py                 #   chunk embedding (calls shared/)
│   ├── persistence.py               #   idempotent bulk upsert
│   └── db.py                        #   SYNC engine + sync repositories
│
├── docling_service/                 # ★ NEW — its own container
│   ├── main.py                      #   FastAPI: POST /parse
│   ├── Dockerfile
│   └── requirements.txt
│
├── alembic/versions/                # CLI-generated only
├── tests/
├── docker-compose.yml
└── docs/
```

**Why `shared/` and not `app/shared/`:** the worker must not import anything under `app/` that pulls
in FastAPI. Keeping shared code in a sibling package makes that boundary structural rather than a
convention people remember.

---

## 5. Data model

### 5.1 Migrating from the scaffold — read this first

The existing scaffold has one flat table:

```sql
documents (id, title, content, embedding vector(1536), created_at, updated_at)
```

That was a CRUD demo. In the real system, a **document is an uploaded file**, not a title/content
pair, and text lives in chunks. This is a **breaking change**, and it needs stating plainly:

- `documents.title` / `documents.content` / `documents.embedding` are **dropped**.
- `documents` gains file-level columns (hash, object key, status, stage…).
- `parent_chunks` and `child_chunks` are **new**.
- The demo CRUD endpoints (`POST/PATCH/DELETE /documents` with a JSON body) **no longer make sense** —
  a document is created by uploading a PDF. They're replaced by the ingestion endpoints in §9.
- **Roughly 10 of the existing 55 tests** target that CRUD surface and will be replaced, not adapted.
  The infrastructure tests (config, logging, middleware, exception handlers, envelope, transaction
  boundary) are unaffected — that's ~45 tests carried over untouched.

Since there's no production data, one Alembic migration does the whole thing. Generated by CLI, as
always.

### 5.2 Schema

```sql
-- ── file-level entity + pipeline state ──────────────────────────────
CREATE TABLE documents (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sha256_hash    char(64)    NOT NULL UNIQUE,     -- closes the dedup race
  filename       text        NOT NULL,
  object_key     text        NOT NULL,            -- MinIO: raw/{sha256}.pdf
  size_bytes     bigint      NOT NULL,
  page_count     int,                             -- known after S1

  status         text        NOT NULL DEFAULT 'QUEUED',
  stage          text,                            -- resume pointer: last COMPLETED stage
  attempts       int         NOT NULL DEFAULT 0,
  failed_stage   text,
  last_error     text,

  created_at     timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now(),
  completed_at   timestamptz
);
CREATE INDEX ON documents (status) WHERE status NOT IN ('COMPLETED', 'DEAD_LETTER');

-- ── generation context ──────────────────────────────────────────────
CREATE TABLE parent_chunks (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id   uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index   int  NOT NULL,                    -- deterministic from parse output
  content       text NOT NULL,
  token_count   int  NOT NULL,
  page_start    int  NOT NULL,
  page_end      int  NOT NULL,
  UNIQUE (document_id, chunk_index)               -- natural key → idempotent upsert
);
CREATE INDEX ON parent_chunks (document_id);

-- ── retrieval units ─────────────────────────────────────────────────
CREATE TABLE child_chunks (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id    uuid NOT NULL REFERENCES documents(id)      ON DELETE CASCADE,
  parent_id      uuid NOT NULL REFERENCES parent_chunks(id)  ON DELETE CASCADE,
  chunk_index    int  NOT NULL,
  content        text NOT NULL,                   -- raw chunk text
  contextualized text NOT NULL,                   -- context sentence + content (what we embed)
  page_number    int  NOT NULL,                   -- citation precision lives HERE
  token_count    int  NOT NULL,
  embedding      vector(1536) NOT NULL,
  category       text,                            -- soft metadata, closed enum
  UNIQUE (document_id, chunk_index)
);
CREATE INDEX ON child_chunks (parent_id);
CREATE INDEX ON child_chunks USING hnsw (embedding vector_ip_ops)
  WITH (m = 16, ef_construction = 64);
```

**Why `vector_ip_ops`:** embeddings are L2-normalized at S4, so inner product is equivalent to cosine
and cheaper. The normalization is not optional — if it's skipped, the index silently returns wrong
neighbors with no error.

---

## 6. Ingestion pipeline

Seven original nodes, five stages, grouped by contended resource.

### S1 — Parse · queue `cpu` → HTTP to `docling`

**In:** `document_id`  **Out:** `staging/{doc_id}/parsed.json`

```python
{
  "page_count": 42,
  "pages": [
    {"page": 1, "markdown": "...", "source": "pymupdf", "confidence": 1.0},
    {"page": 2, "markdown": "...", "source": "docling", "confidence": 0.91}
  ]
}
```

1. Fetch PDF from MinIO `raw/{sha256}.pdf`.
2. **Guard:** encrypted → `AppException(PDF_ENCRYPTED)`, straight to `DEAD_LETTER`. Page count over
   limit → `PDF_TOO_LARGE`.
3. PyMuPDF fast pass, all pages.
4. **Scanned detection:** `total_text_chars / page_count < 100` → route *every* page to Docling.
5. **Per-page routing:** pages with detected tables/images → Docling, one HTTP call per page.
6. **Per-page timeout 90s** → fall back to PyMuPDF text for that page, `confidence: 0.0`, log it.
7. Write `parsed.json`, update `documents.page_count`.

The task runs on `worker-cpu` but the *work* happens in the `docling` container — the worker holds an
HTTP connection, not model weights. That's what makes S1 safe to run at concurrency 8.

**Docling service contract:**

```
POST /parse
  { "object_key": "raw/abc123.pdf", "pages": [2, 7, 11] }
→ { "pages": [ {"page": 2, "markdown": "...", "confidence": 0.91} ] }
```

It pulls from MinIO itself — no multi-MB payloads over the wire. Models load once at process start,
never per request. Concurrency **1**.

### S2 — Structure · queue `cpu`

**In:** `parsed.json`  **Out:** `staging/{doc_id}/chunks.json`

1. **Sanitize:** repair broken table pipes, collapse blank runs, strip repeated headers/footers.
   **Fenced code blocks are skipped entirely** — pipe-repair regex corrupts them.
2. **Parent chunks:** 500–1000 tokens, split on markdown headings first, then paragraphs.
3. **Child chunks:** 100–200 tokens within each parent, carrying `parent_index` and `page_number`.
4. **Drop empties:** chunks under 20 tokens, or pure boilerplate — *before* anything bills per token.
5. Assign deterministic `chunk_index` in document order. This is what makes S5 idempotent.

Pure CPU, no network, no external calls. The one stage where a full re-run costs milliseconds.

> **Tokenizer note:** targets are `cl100k_base` tokens via `tiktoken`. CJK text tokenizes very
> differently — if you ingest non-Latin documents, recalibrate rather than assuming these numbers
> transfer.

### S3 — Enrich · queue `llm` · rate-limited

**In:** `chunks.json`  **Out:** `staging/{doc_id}/enriched.json`

Two LLM concerns, one stage, because they share a quota and a failure mode.

**Hard metadata** — filename, page number, created_at. Free, no LLM.

**Soft metadata + context** — batched `gpt-4o-mini` calls, **20 child chunks per request**, using
structured output keyed by chunk id:

```python
{"chunks": [
   {"id": 0, "context": "This section of the 2024 annual report covers…", "category": "FINANCIAL"},
   {"id": 1, "context": "…", "category": "FINANCIAL"}
]}
```

**Validation before anything is written** — this is not optional:

- `len(response.chunks) == len(batch)`
- every input id present in the response, exactly once
- `category` ∈ the closed enum (§13)

A failed batch **retries only the missing ids individually**, never the whole batch of 20. A chunk
that fails solo twice gets `context = ""` and proceeds — a missing context sentence degrades
retrieval slightly; a dead-lettered document helps nobody.

**Output:** `contextualized = f"{context}\n\n{content}"` — this is what gets embedded.

### S4 — Embed · queue `llm` · rate-limited

**In:** `enriched.json`  **Out:** `staging/{doc_id}/embeddings.npy` + manifest

1. Batch **100 texts per request** to `text-embedding-3-small`, `dimensions=1536`.
2. **L2-normalize every vector.** Non-negotiable — `vector_ip_ops` assumes it.
3. Assert returned count == input count; assert every vector is 1536-dim and finite.
4. Write `.npy` (float32) plus a JSON manifest mapping row index → `chunk_index`.

Separate queue from S3's model even though both are "OpenAI" — chat and embedding quotas are
independent, and sharing a bucket makes one starve the other.

### S5 — Persist · queue `cpu`

**In:** `chunks.json` + `enriched.json` + `embeddings.npy`  **Out:** committed rows

One transaction, sync `Session`:

1. Upsert `parent_chunks` on `(document_id, chunk_index)`, capture id map.
2. Upsert `child_chunks` on `(document_id, chunk_index)`, resolving `parent_id` from the map.
3. `documents.status = 'COMPLETED'`, `completed_at = now()`.

Always safe to re-run. Concurrency capped at 4–8 to respect the connection pool.

---

## 7. Checkpoints — MinIO layout

```
raw/
  {sha256}.pdf                       # immutable original, uploaded by the client

staging/
  {document_id}/
    parsed.json                      # S1 → S2
    chunks.json                      # S2 → S3
    enriched.json                    # S3 → S4
    embeddings.npy                   # S4 → S5
    manifest.json                    # row index → chunk_index
```

**Retention:** `staging/` gets a **7-day lifecycle rule**. Long enough to debug a failure, short
enough that it doesn't grow without bound. `raw/` is kept indefinitely — it's the source of truth and
the only way to re-ingest after a chunking-strategy change.

**Resume rule:** each stage checks whether its own output already exists. If yes, skip and advance the
pointer. That makes every stage safe to re-run without a distributed lock.

---

## 8. State machine and idempotency

```
QUEUED ─▶ PARSING ─▶ STRUCTURING ─▶ ENRICHING ─▶ EMBEDDING ─▶ PERSISTING ─▶ COMPLETED
                                                                                │
   any stage ──▶ RETRYING ──(attempts > max)──▶ DEAD_LETTER                     │
                                                (failed_stage, last_error)      ▼
```

`documents.stage` holds the **last completed** stage. `status` holds the current state. The pointer
advances only after the artifact is durably written, so the crash window always resolves to a
harmless re-run of exactly one stage.

| Stage | Skip if | Wasted re-run costs | Retry |
|-------|---------|---------------------|-------|
| S1 | `parsed.json` exists | **High** — Docling inference | 3× backoff |
| S2 | `chunks.json` exists | Trivial — pure CPU | 5× fast |
| S3 | `enriched.json` exists | **High** — dominant invoice line | 3× backoff |
| S4 | `embeddings.npy` exists | Moderate — embedding tokens | 3× backoff |
| S5 | upsert is idempotent | Trivial — one transaction | 5× fast |

**Three idempotency mechanisms, each closing a different hole:**

1. **Upload dedup** — `INSERT … ON CONFLICT (sha256_hash) DO NOTHING RETURNING id`. Publish to the
   broker *only* on a returned row. Makes enqueue exactly-once per file, no lock needed.
2. **Checkpoint skip** — a re-delivered task finds its artifact and advances.
3. **Chunk upsert** — `ON CONFLICT (document_id, chunk_index) DO UPDATE`.

---

## 9. API surface

### Ingestion

```
POST /documents/upload-url
  → { code, message, data: { upload_url, object_key, expires_in } }
  Presigned MinIO PUT. Client uploads directly — bytes never touch the API.

POST /documents
  { object_key, filename, sha256, size_bytes }
  → 202 { data: { document_id, status: "QUEUED" } }
  → 409 { message: "Document already ingested" }    ← the ON CONFLICT path
  Verifies the object exists and its hash matches, then registers + publishes.

GET  /documents/{id}/status
  → { data: { status, stage, attempts, failed_stage, last_error, progress } }

GET  /documents            list + filter by status
DELETE /documents/{id}     cascades to chunks; leaves raw/ intact
```

**Why the client sends the hash:** the API re-verifies it against the stored object before
registering. Trusting a client-supplied hash would let one client poison another's dedup entry.

### Retrieval

```
POST /query
  { question, top_k?: 10, category?: "FINANCIAL" }
  → { data: { answer, citations: [ {document_id, filename, page_number, quote} ], latency_ms } }
```

---

## 10. `app/core/` — the read path in detail

```python
# app/core/retriever.py
async def retrieve(question: str, top_k: int = 10, category: str | None = None) -> list[RetrievedParent]:
    vector = await embed_query(question)          # shared/embeddings.py, L2-normalized
    rows   = await chunk_repository.search(vector, over_fetch=top_k * 5, category=category)
    return rows
```

The query that makes parent/child work — and the one with a silent failure mode:

```sql
WITH candidates AS (
  SELECT c.id, c.parent_id, c.page_number, c.content,
         c.embedding <#> :qvec AS distance
  FROM   child_chunks c
  WHERE  (:category IS NULL OR c.category = :category)
  ORDER  BY c.embedding <#> :qvec
  LIMIT  :over_fetch                       -- top_k × 5
),
best_per_parent AS (
  SELECT DISTINCT ON (parent_id)
         parent_id, distance, page_number, content AS matched_child
  FROM   candidates
  ORDER  BY parent_id, distance            -- parent_id MUST lead DISTINCT ON
)
SELECT p.id, p.content, p.page_start, p.page_end,
       b.distance, b.page_number, b.matched_child,
       d.filename, d.id AS document_id
FROM   best_per_parent b
JOIN   parent_chunks p ON p.id = b.parent_id
JOIN   documents      d ON d.id = p.document_id
ORDER  BY b.distance
LIMIT  :top_k;                             -- top_k DISTINCT parents, guaranteed
```

**Two traps this closes.** `DISTINCT ON` requires `parent_id` to lead the `ORDER BY` — otherwise
Postgres keeps an *arbitrary* child per parent, not the best-matching one, with no error. And
collapsing after `LIMIT` would yield fewer than `top_k` parents; over-fetching then collapsing
guarantees the count.

**`context_builder.py`** assembles parents newest-first under a token budget (default 8000), dropping
the tail rather than truncating mid-parent. **`generator.py`** calls `gpt-4o-mini` with the assembled
context and maps each cited passage back to `(document_id, page_number)` — which works precisely
because `page_number` lives on the *child*, as the original design had it.

---

## 11. Celery configuration

```python
# worker/celery_app.py
app = Celery("rag", broker=settings.rabbitmq_url)
app.conf.update(
    result_backend=None,                    # Postgres is authoritative for status
    task_acks_late=True,                    # crash → redeliver, not lose
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,           # don't hoard tasks on slow stages
    task_routes={
        "worker.stages.parse":     {"queue": "cpu"},
        "worker.stages.structure": {"queue": "cpu"},
        "worker.stages.enrich":    {"queue": "llm"},
        "worker.stages.embed":     {"queue": "llm"},
        "worker.stages.persist":   {"queue": "cpu"},
    },
)
```

**The 3→5 pool split later is this dict plus compose replicas. No task code changes.**

```python
# worker/stages.py
chain(
    parse.s(document_id), structure.s(), enrich.s(), embed.s(), persist.s()
).apply_async()
```

Retry policy per stage:

```python
@app.task(bind=True, autoretry_for=(TransientError,),
          retry_backoff=True, retry_jitter=True, max_retries=3)   # S1, S3, S4
@app.task(bind=True, autoretry_for=(TransientError,),
          retry_backoff=False, max_retries=5)                     # S2, S5
```

`AppException` subclasses that represent **permanent** failures (encrypted PDF, oversized file) are
excluded from `autoretry_for` — they go straight to `DEAD_LETTER` rather than burning five attempts
on a file that will never parse.

---

## 12. Rate limiting

Redis token bucket, one Lua script, atomic in one round trip. **Four buckets**, because requests and
tokens are separate quotas and the one you don't model is the one that 429s you:

```
rl:openai:chat:rpm     rl:openai:chat:tpm
rl:openai:embed:rpm    rl:openai:embed:tpm
```

Backpressure on saturation — **requeue, never sleep**:

```python
allowed, wait_ms = bucket.acquire(tokens=estimated)
if not allowed:
    raise self.retry(
        countdown=(wait_ms / 1000) * random.uniform(1.0, 1.3),   # jitter: avoid stampede
        max_retries=None,                                        # NOT a failure — don't count it
    )
```

Both conditions matter. Without `max_retries=None`, a busy afternoon dead-letters healthy documents.
Without jitter, 20 deferred tasks wake simultaneously and 19 immediately defer again.

A sleeping worker holds its slot and reports as busy — so the pool reads 100% utilized while doing
nothing, which inverts your main diagnostic signal. Requeuing frees the slot and makes
"tasks in retry" a direct, honest measure of rate-limit pressure.

> **Prompt caching — a real deviation from the original design.** Node 5 was specified around
> Anthropic's explicit caching (~90% discount on cache reads). **OpenAI's caching is automatic, ~50%,
> and not directly controllable.** Cached tokens therefore cost roughly 5× what the original design
> assumed. At study volume this is pennies and not worth restructuring for — but the cost model does
> **not** extrapolate. If contextual retrieval moves to production volume, the contextualizer is the
> component to move to Anthropic. `shared/` keeps that a one-interface change.

---

## 13. Configuration

```bash
# existing — the API composes its async DSN from these five parts
DB_HOST=db
DB_PORT=5432
DB_USER=rag
DB_PASSWORD=rag
DB_NAME=rag_beginner
APP_NAME=rag-beginner
ENVIRONMENT=development
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000

# worker — SYNC driver, deliberately different from the API's
WORKER_DATABASE_URL=postgresql+psycopg://rag:rag@db:5432/rag_beginner

RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672//
REDIS_URL=redis://redis:6379/0

MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin           # dev only
MINIO_BUCKET_RAW=raw
MINIO_BUCKET_STAGING=staging

OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
EMBED_DIMENSIONS=1536

DOCLING_URL=http://docling:8100

# limits
MAX_FILE_SIZE_MB=50
MAX_PAGE_COUNT=500
DOCLING_PAGE_TIMEOUT_S=90
ENRICH_BATCH_SIZE=20
EMBED_BATCH_SIZE=100
RL_CHAT_RPM=500
RL_CHAT_TPM=200000
RL_EMBED_RPM=3000
RL_EMBED_TPM=1000000
```

**Closed metadata taxonomy** — off-enum tags are rejected, not stored, and logged as proposals:

```
FINANCIAL · LEGAL · TECHNICAL · MARKETING · HR · RESEARCH · OPERATIONS · OTHER
```

Open-ended tagging accumulates `Finance` / `Financial` / `financial-reports` within a week and
quietly degrades filtering. A closed enum with a review path is the cheap fix.

---

## 14. Guardrails

| Severity | Guard | Where | On trip |
|----------|-------|-------|---------|
| **Critical** | Max size / page count | API + S1 | 413 before storage; `DEAD_LETTER` if found at S1 |
| **Critical** | Encrypted / unopenable PDF | S1 | `DEAD_LETTER` with reason — never a crash |
| **Critical** | Batch id + length validation | S3 | Retry missing ids solo, never the batch |
| **Critical** | Dead-letter after max attempts | all | Alert with `failed_stage`; no silent loop or drop |
| **Critical** | Client hash re-verified server-side | API | 400 on mismatch — prevents dedup poisoning |
| High | Scanned-PDF detection | S1 | Route whole document to OCR |
| High | Per-page vision timeout | S1 | Fall back to raw text, `confidence: 0.0` |
| High | Closed-enum taxonomy | S3 | Reject off-enum; log as proposal |
| High | Empty/boilerplate filter | S2 | Drop before S3/S4 bill for it |
| High | L2-normalization assertion | S4 | Fail loudly — `vector_ip_ops` depends on it |
| Medium | Sanitizer skips fenced code | S2 | Prevents regex corrupting code blocks |
| Medium | Reading-order confidence | S1 | Log low-confidence pages — silent quality loss |
| Medium | Staging lifecycle 7d | MinIO | Bounded growth |

---

## 15. Testing strategy

Carried over from the scaffold: **real Postgres, rolled-back transactions per test, no mocked DB.**

| Layer | Approach | Real dependency |
|-------|----------|-----------------|
| `worker/chunking.py`, `parsing.py` | Pure unit — fixture PDFs | none |
| `worker/enrichment.py` | Batch validation with a stub LLM | none |
| `shared/rate_limiter.py` | Integration — real Redis | redis |
| `worker/persistence.py` | Integration — real Postgres, rolled back | db |
| `app/core/retriever.py` | Integration — seeded vectors, real pgvector | db |
| Stage chain | Integration — `task_always_eager`, real db + MinIO | db, minio |
| API | Existing async client fixture | db |

**Fixture corpus** — the parser edge cases only get caught if they're in the repo: a clean text PDF, a
scanned/image-only PDF, a table-heavy PDF, a multi-column academic PDF, an encrypted PDF, and a
malformed PDF. Six files, and they're the difference between "parses on my machine" and "parses."

**Not mocked:** pgvector similarity, the token bucket, the upsert path. Those are precisely the
components most likely to be silently wrong.

---

## 16. Deliberately deferred

- **Reranking** — a cross-encoder over retrieved children would improve precision. Add after there's
  an eval set to measure it against, not before.
- **HNSW bulk-load path** — matters for corpus backfills, not one-at-a-time ingestion.
- **Auth / multi-tenancy** — no `tenant_id` on any table. Retrofitting means touching every query, so
  decide before real users, not after.
- **Streaming `/query` responses** — SSE is straightforward later; blocking JSON is fine for study.
- **Anthropic contextualizer** — see §12. The interface boundary is already in `shared/`.
- **Evaluation harness** — no retrieval quality measurement. This is the biggest real gap: without it
  you're tuning chunk sizes by intuition.

---

## 17. Build phases

The system splits cleanly at the write/read boundary, and the two phases ship independently.

### Phase 1 — Ingestion (upload + processing)

**This is the scope of the first implementation plan.** See `ingestion-plan.md`.

Everything in §§3–9, 11–14 of this document. Ends with real embeddings in `parent_chunks` and
`child_chunks`, verified by a similarity query.

### Phase 2 — Retrieval (deferred)

Everything in §10: `app/core/` (`retriever.py`, `context_builder.py`, `generator.py`),
`POST /query`, `query_controller.py`, `query_service.py`, and the `DISTINCT ON` retrieval SQL.

**Nothing from Phase 2 is stubbed during Phase 1** — no empty `app/core/` directory, no placeholder
router. It arrives whole, or not at all.

The one exception is a **DB-level similarity test** that stays in Phase 1. Without it, Phase 1 ends
with no evidence the embeddings are usable: wrong normalization, wrong dimension, and a misconfigured
`vector_ip_ops` index all look identical to "rows inserted successfully."
