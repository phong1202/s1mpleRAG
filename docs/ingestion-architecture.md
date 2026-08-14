# Staged Ingestion Pipeline — Architecture Revision

**Agentic RAG Ingestion · Rev 2 — post-review · design only, no implementation code**

The original design was *a synchronous pipeline described asynchronously*: seven nodes in one worker
task, one crash away from re-paying the entire LLM bill. This revision keeps every node, and changes
what holds them together — five checkpointed stages, partitioned by the resource they contend for.

---

## 1. What changes

Four structural changes. Everything else in the review — parser edge cases, batch validation,
taxonomy drift — is a guardrail hung off one of these four.

### Change 01 — Monolithic task becomes a checkpointed saga

Five stages chained through the broker, each writing a durable artifact before the next begins. A
crash resumes from the last completed stage instead of re-running vision inference and
contextualization from scratch.

> **Was:** Nodes 1–7 in one long-lived task.

### Change 02 — Pools partitioned by contended resource

Stages are grouped by what they compete for — VRAM, CPU, provider quota, DB connections — not by what
they mean. Each pool then scales on its own signal, and Docling stops sharing a memory ceiling with
cheap CPU work.

> **Was:** One worker pool, uniform concurrency.

### Change 03 — Rate limiting moves to a shared bucket

A Redis token bucket every worker draws from, sized to the provider quota. Per-document batching
stays, but it is no longer what protects you — batching controls call count, the bucket controls
aggregate rate.

> **Was:** 20-chunk batches, no cross-worker coordination.

### Change 04 — Every write becomes idempotent

Deterministic natural keys on chunks, `ON CONFLICT` upserts, and an atomic insert-or-conflict for
file dedup. Re-delivery becomes boring rather than corrupting.

> **Was:** SELECT-then-INSERT dedup, generated PKs.

---

## 2. Stage topology

The seven original nodes are preserved and regrouped into five stages. The grouping rule is resource
contention: nodes that compete for the same scarce thing belong in the same stage, so that stage's
pool can be sized for it. Between each pair, a checkpoint artifact.

```
┌─────────────────────────────────────────────────────────────────────┐
│ S1  PARSE                                    node 1 — hybrid parser │
│     queue: parse.vision · VRAM-bound · concurrency 1/replica        │
│     per-page timeout 90s                                            │
│                                                                     │
│     PyMuPDF fast pass, then per-page routing to Docling. Runs as    │
│     its own service or its own pool — never in-process alongside    │
│     cheap work. Per-page timeout falls back to raw text extraction  │
│     rather than failing the document.                               │
└─────────────────────────────────────────────────────────────────────┘
         │
         ├── checkpoint: s3://staging/{doc_id}/parsed.md + layout.json
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ S2  STRUCTURE                          nodes 2–3 — sanitize, chunk  │
│     queue: cpu.fast · CPU-bound · concurrency 8–16                  │
│                                                                     │
│     Markdown repair and parent/child hierarchical chunking. Pure    │
│     CPU, no external calls, cheap to retry — the one stage where a  │
│     full re-run costs milliseconds. Empty and boilerplate chunks    │
│     are dropped here, before anything bills per token.              │
└─────────────────────────────────────────────────────────────────────┘
         │
         ├── checkpoint: s3://staging/{doc_id}/chunks.json
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ S3  ENRICH                     nodes 4–5 — metadata, contextualize  │
│     queue: llm.enrich · rate-limited · I/O-bound · concurrency 20–40│
│                                                                     │
│     The expensive stage, and the one most worth checkpointing.      │
│     Batched structured-output calls keyed by chunk id, drawing from │
│     the shared token bucket. Prompt cache prefix is built once per  │
│     document and pinned to a deterministic byte-identical string.   │
└─────────────────────────────────────────────────────────────────────┘
         │
         ├── checkpoint: s3://staging/{doc_id}/enriched.json
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ S4  EMBED                                        node 6 — embedding │
│     queue: llm.embed · rate-limited · I/O-bound · concurrency 20–40 │
│                                                                     │
│     L2-normalized vectors over the *contextualized* child text.     │
│     Separate queue from S3 because embedding and chat quotas are    │
│     separate — sharing a bucket would make one starve the other.    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ├── checkpoint: staging.chunk_embeddings (or .npy blob)
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ S5  PERSIST                                 node 7 — database writer│
│     queue: db.write · connection-bound · concurrency 4–8            │
│                                                                     │
│     One transaction, parents then children, idempotent upserts on   │
│     the natural key. Concurrency is capped to the connection pool — │
│     an unbounded writer pool exhausts Postgres connections long     │
│     before it saturates disk.                                       │
└─────────────────────────────────────────────────────────────────────┘
```

> **Why grouping matters more than splitting.** Splitting into seven stages instead of five would add
> four more checkpoint round-trips to buy nothing: nodes 2 and 3 contend for the same resource and
> fail the same way, so they retry as a unit. Stage boundaries are placed where the *resource profile
> changes*, which is also where the cost of re-running changes.

---

## 3. Crash recovery and idempotency

Each document carries a stage pointer. A retried job reads it, skips completed stages, and resumes.
The pointer advances only after that stage's artifact is durably written — so the failure window is
always "artifact written but pointer not advanced", which resolves to a harmless re-run of one stage.

### State machine

```
QUEUED → PARSING → STRUCTURING → ENRICHING → EMBEDDING → PERSISTING → COMPLETED

any stage ⤳ RETRYING → DEAD_LETTER   (after max_attempts, with failed_stage + last_error)
```

### Resume contract per stage

| Stage | Skip condition on retry | Cost of a wasted re-run | Retry policy |
|-------|------------------------|-------------------------|--------------|
| S1 | `parsed.md` exists in staging | **HIGH** — GPU seconds, per-page vision inference | 3× / backoff |
| S2 | `chunks.json` exists | **TRIVIAL** — pure CPU, milliseconds | 5× / fast |
| S3 | `enriched.json` exists | **HIGH** — the dominant line on the invoice | 3× / backoff |
| S4 | staging rows present for all chunk ids | **MODERATE** — embedding tokens | 3× / backoff |
| S5 | upsert is idempotent — always safe to re-run | **TRIVIAL** — one transaction | 5× / fast |

### Closing the dedup race

The check-then-insert in the original Phase 1 lets two concurrent uploads of the same file both pass.
Push the decision into the constraint, where concurrency is already handled:

```sql
-- requires: UNIQUE (sha256_hash) on documents
INSERT INTO documents (sha256_hash, filename, object_key, status)
VALUES ($1, $2, $3, 'QUEUED')
ON CONFLICT (sha256_hash) DO NOTHING
RETURNING id;

-- zero rows returned  => the file already exists  => 409 Conflict
-- one row returned    => we own it                => enqueue S1, return 202
```

Enqueue **only** on a returned row. That single rule makes the enqueue exactly-once with respect to
file identity, no distributed lock required.

### Idempotent chunk writes

Chunks need identity that survives a retry. `chunk_index` is deterministic given the same parsed
input, which makes `(document_id, chunk_index)` a natural key:

```sql
INSERT INTO child_chunks
  (document_id, parent_id, chunk_index, content, contextualized, page_number, embedding)
SELECT * FROM unnest($1::uuid[], $2::uuid[], $3::int[], $4::text[],
                     $5::text[], $6::int[], $7::vector[])
ON CONFLICT (document_id, chunk_index) DO UPDATE SET
  content        = EXCLUDED.content,
  contextualized = EXCLUDED.contextualized,
  page_number    = EXCLUDED.page_number,
  embedding      = EXCLUDED.embedding;
```

> ### ⚠ Visibility timeout — the failure that looks like success
>
> Redis-backed brokers are at-least-once. If S3 outruns the visibility timeout, a second worker picks
> up the same job while the first is still calling the provider — double billing, with both workers
> believing they are the only one. Staging keeps individual tasks short, which makes this unlikely; a
> heartbeat that extends the lease while the task is alive is what makes it safe. Set the timeout from
> the **p99** of each stage, not the mean.

---

## 4. Worker pools and memory

One pool at uniform concurrency is what turns Docling into an OOM kill. Model weights are resident per
process, so `concurrency=8` means up to eight copies. Partitioning by resource makes the ceiling
explicit and independently scalable.

| Pool | Stages | Scarce resource | Concurrency | Scales on |
|------|--------|-----------------|-------------|-----------|
| `parse.vision` | S1 | VRAM / model weights | 1–2 per replica | queue depth, slowly |
| `cpu.fast` | S2 | CPU cores | 8–16 | CPU utilization |
| `llm.enrich` | S3 | provider chat quota | 20–40 | bucket wait time |
| `llm.embed` | S4 | provider embed quota | 20–40 | bucket wait time |
| `db.write` | S5 | Postgres connections | 4–8 | pool saturation |

Note the concurrency spread: the LLM stages run **twenty times** the concurrency of the vision stage,
because they block on network rather than memory. That ratio is impossible to express in a single
pool, which is the practical argument for splitting them.

### Docling as a service

The stronger version of change 02 is to lift S1 out of the worker entirely — a small inference service
the worker calls over HTTP. This decouples the worker's memory footprint from whether Docling is
loaded at all, lets the GPU tier autoscale on its own utilization, and turns "how many copies of the
weights are resident" from a tuning question into a deployment fact. Load weights once at process
start, never per task.

Size the vision tier for the worst case, not the average. "Only pages with tables" sounds selective
until a 200-page financial report routes every page.

---

## 5. Shared rate limiting

Batching decides how many calls one document makes. It says nothing about how many documents call at
once — which is exactly what a worker pool is for. The limiter has to live outside the worker, in
Redis, atomic:

```lua
-- token bucket, atomic in one round trip. KEYS[1] = bucket key
-- ARGV: rate_per_sec, burst, now_ms, requested_tokens
local rate, burst = tonumber(ARGV[1]), tonumber(ARGV[2])
local now,  want  = tonumber(ARGV[3]), tonumber(ARGV[4])

local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or burst
local ts     = tonumber(b[2]) or now

tokens = math.min(burst, tokens + (now - ts) / 1000 * rate)

if tokens < want then
  -- tell the caller exactly how long to wait; do not spin
  return {0, math.ceil((want - tokens) / rate * 1000)}
end

redis.call('HSET', KEYS[1], 'tokens', tokens - want, 'ts', now)
redis.call('PEXPIRE', KEYS[1], 60000)
return {1, 0}
```

One bucket per provider **and** per limit dimension — requests/min and tokens/min are separate quotas
and need separate buckets, or the one you did not model is the one that 429s you. Returning the wait
time rather than a bare rejection lets the worker sleep precisely instead of retry-storming.

> ### ⚠ Prompt caching is conditional, and fails silently
>
> A cache hit requires a byte-identical prefix within the TTL. Any nondeterminism — dict ordering, a
> timestamp, a re-serialized float — misses the cache and bills full price with no error to catch.
> Build the prefix once per document, hash it, and log the hash alongside reported cache-read tokens
> so a regression shows up as a metric rather than an invoice. For documents that exceed the context
> window, fall back to per-section prefixes; the whole-document assumption breaks first on your
> largest and most expensive files.

---

## 6. Schema and retrieval

Unchanged in shape from the original — parent/child with page number at the child level is right. What
is added is the identity and state needed to make retries safe.

```sql
CREATE TABLE documents (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sha256_hash   char(64) NOT NULL UNIQUE,        -- closes the dedup race
  object_key    text     NOT NULL,
  status        text     NOT NULL,               -- QUEUED … COMPLETED | DEAD_LETTER
  stage         text,                            -- resume pointer: last completed stage
  attempts      int      NOT NULL DEFAULT 0,
  failed_stage  text,
  last_error    text,
  page_count    int,
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE parent_chunks (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index  int  NOT NULL,
  content      text NOT NULL,
  UNIQUE (document_id, chunk_index)              -- natural key for upsert
);

CREATE TABLE child_chunks (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id    uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  parent_id      uuid NOT NULL REFERENCES parent_chunks(id) ON DELETE CASCADE,
  chunk_index    int  NOT NULL,
  content        text NOT NULL,
  contextualized text NOT NULL,
  page_number    int  NOT NULL,                  -- citation precision lives here
  embedding      vector(1536) NOT NULL,
  UNIQUE (document_id, chunk_index)
);

-- inner product, because vectors are L2-normalized at S4
CREATE INDEX ON child_chunks
  USING hnsw (embedding vector_ip_ops)
  WITH (m = 16, ef_construction = 64);
```

### The retrieval query has a silent failure mode

`DISTINCT ON` requires its expression to lead the `ORDER BY`. Get that wrong and Postgres keeps an
*arbitrary* child per parent rather than the best-matching one — no error, just quietly worse
retrieval. And collapsing after `LIMIT` yields fewer parents than requested. Both are fixed by
over-fetching, then collapsing:

```sql
WITH candidates AS (
  SELECT id, parent_id, embedding <#> $1 AS distance
  FROM   child_chunks
  ORDER BY embedding <#> $1
  LIMIT  50                                  -- over-fetch: k × ~5
),
best_per_parent AS (
  SELECT DISTINCT ON (parent_id) parent_id, distance
  FROM   candidates
  ORDER BY parent_id, distance               -- parent_id MUST lead
)
SELECT p.id, p.content
FROM   best_per_parent b
JOIN   parent_chunks p ON p.id = b.parent_id
ORDER BY b.distance
LIMIT  10;                                   -- 10 distinct parents, guaranteed
```

One index note for backfills: HNSW does graph work per insert and degrades under concurrent bulk
writes. For a large initial load, drop the index, load, then `CREATE INDEX CONCURRENTLY` — rather than
paying per-row graph maintenance across thousands of parallel inserts.

---

## 7. Guardrails

Ordered by what they cost you when missing. The first four are admission control — they belong at
upload time, before a bad file can consume a GPU slot.

| Severity | Guard | Where | Behavior when tripped |
|----------|-------|-------|-----------------------|
| **CRITICAL** | Max file size, max page count | API | 413 before the object is stored |
| **CRITICAL** | Encrypted / unopenable PDF | API | 422 with a specific reason, never a pipeline crash |
| **CRITICAL** | Batch response id + length validation | S3 | Retry the missing ids solo, never the whole batch |
| **CRITICAL** | Dead-letter after max attempts | all | Alert with `failed_stage`; never silent retry-loop or silent drop |
| HIGH | Scanned-PDF detection (text length ÷ page count) | S1 | Route the whole document to OCR, not just flagged pages |
| HIGH | Per-page vision timeout | S1 | Fall back to raw text for that page, flag low confidence |
| HIGH | Closed-enum metadata taxonomy | S3 | Reject off-taxonomy tags; queue proposals for review |
| HIGH | Empty / boilerplate chunk filter | S2 | Drop before billing applies at S3 and S4 |
| MEDIUM | Sanitizer skips fenced code blocks | S2 | Prevents pipe-repair regex corrupting legitimate content |
| MEDIUM | Multi-column reading-order check | S1 | Log low-confidence pages — silent quality loss, not an exception |
| MEDIUM | Token targets calibrated per script | S2 | CJK tokenizes very differently at the same nominal size |

One API-layer change worth folding in: move the upload off the request path. Issue a presigned URL,
let the client PUT directly to object storage, and have the API register the key and enqueue. The
original design's step 3 puts a large file transfer inside the request you wanted to keep fast.

---

## 8. Decisions before implementation

Four choices shape how the code gets written. The rest can be settled during implementation.

**Docling** — *Recommendation: separate inference service.* The in-pool alternative works, but keeps
memory footprint coupled to worker concurrency forever, and makes the GPU tier scale on the wrong
signal. Decide now — it changes S1's interface, not just its deployment.

**Checkpoints** — *Recommendation: object storage for S1–S3, staging table for S4.* Markdown and chunk
JSON are blobs; embeddings want to be queryable so S5 can verify completeness before it commits.

**Orchestration** — Celery `chain` / BullMQ flows are sufficient and cheap. A durable-execution engine
(Temporal, Restate) buys automatic resume and visibility, at the cost of new infrastructure.
*Recommendation: broker chains now* — the staging design is what makes resume possible, and it does
not depend on the engine.

**Backpressure** — When the bucket is saturated, S3 tasks can sleep-and-hold a worker slot, or requeue
with delay and release it. *Recommendation: requeue with the bucket's returned wait time* — sleeping
workers look healthy while doing nothing, which is the harder failure to diagnose.

**Still open** — Embedding dimension is written as `1536` above, matching `text-embedding-3-small`.
`gemini-embedding-001` differs — the column type is fixed at migration time, so this needs to be
settled before the first migration rather than after.

---

*Rev 2 — post-review · 7 nodes preserved / 5 stages / 5 pools · no implementation code, design only*
