# s1mpleRAG

Question answering over your own PDFs, with citations down to the page.

Upload a PDF and a pipeline of workers parses it (OCR for scans), splits it
into parent and child chunks, enriches and embeds them, and stores them in
Postgres with pgvector. Ask a question and the API retrieves the closest
chunks, builds a context from their parents, and returns an answer that cites
the document and page each claim was taken from.

The parts: a FastAPI service, three Celery workers, an OCR server on the GPU,
and a Next.js frontend in `web/`.

## Architecture

**Write path — ingestion.** Five stages, each a Celery task that checkpoints
its output to MinIO, so a failed stage is retried without redoing the ones
before it:

| Stage | Does | Queue |
| --- | --- | --- |
| S1 Parse | Text from the PDF; scanned pages go to the OCR server | `ocr` |
| S2 Structure | Parent and child chunks, heading paths, language | `cpu` |
| S3 Enrich | One sentence of context and a category per chunk | `llm` |
| S4 Embed | One vector per child chunk | `llm` |
| S5 Persist | Idempotent upsert into Postgres | `cpu` |

**Read path — retrieval.** One loop in `app/core/pipeline.py`: rewrite, plan,
retrieve, fuse, expand to parents, build the context, gate, generate, reflect.
Retrieval, expansion, context building and generation are real today; the
other five are no-ops that later stages replace, each behind a
`RETRIEVAL_*_ENABLED` flag.

The design, and the reasons behind it, are in
[`docs/system-design.md`](docs/system-design.md) (ingestion) and
[`docs/retrieval-design.md`](docs/retrieval-design.md) (retrieval). Diagrams
are in [`docs/diagrams/`](docs/diagrams/).

Services in `docker-compose.yml`:

| Service | Role |
| --- | --- |
| `db` | Postgres 16 with pgvector: documents, chunks, embeddings, query traces. |
| `rabbitmq` | Celery broker. |
| `redis` | Rate limiter for the LLM provider, and live ingestion progress. |
| `minio` | Object storage: uploaded PDFs and per-stage checkpoints. |
| `minio-init` | Creates the buckets and the checkpoint expiry rule, then exits. |
| `chandra` | The OCR model, served by vLLM. Needs the GPU. |
| `api` | The FastAPI service. |
| `worker-ocr` | Runs S1. |
| `worker-cpu` | Runs S2 and S5. |
| `worker-llm` | Runs S3 and S4. |

## Layout

| Path | Responsibility |
| --- | --- |
| `main.py` | Starts uvicorn. Nothing else. |
| `app/app.py` | `create_app()` — assembles middleware, handlers, controllers. |
| `app/config.py` | Settings from `.env`. |
| `app/controllers/` | HTTP layer. |
| `app/services/` | Business rules. |
| `app/repositories/` | Data access. |
| `app/models/` | ORM models. |
| `app/schemas/` | Request/response shapes. |
| `app/core/` | The read path: contracts, pipeline, retrievers, generator. |
| `app/utils/`, `app/exceptions/`, `app/middleware/` | Logging and database plumbing, the error registry, CORS and request ids. |
| `worker/` | The Celery app and the five ingestion stages. |
| `shared/` | What the API and the workers both use: LLM providers, storage, rate limiter, progress. |
| `eval/` | The labelled question set and the retrieval evaluation runner. |
| `web/` | The Next.js frontend. |
| `alembic/` | Migrations. |
| `tests/` | The test suite. |
| `docs/` | Design documents, plans, diagrams. |

Requests flow **controller → service → repository → model**. A repository
returns `None` for a missing row; the service turns that into an
`AppException`; the handler renders it as a 404 envelope. `app/core/` never
imports from `worker/`, nor the other way round.

## Requirements

- Docker with Compose, and [`uv`](https://docs.astral.sh/uv/).
- `pnpm`, for the frontend only.
- For OCR: an NVIDIA GPU with 16 GB of memory and the NVIDIA container
  runtime. The OCR model fills that GPU; do not share it with anything else
  while documents are being parsed.
- On WSL, raise the VM's memory limit before starting the full stack: the
  default is not enough for the OCR server and the workers together. See §3 of
  [`docs/system-design.md`](docs/system-design.md).

Without a GPU, PDFs that need no OCR can still be ingested and queried; see
"Running with Docker".

## Running locally (uv)

```bash
uv sync                       # creates .venv from uv.lock

cp .env.example .env          # a complete local configuration as it stands
docker compose up -d --wait db rabbitmq redis minio
docker compose up --exit-code-from minio-init minio-init
uv run alembic upgrade head

uv run python main.py         # http://localhost:8000/docs
```

Postgres is published on the host at `DB_PORT`; change it in `.env` if that
port is already taken.

`LLM_PROVIDER=stub` is the default: no network, deterministic output, nothing
to pay. `LLM_PROVIDER=openai` with an `OPENAI_API_KEY` uses the real models,
and every ingested document and every question then costs money.

The frontend:

```bash
cd web
pnpm install
pnpm dev                      # http://localhost:3000
```

`uv run` resolves the project environment itself, so there is no environment to
activate and no way to run against the wrong interpreter by forgetting to. If
you prefer an activated shell, `source .venv/bin/activate` works and the `uv
run` prefix then becomes optional.

Dependencies live in `pyproject.toml`; `uv.lock` pins the exact resolved
versions and is committed, so every machine and the Docker image install the
same set. Add a dependency with `uv add <package>` (`uv add --dev <package>`
for a test-only one) rather than editing `pyproject.toml` by hand — that keeps
the lock file in step.

## Running with Docker

```bash
docker compose up -d --build
docker compose exec api alembic upgrade head
curl http://localhost:8000/health
```

The first start downloads the OCR model and needs the GPU. To run without
one, leave out the OCR server but keep its worker:

```bash
docker compose up -d --build db rabbitmq redis minio minio-init api worker-cpu worker-llm
docker compose up -d --no-deps worker-ocr
```

`worker-ocr` has to run either way: every document starts at S1, on the `ocr`
queue. A PDF with a text layer and no tables or images never calls the OCR
server and is ingested as usual. A page that does need OCR finds the server
down; its document is deferred, and marked failed once the outage has lasted
too long.

The uv path and the Docker path are two independent ways to run the same
application. Pick one per session. Mixing them — for example running migrations
in the container against a database reached from the host — is the most likely
source of confusion.

After changing Celery's `task_routes`, restart `api` as well as the workers:
the publisher is what decides which queue a task goes to.

## Response envelope

Every response, success or error, has the same shape. `code` always equals the
HTTP status.

```jsonc
// 200
{"code": 200, "message": "Success",
 "data": {"document_id": "0b9c2a4e-…", "status": "COMPLETED", "page_count": 14}}

// 404
{"code": 404, "message": "Document 0b9c2a4e-… not found", "data": null}

// 422
{"code": 422, "message": "Validation failed",
 "data": {"errors": [{"field": "question", "message": "Field required"}]}}
```

`DELETE` returns 200 with `data: null` rather than 204, because a 204 has no
body and would break this rule.

Every response also carries an `X-Request-ID` header — echoed back if the
caller sent one, otherwise a generated UUID4. It correlates a response with
the server-side log line for that request, including on a 500: worth quoting
in a bug report.

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | Liveness. |
| POST | `/documents/upload-url` | A presigned PUT for one PDF; 409 if a file with that SHA-256 already exists. |
| POST | `/documents` | 202; registers the uploaded file and sends it down the pipeline. |
| GET | `/documents` | Paginated list, with live progress. |
| GET | `/documents/{id}/status` | One document's status, metadata and progress; 404 if absent. |
| GET | `/documents/{id}/file-url` | A short-lived URL (300 seconds) for the PDF viewer to fetch the file. |
| DELETE | `/documents/{id}` | 200 with `data: null`; 404 if absent. |
| POST | `/query` | An answer with citations, and the `conversation_id` to send with the next question. |

Interactive documentation is at `/docs` on the running API.

## Linting

```bash
uv run ruff check .            # lint
uv run ruff check --fix .      # apply the safe fixes
uv run ruff format .           # format
```

Rules and their exceptions live in `[tool.ruff]` in `pyproject.toml`, each
exception carrying the reason it exists. Two are worth knowing about:

- `Depends()` in a default argument is FastAPI's calling convention, so
  `fastapi.Depends`, `fastapi.Query` and `fastapi.Path` are registered as
  immutable calls; without that, B008 fires on every endpoint.
- `alembic/versions/` and `*.md` are excluded. Migrations come from Alembic's
  own template, and `ruff format` rewrites ```python blocks inside Markdown —
  the design docs use those for illustrative payloads, not runnable code.

A **pre-commit hook** runs both checks and blocks the commit on failure. Git
hooks are not tracked, so a fresh clone has to reinstall it:

```bash
cat > .git/hooks/pre-commit <<'EOF'
#!/bin/sh
if ! uv run ruff check .; then
    echo "ruff check failed - fix with: uv run ruff check --fix ."
    exit 1
fi
if ! uv run ruff format --check .; then
    echo "formatting differs - fix with: uv run ruff format ."
    exit 1
fi
EOF
chmod +x .git/hooks/pre-commit
```

Bypass once with `git commit --no-verify`.

CI runs the same two commands on every pull request into `main`
(`.github/workflows/ci.yml`), so a commit that skipped the hook still gets caught.

## Testing

**The services must be running first** — the suite runs against a real
Postgres, RabbitMQ, Redis and MinIO, not mocks, because pgvector, the async
driver and the storage and broker clients are exactly the pieces worth
verifying:

```bash
docker compose up -d --wait db rabbitmq redis minio
docker compose up --exit-code-from minio-init minio-init
uv run pytest --ignore=tests/test_ocr_service.py
```

The suite creates a `rag_beginner_test` database if it is missing and brings it
to head with Alembic. Each test runs inside a transaction that is rolled back,
so tests never share state.

Every LLM call in the suite goes through a stub, so it needs no API key and
costs nothing. Two things are outside that:

- `tests/test_ocr_service.py` talks to the real OCR server and needs `chandra`
  running on a GPU. CI skips it.
- Tests marked `real_llm` call the real provider. They need
  `LLM_PROVIDER=openai` and a key, and cost money.

## Migrations

Migration files are **always** produced by the Alembic CLI. Never hand-create a
file in `alembic/versions/` — the revision id and history chain are Alembic's to
own.

```bash
uv run alembic revision --autogenerate -m "describe the change"
# review and edit the generated file's body, then:
uv run alembic upgrade head
```

Editing the *body* of a generated revision is normal — that is how the initial
migration gained its `CREATE EXTENSION IF NOT EXISTS vector` statement, which
autogenerate cannot infer.

**Read the generated file before applying it.** Part of the schema belongs to
the database and is deliberately not mapped on the models: the HNSW index on
`child_chunks.embedding`, the generated column `child_chunks.tsv` and its GIN
index, two indexes and a check constraint on `documents`. Autogenerate sees
them as "removed" and proposes to drop every one. Delete those operations from
the revision; no test would notice the HNSW index going, queries would only
get slower.

## Re-ingesting after a change to chunking

A change to chunking, enrichment or embedding does not need OCR again. The parse checkpoint of every
document is kept in object storage for seven days, and this re-runs the stages after it:

```bash
uv run python -m worker.reingest --all                                # every completed document
uv run python -m worker.reingest <document_id> [<document_id> ...]
uv run python -m worker.reingest --all --restore-from data/parsed-backup   # checkpoints expired
```

It calls the LLM provider for every chunk again. A document that is still being processed is skipped
and reported; `--force` relaunches one anyway, for a document stuck in a running status because its
task was lost. The old chunks stay searchable until the new ones replace them.

## Reranking

R3 sends the fused candidates to a reranker and keeps the ten it ranks highest. The default reranker
is the `reranker` container: a small multilingual cross-encoder on the CPU, built into the image, no
GPU and no key needed.

```bash
docker compose build reranker            # model chosen by RERANKER_LOCAL_* in .env
docker compose up -d --no-deps reranker
curl localhost:8200/health
```

Then set `RETRIEVAL_RERANK_ENABLED=true`. To use a hosted model instead, point the API elsewhere; no
code changes:

```bash
RERANKER_URL=https://openrouter.ai/api/v1
RERANKER_MODEL=qwen/qwen3-reranker-4b
RERANKER_API_KEY=sk-or-...
```

The models measured for this, and why the CPU one is the default, are in
`docs/retrieval-design.md` §5.4.

## Evaluation

`eval/dataset.yaml` holds a fixed set of labelled questions, anchored to
`(sha256, page)` so they survive re-chunking.

```bash
uv run python -m eval.run --check-labels   # every quote sits on its gold page; calls no provider
uv run python -m eval.run                  # recall@10 and MRR
uv run python -m eval.run --full           # adds citation accuracy and mean iterations
```

`--embed-cache data/eval-embeddings.json` keeps the question embeddings between runs: only the
first run pays for them.

The last two need `LLM_PROVIDER=openai` and an ingested corpus: the corpus is
embedded by the real model, and a stub query vector measured against it is
noise. The baseline every retrieval change has to beat is recorded in §8.7 of
[`docs/retrieval-design.md`](docs/retrieval-design.md).

## Configuration

Every variable, with the value `.env.example` gives it. Only the five `DB_*`
variables are required; the rest fall back to defaults in `app/config.py`.

| Variable | In `.env.example` | Purpose |
| --- | --- | --- |
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | `localhost`, `5432`, `rag`, `rag`, `rag_beginner` | **Required.** The database, as seen from the host. |
| `DATABASE_URL` | unset | A full DSN; wins over the five above. |
| `APP_NAME`, `ENVIRONMENT`, `LOG_LEVEL` | `rag-beginner`, `development`, `INFO` | |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated. |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | Where the API listens. |
| `RABBITMQ_HOST`, `RABBITMQ_PORT`, `RABBITMQ_USER`, `RABBITMQ_PASSWORD`, `RABBITMQ_VHOST` | `localhost`, `5672`, `guest`, `guest`, `/` | The broker. `RABBITMQ_URL`, when set, wins over all five. |
| `REDIS_URL` | `redis://localhost:6379/0` | |
| `OCR_URL` | `http://localhost:8100` | The OCR server. |
| `OCR_ALL_PAGES` | `false` | `true` sends every page through OCR, not only scans and pages with a table or image. |
| `MINIO_SCHEME`, `MINIO_HOST`, `MINIO_PORT` | `http`, `localhost`, `9000` | The storage endpoint the services use. `MINIO_ENDPOINT`, when set, replaces all three. |
| `MINIO_PUBLIC_ENDPOINT` | `http://localhost:9000` | The host a presigned URL is signed for: what the browser must be able to reach. |
| `MINIO_REGION`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | `us-east-1`, `minioadmin`, `minioadmin` | |
| `MINIO_BUCKET_RAW`, `MINIO_BUCKET_STAGING` | `raw`, `staging` | Uploaded PDFs, and per-stage checkpoints. |
| `LLM_PROVIDER` | `stub` | `stub` or `openai`. Anything else is refused at startup. |
| `OPENAI_API_KEY` | unset | Required when `LLM_PROVIDER=openai`. |
| `OPENAI_CHAT_MODEL`, `OPENAI_REASONING_EFFORT` | `gpt-6-luna`, `low` | Enrichment and answer generation. |
| `OPENAI_EMBED_MODEL`, `EMBED_DIMENSIONS` | `text-embedding-3-large`, `1536` | Changing either means re-ingesting every document. |
| `MAX_FILE_SIZE_MB`, `MAX_PAGE_COUNT`, `MAX_CHUNKS_PER_DOC` | `50`, `500`, `5000` | Upload and document limits. |
| `ENRICH_BATCH_SIZE`, `EMBED_BATCH_SIZE` | `20`, `100` | Chunks per provider call. |
| `RL_CHAT_RPM`, `RL_CHAT_TPM`, `RL_EMBED_RPM`, `RL_EMBED_TPM` | `500`, `200000`, `3000`, `1000000` | The workers' rate limits against the provider. |
| `RETRIEVAL_TOP_K`, `RETRIEVAL_OVER_FETCH`, `RETRIEVAL_EF_SEARCH` | `10`, `50`, `100` | `RETRIEVAL_EF_SEARCH` must be at least twice `RETRIEVAL_OVER_FETCH`. |
| `RETRIEVAL_RRF_K`, `RETRIEVAL_RERANK_KEEP`, `RETRIEVAL_RERANK_CANDIDATES`, `CONTEXT_TOKEN_BUDGET` | `60`, `10`, `50`, `8000` | |
| `RETRIEVAL_MAX_ITERATIONS` | `1` | How many times the read path may loop. |
| `RETRIEVAL_BM25_ENABLED` | `false` | Hybrid search (R2): keyword search beside the vectors, fused by reciprocal rank. |
| `RETRIEVAL_RERANK_ENABLED` | `false` | Reranking (R3): a cross-encoder reorders the fused candidates. Needs a reranker server; see below. |
| `RERANKER_URL`, `RERANKER_MODEL`, `RERANKER_API_KEY`, `RERANKER_TIMEOUT_S` | unset, unset, unset, `5` | Any server speaking Cohere's `/rerank`. Leave the first three unset for the CPU container: the URL becomes `http://reranker:8200` in compose and `http://localhost:8200` on the host, and the container uses the model it was built with. On any failure, or past the timeout, the fused order is kept. |
| `RETRIEVAL_INTENT_ENABLED` | `false` | Follow-up questions (R4): one model call rewrites the question, using the conversation the server keeps, into one that stands alone; small talk is answered without searching. Conversations are kept either way: `POST /query` returns a `conversation_id` to send back with the next question. |
| `RETRIEVAL_ROUTER_ENABLED` | `false` | Routing (R5): a request to summarise a named document is answered from its outline -- chapters and the titles under them -- instead of a search. Needs `RETRIEVAL_INTENT_ENABLED`. |
| `RETRIEVAL_REFLECT_ENABLED`, `WEB_SEARCH_ENABLED` | all `false` | Stage flags for retrieval stages that are not built yet. |

Addresses in `.env` are as seen **from the host**. `docker-compose.yml` reads
the same file and overrides them with service names for the containers.

The five `DB_*` variables have no defaults on purpose: a silently-wrong
database is worse than a refusal to start, so an incomplete set is rejected
by name rather than filled in. `app.config` assembles them into the
SQLAlchemy DSN, percent-encoding user and password so a password containing
`@` or `/` cannot corrupt the URL.

`DATABASE_URL` is an escape hatch for environments that hand out one
ready-made DSN — a managed Postgres, or the test suite pointing Alembic at a
scratch database. When set it wins over the five parts, which may then be
omitted.

`docker-compose.yml` reads the same `.env`: the `db` service takes its
`POSTGRES_*` credentials and its published port from `DB_USER` /
`DB_PASSWORD` / `DB_NAME` / `DB_PORT`, and the `api` service overrides only
`DB_HOST=db` and `DB_PORT=5432`, since container-to-container traffic uses
the service name and the internal port rather than the host mapping.

## Deliberate gaps

- **No authentication.** Anyone who can reach the API can list, delete, query
  and fetch file URLs. Acceptable on localhost only.
- **Redis and RabbitMQ have no volume.** Recreating either container drops
  queued messages, rate-limiter state and live progress. Postgres and the
  checkpoints in MinIO remain the source of truth, but a document whose message
  was lost stays in a running state.
- **Citations point at a page, not a region.** The `locations` format is fixed
  in §5.3 of `docs/system-design.md`, but nothing produces it yet.
- **Retrieval stops at R5.** Hybrid search, reranking, query rewriting and
  routing are built behind flags that ship off; the sufficiency loop,
  reflection and web search are designed, not built.
- **No streaming.** An answer arrives whole.
- **The frontend does not send `conversation_id` back yet** (handed over in
  `docs/fe-query-handoff.md` §12), so every question in the chat starts a new
  conversation until it does.
