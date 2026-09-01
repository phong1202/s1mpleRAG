# rag-beginner

A layered FastAPI base. It boots, serves documented CRUD endpoints through a
uniform response envelope, and persists to Postgres with pgvector.

**There is no RAG logic here yet.** The `embedding` column exists and is
migrated, but nothing writes to it and there is no search endpoint — see
"Deliberate gaps" below.

## Layout

| Path | Responsibility |
| --- | --- |
| `main.py` | Starts uvicorn. Nothing else. |
| `app/app.py` | `create_app()` — assembles middleware, handlers, controllers. |
| `app/config.py` | Settings from `.env`. |
| `app/utils/` | Logging and database infrastructure. |
| `app/exceptions/` | Error registry, `AppException`, handlers. |
| `app/middleware/` | CORS and request-id. |
| `app/controllers/` | HTTP layer. |
| `app/services/` | Business rules. |
| `app/repositories/` | Data access. |
| `app/models/` | ORM models. |
| `app/schemas/` | Request/response shapes. |

Requests flow **controller → service → repository → model**. A repository
returns `None` for a missing row; the service turns that into an
`AppException`; the handler renders it as a 404 envelope.

## Running locally (uv)

```bash
uv sync                       # creates .venv from uv.lock

cp .env.example .env          # edit the DB_* variables if needed
docker compose up -d db       # Postgres + pgvector on host port 5433
uv run alembic upgrade head

uv run python main.py         # http://localhost:8000/docs
```

**The host port is 5433, not 5432** — port 5432 is commonly already taken by
another Postgres container.

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

The uv path and the Docker path are two independent ways to run the same
application. Pick one per session. Mixing them — for example running migrations
in the container against a database reached from the host — is the most likely
source of confusion.

## Response envelope

Every response, success or error, has the same shape. `code` always equals the
HTTP status.

```jsonc
// 200
{"code": 200, "message": "Success", "data": {"id": 1, "title": "Intro"}}

// 404
{"code": 404, "message": "Document 9 not found", "data": null}

// 422
{"code": 422, "message": "Validation failed",
 "data": {"errors": [{"field": "title", "message": "Field required"}]}}
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
| POST | `/documents` | 201; 409 if the title is taken. |
| GET | `/documents` | `limit` (1–100, default 20), `offset` (default 0). |
| GET | `/documents/{id}` | 404 if absent. |
| PATCH | `/documents/{id}` | Partial update. 404 if absent; 409 on a title collision. |
| DELETE | `/documents/{id}` | 200, `data: null`. 404 if absent. |

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

## Testing

**Postgres must be running first** — the suite runs against a real database, not
mocks, because the pgvector extension and the async driver are exactly the
pieces worth verifying:

```bash
docker compose up -d db
uv run pytest
```

The suite creates a `rag_beginner_test` database if it is missing and brings it
to head with Alembic. Each test runs inside a transaction that is rolled back,
so tests never share state.

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

## Configuration

| Variable | Required | Default |
| --- | --- | --- |
| `DB_HOST` | **yes** | — |
| `DB_PORT` | **yes** | — |
| `DB_USER` | **yes** | — |
| `DB_PASSWORD` | **yes** | — |
| `DB_NAME` | **yes** | — |
| `DATABASE_URL` | no | — |
| `APP_NAME` | no | `rag-beginner` |
| `ENVIRONMENT` | no | `development` |
| `LOG_LEVEL` | no | `INFO` |
| `CORS_ORIGINS` | no | `http://localhost:3000` |
| `HOST` | no | `0.0.0.0` |
| `PORT` | no | `8000` |

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

- **`embedding` is never populated.** The column is `vector(1536)` and always
  `NULL`. Choosing an embedding provider was deferred rather than baked into
  this scaffold.
- **No search endpoint.** With every vector `NULL` there would be nothing to
  search. It belongs with the embedding work.
- **No index on `embedding`.** An HNSW or IVFFlat index tunes for a query
  pattern that does not exist yet; it belongs in the migration that introduces
  search.
- **No authentication or rate limiting.**
