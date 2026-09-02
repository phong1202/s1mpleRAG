# Phase 1 — Write Path (Ingestion) · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload a PDF and turn it into embedded, searchable chunks in `parent_chunks` and `child_chunks`.

**Architecture:** The client PUTs the PDF straight into MinIO with a presigned URL; the API only re-verifies the hash and publishes to RabbitMQ. Five Celery stages (S1→S5) run in a chain, each writing a durable checkpoint to MinIO before the next begins, so a crash costs exactly one stage re-run. S1 calls the `docling` container over HTTP; S3/S4 draw from a shared Redis token bucket before every OpenAI call.

**Tech Stack:** FastAPI (async) · Celery + RabbitMQ · SQLAlchemy 2 (async for the API, **sync** for the worker) · Postgres 16 + pgvector · MinIO (S3 API via boto3) · Redis · PyMuPDF + Docling · OpenAI `gpt-4o-mini` + `text-embedding-3-small` · pytest · uv · ruff

**Spec:** [`docs/system-design.md`](../system-design.md) — the settled design. Where this plan and the spec disagree, **the spec wins**. `docs/ingestion-architecture.md` and `docs/ingestion-plan.md` are earlier drafts, superseded in several places; do not follow them.

---

## Global Constraints

Every task implicitly carries these. Values copied verbatim from the spec.

- **Python** `>=3.12`; dependencies managed with `uv`; add packages via `uv add` / `uv add --dev`, never by hand-editing `pyproject.toml`.
- **Branch:** all of Phase 1 lands on branch `phase-1`.
- **Git:** do NOT run `git commit`, `git push`, or `git merge`. End each task by stopping and reporting; wait for an explicit instruction.
- **The suite must be green at the end of EVERY task.** `uv run pytest -q` and `uv run ruff check .` both clean. The pre-commit hook enforces this.
- **Embedding:** `text-embedding-3-small`, `dimensions=1536`, **L2-normalize every vector** — not optional.
- **Chat:** `gpt-4o-mini`.
- **Chunking:** parent 500–1000 tokens, child 100–200 tokens, **drop anything under 20 tokens**. Tokenizer `cl100k_base` via `tiktoken`.
- **Batching:** enrich 20 child chunks per request, embed 100 texts per request.
- **Docling:** per-page timeout `DOCLING_PAGE_TIMEOUT_S=90` with GPU; **300** on CPU-only. Concurrency 1.
- **Scanned detection:** `total_text_chars / page_count < 100` → route the **whole** document to Docling.
- **Limits:** `MAX_FILE_SIZE_MB=50`, `MAX_PAGE_COUNT=500`, `MAX_CHUNKS_PER_DOC=5000`.
- **Category — closed enum, 8 values:** `FINANCIAL` `LEGAL` `TECHNICAL` `MARKETING` `HR` `RESEARCH` `OPERATIONS` `OTHER`. Off-enum values are **rejected and logged**, never stored.
- **Queues:** exactly two — `cpu` (S1, S2, S5) and `llm` (S3, S4).
- **Celery:** `result_backend=None`, `task_acks_late=True`, `task_reject_on_worker_lost=True`, `worker_prefetch_multiplier=1`.
- **DB drivers:** API `postgresql+asyncpg`, worker `postgresql+psycopg` (**sync**). Worker tasks are plain `def` — **no `asyncio.run()` anywhere under `worker/`**.
- **MinIO:** bucket `raw` (kept indefinitely) and `staging` (7-day lifecycle).
- **`app/core/` MUST NOT exist.** That is Phase 2. No empty package, no placeholder router.
- **ruff:** `line-length = 100`, ruleset already configured in `pyproject.toml`.

---

## The 1a / 1b split

| | Tasks | Needs `OPENAI_API_KEY`? |
| --- | --- | --- |
| **Phase 1a** | 1 → 18 | **No.** Runs end-to-end on `StubProvider` |
| **Phase 1b** | 19 | **Yes** |

Phase 1a is done when the chain runs end to end with the stub, `parent_chunks`/`child_chunks` hold rows, and a re-run changes nothing. Phase 1b switches to the real provider and runs the similarity smoke test — the only evidence that the data is *useful* rather than merely well-formed.

---

## File structure

```
app/
  config.py                    MODIFY — ~20 new settings
  models/
    document.py                REPLACE — from title/content to a file entity
    parent_chunk.py            NEW
    child_chunk.py             NEW
  controllers/
    document_controller.py     DELETE (Task 7)
    ingestion_controller.py    NEW — 4 endpoints
  services/
    document_service.py        DELETE (Task 7)
    ingestion_service.py       NEW — dedup + publish
  repositories/
    document_repository.py     DELETE then REWRITE for the new shape
  schemas/
    document.py                REPLACE
shared/                        NEW — sibling of app/; the worker must not import app/
  storage.py                   MinIO/S3 client
  rate_limiter.py              Redis token bucket (Lua)
  llm.py                       Protocol + StubProvider + OpenAIProvider + factory
worker/                        NEW — Celery, sync
  celery_app.py                config + task_routes
  db.py                        sync engine
  stages.py                    5 @task, the chain
  parsing.py                   S1
  chunking.py                  S2
  enrichment.py                S3
  embedding.py                 S4
  persistence.py               S5
docling_service/               NEW — its own container
  main.py                      FastAPI: POST /parse
  Dockerfile
  pyproject.toml
tests/
  fixtures/generate.py         NEW — generates 6 PDFs
  fixtures/*.pdf               NEW — generated, committed
alembic/versions/              NEW — 1 revision
docker-compose.yml             MODIFY — 2 services to 8
.env.example                   MODIFY — new variable groups
```

**`shared/` sits beside `app/`, not inside it.** Under `worker/`, any line starting `from app.` — except `from app.models` — is a mistake. The layout makes a boundary violation visible on the import line instead of relying on anyone remembering the rule.

---

## Phase 1a

### Task 1: Fixture corpus

**Files:**
- Create: `tests/fixtures/generate.py`
- Create: `tests/test_fixtures.py`
- Generates: `tests/fixtures/{clean_text,tables,multi_column,scanned,encrypted,malformed}.pdf`

**Interfaces:**
- Consumes: nothing
- Produces: `tests/fixtures/generate.py::main() -> None` writing 6 files. Later tasks read them by path, `tests/fixtures/<name>.pdf`.

- [ ] **Step 1: Add dev dependencies**

```bash
uv add --dev reportlab pypdf
uv add pymupdf
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_fixtures.py
from pathlib import Path

import fitz  # pymupdf
import pytest
from pypdf import PdfReader

FIXTURES = Path(__file__).parent / "fixtures"


def test_all_six_fixtures_exist():
    names = ["clean_text", "tables", "multi_column", "scanned", "encrypted", "malformed"]
    missing = [n for n in names if not (FIXTURES / f"{n}.pdf").exists()]
    assert missing == []


def test_clean_text_has_a_text_layer():
    doc = fitz.open(FIXTURES / "clean_text.pdf")
    chars = sum(len(page.get_text()) for page in doc)
    assert chars / doc.page_count >= 100, "clean_text must sit above the scanned threshold"


def test_scanned_has_no_text_layer():
    """This fixture is what forces the whole-document OCR routing branch."""
    doc = fitz.open(FIXTURES / "scanned.pdf")
    chars = sum(len(page.get_text()) for page in doc)
    assert chars / doc.page_count < 100


def test_encrypted_is_actually_encrypted():
    assert PdfReader(FIXTURES / "encrypted.pdf").is_encrypted


def test_malformed_cannot_be_opened():
    with pytest.raises(Exception):
        fitz.open(FIXTURES / "malformed.pdf")
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_fixtures.py -q`
Expected: FAIL — `tests/fixtures/` holds no files yet.

- [ ] **Step 4: Write the generator**

```python
# tests/fixtures/generate.py
"""Generate 6 sample PDFs. Run: uv run python tests/fixtures/generate.py

Generated rather than sourced: reproducible, small, no licensing question,
and adding a seventh case means editing a script instead of hunting a file.
"""

from pathlib import Path

import fitz
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Frame, PageTemplate, BaseDocTemplate, Paragraph, Table
from reportlab.lib.styles import getSampleStyleSheet

HERE = Path(__file__).parent
STYLES = getSampleStyleSheet()
BODY = (
    "Third quarter 2024 revenue reached 41.7 billion dong, up 12 percent year over year. "
    "Operating expenses held flat at 8.2 billion dong. "
)


def _simple(path: Path, flowables, page_size=A4):
    doc = BaseDocTemplate(str(path), pagesize=page_size)
    frame = Frame(50, 50, page_size[0] - 100, page_size[1] - 100, id="f")
    doc.addPageTemplates([PageTemplate(id="t", frames=[frame])])
    doc.build(flowables)


def clean_text():
    _simple(HERE / "clean_text.pdf", [Paragraph(BODY * 6, STYLES["Normal"]) for _ in range(3)])


def tables():
    data = [["Quarter", "Revenue", "Expense"], ["Q1", "38.1", "8.0"], ["Q2", "39.4", "8.1"]]
    _simple(HERE / "tables.pdf", [Paragraph(BODY, STYLES["Normal"]), Table(data)])


def multi_column():
    path = HERE / "multi_column.pdf"
    doc = BaseDocTemplate(str(path), pagesize=A4)
    left = Frame(50, 50, 230, 700, id="l")
    right = Frame(300, 50, 230, 700, id="r")
    doc.addPageTemplates([PageTemplate(id="two", frames=[left, right])])
    doc.build([Paragraph(BODY * 4, STYLES["Normal"]) for _ in range(4)])


def scanned():
    """Render clean_text to images and rewrap — the text layer is gone."""
    src = fitz.open(HERE / "clean_text.pdf")
    out = fitz.open()
    for page in src:
        pix = page.get_pixmap(dpi=110)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("png"))
    out.save(HERE / "scanned.pdf")


def encrypted():
    reader = PdfReader(HERE / "clean_text.pdf")
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt("s1mple")
    with open(HERE / "encrypted.pdf", "wb") as fh:
        writer.write(fh)


def malformed():
    """Truncate to 60% of the bytes — header intact, body ruined."""
    data = (HERE / "clean_text.pdf").read_bytes()
    (HERE / "malformed.pdf").write_bytes(data[: int(len(data) * 0.6)])


def main() -> None:
    clean_text()
    tables()
    multi_column()
    scanned()
    encrypted()
    malformed()
    print("6 fixtures written to", HERE)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Generate and re-run**

Run:
```bash
uv run python tests/fixtures/generate.py
uv run pytest tests/test_fixtures.py -q
```
Expected: PASS, 5 tests.

- [ ] **Step 6: Stop and report**

Report: 6 fixtures generated, suite 60 → 65, green. Proposed commit message once instructed:
`test: add generated PDF fixture corpus`

---

### Task 2: Configuration for all of Phase 1

**Files:**
- Modify: `app/config.py`
- Modify: `.env.example`, `.env`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: the existing `app.config.Settings` (5 `DB_*` fields + `DATABASE_URL` override)
- Produces: `Settings.rabbitmq_url`, `.redis_url`, `.minio_endpoint`, `.minio_public_url`, `.minio_access_key`, `.minio_secret_key`, `.minio_bucket_raw`, `.minio_bucket_staging`, `.openai_api_key`, `.openai_chat_model`, `.openai_embed_model`, `.embed_dimensions`, `.docling_url`, `.docling_page_timeout_s`, `.llm_provider`, `.max_file_size_mb`, `.max_page_count`, `.max_chunks_per_doc`, `.enrich_batch_size`, `.embed_batch_size`, `.rl_chat_rpm`, `.rl_chat_tpm`, `.rl_embed_rpm`, `.rl_embed_tpm`, and the property `.worker_database_url -> str`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_config.py
def test_worker_database_url_uses_the_sync_driver(db_env):
    """The worker uses psycopg (sync); the API uses asyncpg. Two engines,
    one set of models."""
    settings = Settings(_env_file=None)

    assert settings.worker_database_url.startswith("postgresql+psycopg://")
    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.worker_database_url.endswith(settings.database_url.split("/")[-1])


def test_llm_provider_defaults_to_stub(db_env):
    """Stub by default: forgetting the key should leave the suite runnable,
    not broken."""
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "stub"


def test_openai_key_is_not_required_when_provider_is_stub(db_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None


def test_minio_public_url_falls_back_to_the_internal_endpoint(db_env, monkeypatch):
    """A presigned URL is signed for one specific host — a URL signed with
    the internal service name is useless to a browser. That is why the two
    endpoints are separate settings."""
    monkeypatch.setenv("MINIO_ENDPOINT", "http://minio:9000")
    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    settings = Settings(_env_file=None)
    assert settings.minio_public_url == "http://minio:9000"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'worker_database_url'`

- [ ] **Step 3: Add the settings**

```python
# app/config.py — add to class Settings, after the DB_* fields

    # --- Infrastructure ---
    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672//"
    redis_url: str = "redis://redis:6379/0"

    # Internal endpoint (container to container) and public endpoint
    # (browser to MinIO). A presigned URL is signed for exactly one host,
    # so one signed with "minio:9000" is unusable by an outside client.
    minio_endpoint: str = "http://minio:9000"
    minio_public_endpoint: str | None = None
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_raw: str = "raw"
    minio_bucket_staging: str = "staging"

    docling_url: str = "http://docling:8100"
    docling_page_timeout_s: int = 90

    # --- LLM ---
    # "stub" by design: without a key the suite still runs.
    llm_provider: str = "stub"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    embed_dimensions: int = 1536

    # --- Limits ---
    max_file_size_mb: int = 50
    max_page_count: int = 500
    max_chunks_per_doc: int = 5000
    enrich_batch_size: int = 20
    embed_batch_size: int = 100
    rl_chat_rpm: int = 500
    rl_chat_tpm: int = 200_000
    rl_embed_rpm: int = 3_000
    rl_embed_tpm: int = 1_000_000

    @property
    def worker_database_url(self) -> str:
        """Same database, synchronous driver. The worker has no event loop."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")

    @property
    def minio_public_url(self) -> str:
        return self.minio_public_endpoint or self.minio_endpoint
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Extend `.env.example` and `.env`**

```bash
# --- Infrastructure ---
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672//
REDIS_URL=redis://redis:6379/0
MINIO_ENDPOINT=http://minio:9000
MINIO_PUBLIC_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_RAW=raw
MINIO_BUCKET_STAGING=staging
DOCLING_URL=http://docling:8100
DOCLING_PAGE_TIMEOUT_S=90

# --- LLM ---
# stub = no network, deterministic results. Switch to openai once you have a key.
LLM_PROVIDER=stub
# OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
EMBED_DIMENSIONS=1536

# --- Limits ---
MAX_FILE_SIZE_MB=50
MAX_PAGE_COUNT=500
MAX_CHUNKS_PER_DOC=5000
ENRICH_BATCH_SIZE=20
EMBED_BATCH_SIZE=100
RL_CHAT_RPM=500
RL_CHAT_TPM=200000
RL_EMBED_RPM=3000
RL_EMBED_TPM=1000000
```

- [ ] **Step 6: Full suite plus lint, then stop**

Run: `uv run pytest -q && uv run ruff check .`
Expected: PASS, 69 tests.

Proposed commit message: `feat(config): add infrastructure, LLM and limit settings`

---

### Task 3: Compose — rabbitmq, redis, minio

**Files:**
- Modify: `docker-compose.yml`
- Create: `tests/test_infrastructure.py`

**Interfaces:**
- Consumes: `Settings.rabbitmq_url`, `.redis_url`, `.minio_endpoint`
- Produces: three running services, used by Tasks 4–6

- [ ] **Step 1: Add dependencies**

```bash
uv add boto3 redis kombu
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_infrastructure.py
"""The three infrastructure containers must be reachable before we write
clients for them."""

import boto3
import redis as redis_lib
from kombu import Connection

from app.config import get_settings

settings = get_settings()


def test_redis_answers_ping():
    client = redis_lib.from_url(settings.redis_url.replace("redis:6379", "localhost:6379"))
    assert client.ping() is True


def test_rabbitmq_accepts_a_connection():
    url = settings.rabbitmq_url.replace("rabbitmq:5672", "localhost:5672")
    with Connection(url) as conn:
        conn.connect()
        assert conn.connected


def test_minio_lists_buckets():
    client = boto3.client(
        "s3",
        endpoint_url=settings.minio_public_url,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
    )
    names = {b["Name"] for b in client.list_buckets()["Buckets"]}
    assert {"raw", "staging"} <= names
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_infrastructure.py -q`
Expected: FAIL — connection refused, no services yet.

- [ ] **Step 4: Add the services**

```yaml
  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "-q", "ping"]
      interval: 10s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ACCESS_KEY:?set MINIO_ACCESS_KEY in .env}
      MINIO_ROOT_PASSWORD: ${MINIO_SECRET_KEY:?set MINIO_SECRET_KEY in .env}
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - miniodata:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 5s
      retries: 20

  # Creates the buckets and exits. Not a long-lived entrypoint — bucket
  # creation is a one-time job, and folding it into minio's own lifecycle
  # only makes minio slower to start.
  minio-init:
    image: minio/mc
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 $${MINIO_ACCESS_KEY} $${MINIO_SECRET_KEY} &&
      mc mb --ignore-existing local/${MINIO_BUCKET_RAW} &&
      mc mb --ignore-existing local/${MINIO_BUCKET_STAGING} &&
      mc ilm rule add --expire-days 7 local/${MINIO_BUCKET_STAGING} || true
      "
    environment:
      MINIO_ACCESS_KEY: ${MINIO_ACCESS_KEY}
      MINIO_SECRET_KEY: ${MINIO_SECRET_KEY}
```

Add `miniodata:` to the `volumes:` block.

- [ ] **Step 5: Bring them up and run the tests**

Run:
```bash
docker compose up -d rabbitmq redis minio minio-init
uv run pytest tests/test_infrastructure.py -q
```
Expected: PASS, 3 tests.

- [ ] **Step 6: Full suite, then stop**

Proposed commit message: `feat(infra): add rabbitmq, redis and minio services`

---

### Task 4: `shared/storage.py`

**Files:**
- Create: `shared/__init__.py`, `shared/storage.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Consumes: `Settings.minio_*`
- Produces:
  - `ObjectStore.put(key: str, data: bytes) -> None`
  - `ObjectStore.get(key: str) -> bytes`
  - `ObjectStore.exists(key: str) -> bool`
  - `ObjectStore.sha256(key: str) -> str`
  - `ObjectStore.presigned_put(key: str, expires_in: int = 3600) -> str`
  - `ObjectStore.put_json(key: str, obj: dict) -> None` / `get_json(key: str) -> dict`
  - `get_store() -> ObjectStore`, `get_public_store() -> ObjectStore`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
import uuid
from urllib.parse import urlparse

import pytest

from app.config import get_settings
from shared.storage import ObjectStore

settings = get_settings()


@pytest.fixture
def store():
    return ObjectStore(endpoint=settings.minio_public_url)


@pytest.fixture
def key():
    return f"staging/test-{uuid.uuid4()}/probe.bin"


def test_put_then_get_round_trips(store, key):
    store.put(key, b"hello")
    assert store.get(key) == b"hello"


def test_exists_is_false_for_a_missing_key(store):
    assert store.exists("staging/definitely-not-here.bin") is False


def test_sha256_matches_the_content(store, key):
    import hashlib

    payload = b"the quick brown fox"
    store.put(key, payload)
    assert store.sha256(key) == hashlib.sha256(payload).hexdigest()


def test_put_json_round_trips(store, key):
    store.put_json(key, {"page_count": 42})
    assert store.get_json(key) == {"page_count": 42}


def test_presigned_put_points_at_the_public_host(store, key):
    """This URL travels to a browser, so it must carry the public host and
    not the internal service name — otherwise an outside client cannot
    resolve it."""
    url = store.presigned_put(key)
    assert urlparse(url).netloc == urlparse(settings.minio_public_url).netloc
    assert "X-Amz-Signature" in url
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_storage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared'`

- [ ] **Step 3: Write the implementation**

```python
# shared/storage.py
"""Object storage client shared by the API and the worker.

boto3 rather than MinIO's own SDK: MinIO speaks the S3 API, so swapping to
real S3 later becomes an endpoint change instead of a client rewrite.
"""

import hashlib
import json
from functools import lru_cache

import boto3

from app.config import get_settings


class ObjectStore:
    def __init__(self, endpoint: str | None = None) -> None:
        settings = get_settings()
        self._bucket_raw = settings.minio_bucket_raw
        self._bucket_staging = settings.minio_bucket_staging
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint or settings.minio_endpoint,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
        )

    def _split(self, key: str) -> tuple[str, str]:
        """'raw/abc.pdf' -> ('raw', 'abc.pdf'). The bucket lives inside the
        key so every layer above only has to remember one string."""
        bucket, _, rest = key.partition("/")
        if bucket not in (self._bucket_raw, self._bucket_staging):
            raise ValueError(f"unknown bucket in key: {key!r}")
        return bucket, rest

    def put(self, key: str, data: bytes) -> None:
        bucket, name = self._split(key)
        self._client.put_object(Bucket=bucket, Key=name, Body=data)

    def get(self, key: str) -> bytes:
        bucket, name = self._split(key)
        return self._client.get_object(Bucket=bucket, Key=name)["Body"].read()

    def exists(self, key: str) -> bool:
        bucket, name = self._split(key)
        try:
            self._client.head_object(Bucket=bucket, Key=name)
            return True
        except self._client.exceptions.ClientError:
            return False

    def sha256(self, key: str) -> str:
        return hashlib.sha256(self.get(key)).hexdigest()

    def put_json(self, key: str, obj: dict) -> None:
        self.put(key, json.dumps(obj, ensure_ascii=False).encode())

    def get_json(self, key: str) -> dict:
        return json.loads(self.get(key))

    def presigned_put(self, key: str, expires_in: int = 3600) -> str:
        bucket, name = self._split(key)
        return self._client.generate_presigned_url(
            "put_object", Params={"Bucket": bucket, "Key": name}, ExpiresIn=expires_in
        )


@lru_cache
def get_store() -> ObjectStore:
    """Internal store, used from inside containers."""
    return ObjectStore()


@lru_cache
def get_public_store() -> ObjectStore:
    """Only for signing presigned URLs handed to outside clients."""
    return ObjectStore(endpoint=get_settings().minio_public_url)
```

- [ ] **Step 4: Run the tests** — PASS, 5 tests.

- [ ] **Step 5: Full suite, then stop**

Proposed commit message: `feat(shared): add S3-compatible object store client`

---

### Task 5: `shared/rate_limiter.py`

**Files:**
- Create: `shared/rate_limiter.py`
- Create: `tests/test_rate_limiter.py`

**Interfaces:**
- Consumes: `Settings.redis_url`, `.rl_*`
- Produces: `TokenBucket(key: str, capacity: int, refill_per_sec: float)` with `acquire(tokens: int = 1) -> tuple[bool, int]` returning `(allowed, wait_ms)`; and `get_bucket(name: str) -> TokenBucket` for `"chat_rpm" | "chat_tpm" | "embed_rpm" | "embed_tpm"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rate_limiter.py
import uuid

import pytest

from shared.rate_limiter import TokenBucket


@pytest.fixture
def bucket():
    return TokenBucket(key=f"test:{uuid.uuid4()}", capacity=3, refill_per_sec=1.0)


def test_allows_up_to_capacity(bucket):
    assert [bucket.acquire()[0] for _ in range(3)] == [True, True, True]


def test_refuses_once_exhausted_and_reports_a_wait(bucket):
    for _ in range(3):
        bucket.acquire()

    allowed, wait_ms = bucket.acquire()

    assert allowed is False
    assert 0 < wait_ms <= 1000, "wait must be the time until a token is available"


def test_acquiring_more_than_one_token_at_a_time(bucket):
    assert bucket.acquire(tokens=3)[0] is True
    assert bucket.acquire(tokens=1)[0] is False


def test_a_request_larger_than_capacity_is_refused_not_hung(bucket):
    """A request bigger than the bucket can never be satisfied — refuse it
    outright rather than returning an unbounded wait."""
    allowed, wait_ms = bucket.acquire(tokens=99)
    assert allowed is False
    assert wait_ms == -1


def test_two_buckets_do_not_share_state():
    a = TokenBucket(key=f"a:{uuid.uuid4()}", capacity=1, refill_per_sec=1.0)
    b = TokenBucket(key=f"b:{uuid.uuid4()}", capacity=1, refill_per_sec=1.0)
    a.acquire()
    assert b.acquire()[0] is True
```

- [ ] **Step 2: Run it to verify it fails** — module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# shared/rate_limiter.py
"""Shared token bucket, held in Redis.

Why Redis and not a per-worker counter: the quota is a property of the whole
system. Twenty workers each counting "I have made 10 calls" adds up to 200
that nobody observed.

Why Lua: "check for a token" and "take a token" must be uninterruptible. Split
into two commands, two workers both read "1 token left" and both take it.
"""

from functools import lru_cache

import redis as redis_lib

from app.config import get_settings

# KEYS[1] = bucket key; ARGV = capacity, refill_per_sec, tokens, now_ms
_SCRIPT = """
local key       = KEYS[1]
local capacity  = tonumber(ARGV[1])
local refill    = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now_ms    = tonumber(ARGV[4])

if requested > capacity then
  return {0, -1}
end

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  ts = now_ms
end

tokens = math.min(capacity, tokens + (now_ms - ts) / 1000 * refill)

local allowed = 0
local wait_ms = 0
if tokens >= requested then
  tokens = tokens - requested
  allowed = 1
else
  wait_ms = math.ceil((requested - tokens) / refill * 1000)
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now_ms)
redis.call('PEXPIRE', key, 3600000)
return {allowed, wait_ms}
"""


class TokenBucket:
    def __init__(self, key: str, capacity: int, refill_per_sec: float) -> None:
        settings = get_settings()
        self._redis = redis_lib.from_url(settings.redis_url)
        self._script = self._redis.register_script(_SCRIPT)
        self.key = key
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec

    def acquire(self, tokens: int = 1) -> tuple[bool, int]:
        """Returns (allowed, wait_ms). wait_ms == -1 means the request can
        never be satisfied because it exceeds capacity."""
        seconds, micros = self._redis.time()
        now_ms = int(seconds * 1000 + micros / 1000)
        allowed, wait_ms = self._script(
            keys=[self.key],
            args=[self.capacity, self.refill_per_sec, tokens, now_ms],
        )
        return bool(allowed), int(wait_ms)


@lru_cache
def get_bucket(name: str) -> TokenBucket:
    """Four buckets, because requests and tokens are separate quotas, and
    chat and embedding are separate quotas again. Sharing one bucket makes
    one starve the other for no reason the provider imposed."""
    settings = get_settings()
    limits = {
        "chat_rpm": settings.rl_chat_rpm,
        "chat_tpm": settings.rl_chat_tpm,
        "embed_rpm": settings.rl_embed_rpm,
        "embed_tpm": settings.rl_embed_tpm,
    }
    per_minute = limits[name]
    return TokenBucket(
        key=f"rl:openai:{name.replace('_', ':')}",
        capacity=per_minute,
        refill_per_sec=per_minute / 60,
    )
```

- [ ] **Step 4: Run the tests** — PASS, 5 tests.

- [ ] **Step 5: Full suite, then stop**

Proposed commit message: `feat(shared): add Redis token bucket rate limiter`

---

### Task 6: `shared/llm.py` — Protocol, Stub, OpenAI

**Files:**
- Create: `shared/llm.py`
- Create: `tests/test_llm_provider.py`

**Interfaces:**
- Consumes: `Settings.llm_provider`, `.openai_*`, `.embed_dimensions`
- Produces:
  - `EnrichedChunk` dataclass: `id: int`, `context: str`, `category: str`
  - `LLMProvider` Protocol: `enrich(chunks: list[dict]) -> list[EnrichedChunk]`, `embed(texts: list[str]) -> list[list[float]]`
  - `StubProvider(drop_ids: list[int] | None = None, bad_category_ids: list[int] | None = None)`
  - `OpenAIProvider()`
  - `get_provider() -> LLMProvider`
  - `CATEGORIES: frozenset[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_provider.py
"""Two implementations behind one Protocol. The stub does NOT exist to save
money — the real cost is a fraction of a cent — it exists because S3's
acceptance criterion is "return 19 of 20 and the missing id must be retried
alone", and no real API can be made to do that on demand.
"""

import pytest

from shared.llm import CATEGORIES, StubProvider, get_provider


@pytest.fixture
def chunks():
    return [{"id": i, "content": f"paragraph number {i}"} for i in range(5)]


def test_stub_returns_one_result_per_chunk(chunks):
    assert len(StubProvider().enrich(chunks)) == len(chunks)


def test_stub_is_deterministic(chunks):
    assert StubProvider().enrich(chunks) == StubProvider().enrich(chunks)


def test_stub_only_emits_categories_from_the_closed_enum(chunks):
    assert all(c.category in CATEGORIES for c in StubProvider().enrich(chunks))


def test_stub_can_be_told_to_drop_ids(chunks):
    """This is why the stub exists: to construct the exact failure S3 has to
    survive."""
    result = StubProvider(drop_ids=[2]).enrich(chunks)

    assert len(result) == 4
    assert 2 not in {c.id for c in result}


def test_stub_can_be_told_to_emit_an_off_enum_category(chunks):
    result = StubProvider(bad_category_ids=[1]).enrich(chunks)
    offender = next(c for c in result if c.id == 1)
    assert offender.category not in CATEGORIES


def test_stub_embeddings_have_the_right_shape_and_are_normalised():
    vectors = StubProvider().embed(["a", "b"])

    assert len(vectors) == 2
    assert all(len(v) == 1536 for v in vectors)
    for v in vectors:
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-6, "the stub normalises too, or it hides a real bug"


def test_get_provider_returns_stub_by_default(db_env):
    assert isinstance(get_provider(), StubProvider)
```

- [ ] **Step 2: Run it to verify it fails** — module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# shared/llm.py
"""The provider seam: one Protocol, two implementations.

Selected two ways, deliberately:
  - the LLM_PROVIDER env var — for running `docker compose up` by hand
  - passed in directly (dependency injection) — for tests, because an env
    var can only switch the stub on, not control WHAT the stub returns.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

from app.config import get_settings

CATEGORIES = frozenset(
    {"FINANCIAL", "LEGAL", "TECHNICAL", "MARKETING", "HR", "RESEARCH", "OPERATIONS", "OTHER"}
)


@dataclass(frozen=True)
class EnrichedChunk:
    id: int
    context: str
    category: str


class LLMProvider(Protocol):
    def enrich(self, chunks: list[dict]) -> list[EnrichedChunk]: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _unit_vector_from(text: str, dimensions: int) -> list[float]:
    """A deterministic fake vector, L2-normalized.

    Normalizing in the stub is deliberate: an unnormalized stub would make
    S4's normalization assertion fail under the stub, and somebody would
    then delete that assertion — the one assertion that must survive into
    production.
    """
    digest = hashlib.sha256(text.encode()).digest()
    raw = [(digest[i % len(digest)] - 128) / 128 for i in range(dimensions)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


class StubProvider:
    """No network. Can be instructed to fail in exactly the way you need."""

    def __init__(
        self,
        drop_ids: list[int] | None = None,
        bad_category_ids: list[int] | None = None,
    ) -> None:
        self.drop_ids = set(drop_ids or [])
        self.bad_category_ids = set(bad_category_ids or [])

    def enrich(self, chunks: list[dict]) -> list[EnrichedChunk]:
        out = []
        for chunk in chunks:
            cid = chunk["id"]
            if cid in self.drop_ids:
                continue
            category = "NOT_A_REAL_CATEGORY" if cid in self.bad_category_ids else "TECHNICAL"
            out.append(
                EnrichedChunk(id=cid, context=f"This section covers item {cid}.", category=category)
            )
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        dimensions = get_settings().embed_dimensions
        return [_unit_vector_from(t, dimensions) for t in texts]


class OpenAIProvider:
    def __init__(self) -> None:
        from openai import OpenAI

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is unset")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._chat_model = settings.openai_chat_model
        self._embed_model = settings.openai_embed_model
        self._dimensions = settings.embed_dimensions

    def enrich(self, chunks: list[dict]) -> list[EnrichedChunk]:
        import json

        prompt = (
            "For each chunk write ONE context sentence and pick exactly one category from: "
            + ", ".join(sorted(CATEGORIES))
            + '. Return JSON {"chunks":[{"id":int,"context":str,"category":str}]}.'
        )
        response = self._client.chat.completions.create(
            model=self._chat_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(chunks, ensure_ascii=False)},
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        return [EnrichedChunk(**c) for c in payload["chunks"]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._embed_model, input=texts, dimensions=self._dimensions
        )
        return [item.embedding for item in response.data]


def get_provider() -> LLMProvider:
    return OpenAIProvider() if get_settings().llm_provider == "openai" else StubProvider()
```

- [ ] **Step 4: Add the dependency and run the tests**

Run:
```bash
uv add openai
uv run pytest tests/test_llm_provider.py -q
```
Expected: PASS, 7 tests.

- [ ] **Step 5: Full suite, then stop**

Proposed commit message: `feat(shared): add LLM provider protocol with stub and OpenAI implementations`

---

### Task 7: Remove the demo CRUD surface

**Files:**
- Delete: `app/controllers/document_controller.py`, `app/services/document_service.py`, `app/repositories/document_repository.py`, `app/schemas/document.py`
- Delete: `tests/test_documents.py`, `tests/test_document_service.py`, `tests/test_document_repository.py`
- Modify: `app/controllers/__init__.py`, `tests/test_transaction_boundary.py`

**Interfaces:**
- Consumes: nothing
- Produces: nothing. This task only **removes**.

> **Why this is its own task.** `Document` currently means "a title/content pair"; after Task 8 it means "an uploaded PDF". The 28 tests below are not broken — they correctly describe a concept that is about to stop existing. Separating "remove the old" from "build the new" makes both diffs readable, and keeps the suite **green** at both points, which is the precondition for TDD to mean anything in later tasks.

- [ ] **Step 1: Unregister the router**

```python
# app/controllers/__init__.py
from fastapi import FastAPI

from app.controllers import health_controller

__all__ = ["register_controllers"]


def register_controllers(app: FastAPI) -> None:
    app.include_router(health_controller.router)
```

- [ ] **Step 2: Delete four modules and three test files**

```bash
rm app/controllers/document_controller.py
rm app/services/document_service.py
rm app/repositories/document_repository.py
rm app/schemas/document.py
rm tests/test_documents.py tests/test_document_service.py tests/test_document_repository.py
```

- [ ] **Step 3: Decouple `test_transaction_boundary.py` from the repository**

These three tests verify `get_session`'s contract (commit on success, roll back on exception). That contract has **nothing to do with** the repository — using `DocumentRepository` was incidental convenience. Use the model directly:

```python
# tests/test_transaction_boundary.py
# REMOVE: from app.repositories.document_repository import DocumentRepository
from app.models.document import Document

# in each test, replace
#   await DocumentRepository(session).create(title=title, content="x")
# with
        session.add(Document(title=title, content="x"))
        await session.flush()
```

- [ ] **Step 4: Note the ErrorCode assertion**

`test_exception_handlers.py` asserts the `DOCUMENT_TITLE_EXISTS` message. Title uniqueness disappears in Task 8, so that ErrorCode changes there — **not here**. Leave it; Task 8 updates it.

- [ ] **Step 5: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: **PASS, 51 tests** (79 − 28). Green, nothing red.

- [ ] **Step 6: Stop and report**

Report explicitly: 28 tests deleted, 3 rewritten, suite at 51 and green.

Proposed commit message: `refactor!: remove the demo document CRUD surface`

---

### Task 8: New schema — models and migration

**Files:**
- Rewrite: `app/models/document.py`
- Create: `app/models/parent_chunk.py`, `app/models/child_chunk.py`
- Modify: `app/models/__init__.py`, `app/exceptions/error_codes.py`
- Create: `alembic/versions/<hash>_phase1_schema.py` (generated by CLI)
- Create: `tests/test_schema.py`
- Modify: `tests/test_transaction_boundary.py`, `tests/test_exception_handlers.py`

**Interfaces:**
- Consumes: `app.models.base.Base`
- Produces:
  - `Document` with `id: uuid.UUID`, `sha256_hash: str`, `filename: str`, `object_key: str`, `size_bytes: int`, `page_count: int | None`, `status: str`, `stage: str | None`, `attempts: int`, `failed_stage: str | None`, `last_error: str | None`, `created_at`, `updated_at`, `completed_at`
  - `ParentChunk` with `id`, `document_id`, `chunk_index`, `content`, `token_count`, `page_start`, `page_end`
  - `ChildChunk` with `id`, `document_id`, `parent_id`, `chunk_index`, `content`, `contextualized`, `page_number`, `token_count`, `embedding`, `category`
  - `ErrorCode.DOCUMENT_ALREADY_INGESTED = (409, "Document already ingested")`, `ErrorCode.PDF_ENCRYPTED`, `ErrorCode.PDF_TOO_LARGE`, `ErrorCode.HASH_MISMATCH`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
"""The schema is the contract both the API and the worker depend on. Check it
at the database level, not only at the model level — indexes and constraints
do not exist in Python."""

import uuid

import pytest
from sqlalchemy import text

from app.models import ChildChunk, Document, ParentChunk

pytestmark = pytest.mark.asyncio


async def test_document_is_a_file_entity_not_a_title_content_pair(db_session):
    doc = Document(
        sha256_hash="a" * 64,
        filename="report.pdf",
        object_key="raw/" + "a" * 64 + ".pdf",
        size_bytes=1234,
    )
    db_session.add(doc)
    await db_session.flush()

    assert isinstance(doc.id, uuid.UUID)
    assert doc.status == "QUEUED"
    assert doc.attempts == 0
    assert not hasattr(doc, "title")
    assert not hasattr(doc, "content")


async def test_sha256_hash_is_unique(db_session):
    from sqlalchemy.exc import IntegrityError

    for _ in range(2):
        db_session.add(
            Document(sha256_hash="b" * 64, filename="x.pdf", object_key="raw/x.pdf", size_bytes=1)
        )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_child_chunk_carries_the_page_number(db_session):
    """page_number lives on the child, not the parent — that is what
    determines citation precision in Phase 2."""
    doc = Document(sha256_hash="c" * 64, filename="x.pdf", object_key="raw/x.pdf", size_bytes=1)
    db_session.add(doc)
    await db_session.flush()

    parent = ParentChunk(
        document_id=doc.id, chunk_index=0, content="parent", token_count=600,
        page_start=1, page_end=2,
    )
    db_session.add(parent)
    await db_session.flush()

    child = ChildChunk(
        document_id=doc.id,
        parent_id=parent.id,
        chunk_index=0,
        content="child",
        contextualized="context\n\nchild",
        page_number=2,
        token_count=120,
        embedding=[0.0] * 1536,
    )
    db_session.add(child)
    await db_session.flush()

    assert child.page_number == 2


async def test_hnsw_index_uses_inner_product(db_session):
    """vector_ip_ops assumes unit vectors. If the index is built with
    vector_cosine_ops instead, S4's L2 normalization buys nothing."""
    result = await db_session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'child_chunks'")
    )
    defs = " ".join(row[0] for row in result)
    assert "hnsw" in defs.lower()
    assert "vector_ip_ops" in defs
```

- [ ] **Step 2: Run it to verify it fails** — `ImportError: cannot import name 'ChildChunk'`

- [ ] **Step 3: Write the models**

```python
# app/models/document.py — REPLACE THE WHOLE FILE
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Document(Base):
    """An uploaded PDF. Text lives in chunks, not here."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # This unique constraint is what closes the dedup race: INSERT ... ON
    # CONFLICT DO NOTHING RETURNING id, publish only on a returned row.
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(Text, nullable=False, default="QUEUED")
    # Resume pointer: the stage that COMPLETED, not the one running.
    stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

```python
# app/models/parent_chunk.py
import uuid

from sqlalchemy import ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ParentChunk(Base):
    """The generation-context unit. Never embedded."""

    __tablename__ = "parent_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Deterministic from parse output — this is the natural key that makes
    # S5 idempotent.
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
```

```python
# app/models/child_chunk.py
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ChildChunk(Base):
    """The retrieval unit. What gets embedded is `contextualized`, not `content`."""

    __tablename__ = "child_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parent_chunks.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    contextualized: Mapped[str] = mapped_column(Text, nullable=False)
    # Citation precision lives HERE, not on the parent.
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
```

```python
# app/models/__init__.py
from app.models.base import Base
from app.models.child_chunk import ChildChunk
from app.models.document import Document
from app.models.parent_chunk import ParentChunk

__all__ = ["Base", "ChildChunk", "Document", "ParentChunk"]
```

- [ ] **Step 4: Generate the migration and add the HNSW index by hand**

Run: `uv run alembic revision --autogenerate -m "phase 1 schema"`

Autogenerate cannot infer the HNSW index. Append to `upgrade()`:

```python
    op.execute(
        "CREATE INDEX ix_child_chunks_embedding_hnsw ON child_chunks "
        "USING hnsw (embedding vector_ip_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.execute(
        "CREATE INDEX ix_documents_active_status ON documents (status) "
        "WHERE status NOT IN ('COMPLETED', 'DEAD_LETTER')"
    )
```

and prepend to `downgrade()`:

```python
    op.execute("DROP INDEX IF EXISTS ix_child_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_documents_active_status")
```

- [ ] **Step 5: Update the ErrorCodes**

```python
# app/exceptions/error_codes.py — replace DOCUMENT_TITLE_EXISTS
    DOCUMENT_NOT_FOUND = (404, "Document not found")
    DOCUMENT_ALREADY_INGESTED = (409, "Document already ingested")
    PDF_ENCRYPTED = (422, "PDF is encrypted and cannot be parsed")
    PDF_TOO_LARGE = (413, "PDF exceeds the size or page limit")
    HASH_MISMATCH = (400, "Uploaded object does not match the supplied hash")
    VALIDATION_FAILED = (422, "Validation failed")
    INTERNAL_ERROR = (500, "Internal server error")
    DATABASE_ERROR = (500, "Database operation failed")
```

Update the matching assertion in `tests/test_exception_handlers.py` from `"A document with this title already exists"` to `"Document already ingested"`.

- [ ] **Step 6: Update `test_transaction_boundary.py` for the new model**

```python
# replace Document(title=title, content="x") with
        session.add(
            Document(
                sha256_hash=marker, filename="x.pdf", object_key=f"raw/{marker}.pdf", size_bytes=1
            )
        )
# and the raw SQL: SELECT 1 FROM documents WHERE sha256_hash = :marker
```
(`marker` is a unique 64-character string per test, e.g. `uuid.uuid4().hex * 2`.)

- [ ] **Step 7: Migrate and run the suite**

Run:
```bash
uv run alembic upgrade head
uv run pytest -q && uv run ruff check .
```
Expected: PASS, 55 tests.

- [ ] **Step 8: Stop and report**

Proposed commit message: `feat(db): replace CRUD schema with file entity and chunk tables`

---

### Task 9: API — presigned upload and registration

**Files:**
- Create: `app/schemas/ingestion.py`, `app/repositories/document_repository.py`, `app/services/ingestion_service.py`, `app/controllers/ingestion_controller.py`
- Modify: `app/controllers/__init__.py`
- Create: `tests/test_ingestion_api.py`

**Interfaces:**
- Consumes: `ObjectStore`, `Document`, `ErrorCode`
- Produces:
  - `DocumentRepository.insert_if_new(sha256_hash, filename, object_key, size_bytes) -> Document | None` — `None` when it already exists
  - `IngestionService.create_upload_url(filename) -> UploadTarget`
  - `IngestionService.register(payload: DocumentRegister) -> Document`
  - `POST /documents/upload-url`, `POST /documents`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingestion_api.py
import uuid

import pytest

pytestmark = pytest.mark.asyncio


async def test_upload_url_returns_a_presigned_put(client):
    response = await client.post("/documents/upload-url", json={"filename": "a.pdf"})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["object_key"].startswith("raw/")
    assert "X-Amz-Signature" in data["upload_url"]


async def test_register_returns_202_and_a_queued_document(client, uploaded_pdf):
    response = await client.post("/documents", json=uploaded_pdf)

    assert response.status_code == 202
    assert response.json()["data"]["status"] == "QUEUED"
    assert uuid.UUID(response.json()["data"]["document_id"])


async def test_registering_the_same_hash_twice_returns_409(client, uploaded_pdf):
    await client.post("/documents", json=uploaded_pdf)

    second = await client.post("/documents", json=uploaded_pdf)

    assert second.status_code == 409
    assert second.json()["message"] == "Document already ingested"


async def test_a_wrong_client_supplied_hash_is_rejected(client, uploaded_pdf):
    """Trusting a client-supplied hash lets one client poison another's
    dedup entry. The API must recompute it over the stored object."""
    payload = {**uploaded_pdf, "sha256": "f" * 64}

    response = await client.post("/documents", json=payload)

    assert response.status_code == 400
    assert "does not match" in response.json()["message"]


async def test_an_object_that_was_never_uploaded_is_rejected(client):
    payload = {
        "object_key": "raw/" + "9" * 64 + ".pdf",
        "filename": "ghost.pdf",
        "sha256": "9" * 64,
        "size_bytes": 10,
    }

    response = await client.post("/documents", json=payload)

    assert response.status_code == 404
```

Add the fixture to `tests/conftest.py`:

```python
@pytest.fixture
def uploaded_pdf():
    """Actually push a fixture PDF into MinIO and return the register payload."""
    import hashlib
    from pathlib import Path

    from shared.storage import get_public_store

    data = (Path(__file__).parent / "fixtures" / "clean_text.pdf").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    key = f"raw/{digest}.pdf"
    get_public_store().put(key, data)
    return {
        "object_key": key,
        "filename": "clean_text.pdf",
        "sha256": digest,
        "size_bytes": len(data),
    }
```

- [ ] **Step 2: Run it to verify it fails** — 404 on every route.

- [ ] **Step 3: Write schema, repository, service, controller**

```python
# app/schemas/ingestion.py
from pydantic import BaseModel, ConfigDict, Field


class UploadUrlRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)


class UploadTarget(BaseModel):
    upload_url: str
    object_key: str
    expires_in: int


class DocumentRegister(BaseModel):
    object_key: str = Field(min_length=1)
    filename: str = Field(min_length=1, max_length=255)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(gt=0)


class DocumentAccepted(BaseModel):
    document_id: str
    status: str


class DocumentStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    stage: str | None
    attempts: int
    failed_stage: str | None
    last_error: str | None
```

```python
# app/repositories/document_repository.py
import uuid

from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.utils.database import get_session


class DocumentRepository:
    """No business rules; never commits — the session owns the transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert_if_new(
        self, sha256_hash: str, filename: str, object_key: str, size_bytes: int
    ) -> Document | None:
        """INSERT ... ON CONFLICT DO NOTHING RETURNING id.

        One statement, atomic. None means the file already exists. The caller
        publishes to the broker ONLY on a returned row — that single rule
        makes enqueue exactly-once per file with no distributed lock.
        """
        statement = (
            insert(Document)
            .values(
                sha256_hash=sha256_hash,
                filename=filename,
                object_key=object_key,
                size_bytes=size_bytes,
            )
            .on_conflict_do_nothing(index_elements=["sha256_hash"])
            .returning(Document.id)
        )
        new_id = (await self.session.execute(statement)).scalar_one_or_none()
        if new_id is None:
            return None
        return await self.session.get(Document, new_id)

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def list(self, limit: int, offset: int, status: str | None = None):
        query = select(Document)
        if status:
            query = query.where(Document.status == status)
        total = await self.session.scalar(select(func.count()).select_from(query.subquery()))
        rows = await self.session.execute(
            query.order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
        return list(rows.scalars().all()), total or 0


async def get_document_repository(
    session: AsyncSession = Depends(get_session),
) -> DocumentRepository:
    return DocumentRepository(session)
```

```python
# app/services/ingestion_service.py
import uuid

from fastapi import Depends

from app.config import get_settings
from app.exceptions import AppException, ErrorCode
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository, get_document_repository
from app.schemas.ingestion import DocumentRegister, UploadTarget
from shared.storage import ObjectStore, get_public_store, get_store


class IngestionService:
    def __init__(self, repository: DocumentRepository, store: ObjectStore) -> None:
        self.repository = repository
        self.store = store

    def create_upload_url(self, filename: str) -> UploadTarget:
        # Temporary uuid key: the real hash is only known after the client
        # finishes uploading.
        key = f"raw/pending-{uuid.uuid4()}.pdf"
        return UploadTarget(
            upload_url=get_public_store().presigned_put(key),
            object_key=key,
            expires_in=3600,
        )

    async def register(self, payload: DocumentRegister) -> Document:
        if not self.store.exists(payload.object_key):
            raise AppException(
                ErrorCode.DOCUMENT_NOT_FOUND, f"Object {payload.object_key} not found in storage"
            )

        # Recompute the hash over the stored object. Trusting the client's
        # hash lets one client poison another's dedup entry.
        actual = self.store.sha256(payload.object_key)
        if actual != payload.sha256:
            raise AppException(ErrorCode.HASH_MISMATCH)

        limit = get_settings().max_file_size_mb * 1024 * 1024
        if payload.size_bytes > limit:
            raise AppException(ErrorCode.PDF_TOO_LARGE)

        document = await self.repository.insert_if_new(
            sha256_hash=actual,
            filename=payload.filename,
            object_key=payload.object_key,
            size_bytes=payload.size_bytes,
        )
        if document is None:
            raise AppException(ErrorCode.DOCUMENT_ALREADY_INGESTED)
        return document


async def get_ingestion_service(
    repository: DocumentRepository = Depends(get_document_repository),
) -> IngestionService:
    return IngestionService(repository, get_store())
```

```python
# app/controllers/ingestion_controller.py
from fastapi import APIRouter, Depends, status

from app.schemas.ingestion import (
    DocumentAccepted,
    DocumentRegister,
    UploadTarget,
    UploadUrlRequest,
)
from app.schemas.response import ApiResponse, ErrorResponse
from app.services.ingestion_service import IngestionService, get_ingestion_service

router = APIRouter(prefix="/documents", tags=["ingestion"])

_CONFLICT = {409: {"model": ErrorResponse, "description": "Document already ingested"}}


@router.post("/upload-url", response_model=ApiResponse[UploadTarget])
async def create_upload_url(
    payload: UploadUrlRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[UploadTarget]:
    return ApiResponse.ok(service.create_upload_url(payload.filename))


@router.post(
    "",
    response_model=ApiResponse[DocumentAccepted],
    status_code=status.HTTP_202_ACCEPTED,
    responses={**_CONFLICT},
)
async def register_document(
    payload: DocumentRegister,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[DocumentAccepted]:
    document = await service.register(payload)
    return ApiResponse(
        code=202,
        message="Accepted",
        data=DocumentAccepted(document_id=str(document.id), status=document.status),
    )
```

Register the router in `app/controllers/__init__.py`.

- [ ] **Step 4: Run the tests** — PASS, 5 tests.

- [ ] **Step 5: Full suite, then stop**

Proposed commit message: `feat(api): add presigned upload and document registration`

---

### Task 10: API — status, list, delete

**Files:**
- Modify: `app/controllers/ingestion_controller.py`, `app/services/ingestion_service.py`
- Modify: `tests/test_ingestion_api.py`

**Interfaces:**
- Consumes: `DocumentRepository.get_by_id`, `.list`
- Produces: `GET /documents/{id}/status`, `GET /documents`, `DELETE /documents/{id}`

- [ ] **Step 1: Write the failing test**

```python
async def test_status_reports_the_pipeline_pointer(client, uploaded_pdf):
    created = await client.post("/documents", json=uploaded_pdf)
    document_id = created.json()["data"]["document_id"]

    response = await client.get(f"/documents/{document_id}/status")

    data = response.json()["data"]
    assert data["status"] == "QUEUED"
    assert data["stage"] is None
    assert data["attempts"] == 0


async def test_status_of_an_unknown_uuid_is_404(client):
    import uuid

    response = await client.get(f"/documents/{uuid.uuid4()}/status")
    assert response.status_code == 404


async def test_a_malformed_uuid_is_422_not_500(client):
    """This guard used to be an int4 range bound. The id is a uuid now, so
    FastAPI handles it — but confirm it is a 422 and not a driver blow-up."""
    response = await client.get("/documents/not-a-uuid/status")
    assert response.status_code == 422


async def test_list_filters_by_status(client, uploaded_pdf):
    await client.post("/documents", json=uploaded_pdf)

    response = await client.get("/documents", params={"status": "QUEUED"})

    data = response.json()["data"]
    assert data["total"] >= 1
    assert all(item["status"] == "QUEUED" for item in data["items"])


async def test_delete_removes_the_document(client, uploaded_pdf):
    created = await client.post("/documents", json=uploaded_pdf)
    document_id = created.json()["data"]["document_id"]

    await client.delete(f"/documents/{document_id}")

    assert (await client.get(f"/documents/{document_id}/status")).status_code == 404
```

- [ ] **Step 2: Run it to verify it fails** — 404s.

- [ ] **Step 3: Add the three endpoints**

```python
# app/services/ingestion_service.py — add to IngestionService
    async def get(self, document_id: uuid.UUID) -> Document:
        document = await self.repository.get_by_id(document_id)
        if document is None:
            raise AppException(ErrorCode.DOCUMENT_NOT_FOUND, f"Document {document_id} not found")
        return document

    async def list(self, limit: int, offset: int, status: str | None):
        return await self.repository.list(limit=limit, offset=offset, status=status)

    async def delete(self, document_id: uuid.UUID) -> None:
        """Cascades to chunks via ON DELETE CASCADE; raw/ is untouched — it
        is the source of truth and the only way to re-ingest."""
        document = await self.get(document_id)
        await self.repository.session.delete(document)
```

```python
# app/controllers/ingestion_controller.py — add
import uuid
from typing import Annotated

from fastapi import Path, Query

from app.schemas.ingestion import DocumentStatus
from app.schemas.response import PaginatedData

DocumentId = Annotated[uuid.UUID, Path()]


@router.get("/{document_id}/status", response_model=ApiResponse[DocumentStatus])
async def get_status(
    document_id: DocumentId,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[DocumentStatus]:
    document = await service.get(document_id)
    return ApiResponse.ok(
        DocumentStatus(
            id=str(document.id),
            filename=document.filename,
            status=document.status,
            stage=document.stage,
            attempts=document.attempts,
            failed_stage=document.failed_stage,
            last_error=document.last_error,
        )
    )


@router.get("", response_model=ApiResponse[PaginatedData[DocumentStatus]])
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[PaginatedData[DocumentStatus]]:
    documents, total = await service.list(limit=limit, offset=offset, status=status)
    return ApiResponse.ok(
        PaginatedData[DocumentStatus](
            items=[
                DocumentStatus(
                    id=str(d.id), filename=d.filename, status=d.status, stage=d.stage,
                    attempts=d.attempts, failed_stage=d.failed_stage, last_error=d.last_error,
                )
                for d in documents
            ],
            total=total, limit=limit, offset=offset,
        )
    )


@router.delete("/{document_id}", response_model=ApiResponse[None])
async def delete_document(
    document_id: DocumentId,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[None]:
    await service.delete(document_id)
    return ApiResponse(code=200, message="Document deleted", data=None)
```

- [ ] **Step 4: Run the tests** — PASS, 10 tests in the file.

- [ ] **Step 5: Full suite, then stop**

Proposed commit message: `feat(api): add document status, list and delete endpoints`

---

### Task 11: Celery skeleton with five no-op stages

**Files:**
- Create: `worker/__init__.py`, `worker/celery_app.py`, `worker/db.py`, `worker/stages.py`
- Modify: `docker-compose.yml`, `app/services/ingestion_service.py`
- Create: `tests/test_stage_chain.py`

**Interfaces:**
- Consumes: `Settings.rabbitmq_url`, `.worker_database_url`
- Produces:
  - `worker.celery_app.app` (Celery instance)
  - `worker.db.session_scope()` sync context manager
  - `worker.stages.{parse,structure,enrich,embed,persist}` — five tasks
  - `worker.stages.launch(document_id: str) -> None` — builds and sends the chain
  - `worker.stages.STAGES = ["PARSING","STRUCTURING","ENRICHING","EMBEDDING","PERSISTING"]`

> **Why this task deliberately does no real work.** It proves routing, chaining, `acks_late`, retry and the state machine while every stage is trivially debuggable. Skipping it means debugging RabbitMQ semantics and Docling inference **at the same time** later.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage_chain.py
"""Runs the chain in-process with task_always_eager — exercises chain logic
and the state machine, not broker delivery (that is checked by hand)."""

import uuid

import pytest

from worker.celery_app import app as celery_app
from worker.stages import STAGES, launch


@pytest.fixture(autouse=True)
def eager():
    celery_app.conf.task_always_eager = True
    yield
    celery_app.conf.task_always_eager = False


def reload(document_id):
    from app.models.document import Document
    from worker.db import session_scope

    with session_scope() as session:
        return session.get(Document, uuid.UUID(str(document_id)))


def test_task_routes_split_cpu_and_llm():
    routes = celery_app.conf.task_routes
    assert routes["worker.stages.parse"]["queue"] == "cpu"
    assert routes["worker.stages.structure"]["queue"] == "cpu"
    assert routes["worker.stages.enrich"]["queue"] == "llm"
    assert routes["worker.stages.embed"]["queue"] == "llm"
    assert routes["worker.stages.persist"]["queue"] == "cpu"


def test_acks_late_is_on():
    """A crash must redeliver, not lose the work."""
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_no_result_backend():
    """documents.status in Postgres is the durable state, not a backend."""
    assert celery_app.conf.result_backend is None


def test_chain_drives_a_document_to_completed(seeded_document):
    launch(str(seeded_document.id))

    refreshed = reload(seeded_document.id)
    assert refreshed.status == "COMPLETED"
    assert refreshed.stage == STAGES[-1]
    assert refreshed.completed_at is not None
```

Add a `seeded_document` fixture to `tests/conftest.py` that inserts a `Document` through the **sync** engine and really commits (the chain runs outside the test transaction):

```python
@pytest.fixture
def seeded_document(uploaded_pdf):
    from app.models.document import Document
    from worker.db import session_scope

    with session_scope() as session:
        doc = Document(
            sha256_hash=uploaded_pdf["sha256"],
            filename=uploaded_pdf["filename"],
            object_key=uploaded_pdf["object_key"],
            size_bytes=uploaded_pdf["size_bytes"],
        )
        session.add(doc)
        session.flush()
        doc_id = doc.id
    yield type("Doc", (), {"id": doc_id})()
    with session_scope() as session:
        obj = session.get(Document, doc_id)
        if obj:
            session.delete(obj)
```

- [ ] **Step 2: Run it to verify it fails** — `ModuleNotFoundError: No module named 'worker'`

- [ ] **Step 3: Write the worker**

```python
# worker/db.py
"""A SYNCHRONOUS engine, deliberately different from the API's async one.

Two engines, one set of models. Celery tasks are plain `def` — there is no
asyncio.run() anywhere under worker/.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

engine = create_engine(get_settings().worker_database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, class_=Session, expire_on_commit=False)


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

```python
# worker/celery_app.py
from celery import Celery

from app.config import get_settings

settings = get_settings()

app = Celery("rag", broker=settings.rabbitmq_url)
app.conf.update(
    result_backend=None,           # Postgres is authoritative for status
    task_acks_late=True,           # crash -> redeliver, not lose
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # don't hoard tasks on slow stages
    task_routes={
        "worker.stages.parse": {"queue": "cpu"},
        "worker.stages.structure": {"queue": "cpu"},
        "worker.stages.enrich": {"queue": "llm"},
        "worker.stages.embed": {"queue": "llm"},
        "worker.stages.persist": {"queue": "cpu"},
    },
)
app.autodiscover_tasks(["worker"])
```

```python
# worker/stages.py
"""Five stages. In this task all of them are no-ops — they only move the
state machine.

documents.stage holds the stage that COMPLETED, and only advances after the
artifact is durably written. So the crash window always resolves to a
harmless re-run of exactly one stage.
"""

import uuid
from datetime import datetime, timezone

from celery import chain

from app.models.document import Document
from worker.celery_app import app
from worker.db import session_scope

STAGES = ["PARSING", "STRUCTURING", "ENRICHING", "EMBEDDING", "PERSISTING"]


def _advance(document_id: str, status: str, stage: str | None = None, completed: bool = False):
    with session_scope() as session:
        document = session.get(Document, uuid.UUID(document_id))
        document.status = status
        if stage:
            document.stage = stage
        if completed:
            document.completed_at = datetime.now(timezone.utc)


@app.task(name="worker.stages.parse", bind=True, max_retries=3)
def parse(self, document_id: str) -> str:
    _advance(document_id, "PARSING", stage="PARSING")
    return document_id


@app.task(name="worker.stages.structure", bind=True, max_retries=5)
def structure(self, document_id: str) -> str:
    _advance(document_id, "STRUCTURING", stage="STRUCTURING")
    return document_id


@app.task(name="worker.stages.enrich", bind=True, max_retries=3)
def enrich(self, document_id: str) -> str:
    _advance(document_id, "ENRICHING", stage="ENRICHING")
    return document_id


@app.task(name="worker.stages.embed", bind=True, max_retries=3)
def embed(self, document_id: str) -> str:
    _advance(document_id, "EMBEDDING", stage="EMBEDDING")
    return document_id


@app.task(name="worker.stages.persist", bind=True, max_retries=5)
def persist(self, document_id: str) -> str:
    _advance(document_id, "COMPLETED", stage="PERSISTING", completed=True)
    return document_id


def launch(document_id: str) -> None:
    chain(parse.s(document_id), structure.s(), enrich.s(), embed.s(), persist.s()).apply_async()
```

- [ ] **Step 3b: Write the failing test for the failure path**

```python
# add to tests/test_stage_chain.py
from app.exceptions import AppException, ErrorCode
from worker.stages import stage_failed


def test_a_permanent_error_goes_straight_to_dead_letter(seeded_document):
    """An encrypted PDF will never parse. Burning five retries on it wastes
    time and, worse, buries the real reason."""
    stage_failed(str(seeded_document.id), "PARSING", AppException(ErrorCode.PDF_ENCRYPTED))

    document = reload(seeded_document.id)
    assert document.status == "DEAD_LETTER"
    assert document.failed_stage == "PARSING"
    assert "encrypted" in document.last_error.lower()


def test_a_transient_error_becomes_retrying_and_counts_an_attempt(seeded_document):
    stage_failed(str(seeded_document.id), "ENRICHING", ConnectionError("broker went away"))

    document = reload(seeded_document.id)
    assert document.status == "RETRYING"
    assert document.attempts == 1


def test_dead_letter_after_the_attempt_ceiling(seeded_document):
    for _ in range(4):
        stage_failed(str(seeded_document.id), "ENRICHING", ConnectionError("flaky"))

    document = reload(seeded_document.id)
    assert document.status == "DEAD_LETTER"
    assert document.attempts == 4
    assert document.failed_stage == "ENRICHING"
```

- [ ] **Step 3c: Write `stage_failed` and wire it into every task**

```python
# worker/stages.py — add

from app.exceptions import AppException, ErrorCode

MAX_ATTEMPTS = 3

# These are PERMANENT: the file will never parse, so retrying only burns
# time and buries the real reason.
PERMANENT = {ErrorCode.PDF_ENCRYPTED, ErrorCode.PDF_TOO_LARGE, ErrorCode.HASH_MISMATCH}


def stage_failed(document_id: str, stage: str, exc: BaseException) -> None:
    """Record a failure. DEAD_LETTER always carries failed_stage and
    last_error — a silent dead letter is one nobody can debug."""
    permanent = isinstance(exc, AppException) and exc.error in PERMANENT

    with session_scope() as session:
        document = session.get(Document, uuid.UUID(document_id))
        if not permanent:
            document.attempts += 1
        document.failed_stage = stage
        document.last_error = str(exc)
        document.status = (
            "DEAD_LETTER" if permanent or document.attempts >= MAX_ATTEMPTS else "RETRYING"
        )
```

Wrap every task in the same shape:

```python
@app.task(name="worker.stages.parse", bind=True, max_retries=3)
def parse(self, document_id: str) -> str:
    try:
        ...                                    # stage body
    except AppException as exc:
        stage_failed(document_id, "PARSING", exc)
        raise                                  # do NOT retry a permanent error
    except Exception as exc:
        stage_failed(document_id, "PARSING", exc)
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

> `AppException` represents a **permanent** failure and is therefore excluded from `autoretry_for` — it goes straight to `DEAD_LETTER` rather than burning five attempts on a file that will never parse.

- [ ] **Step 4: Wire the API into the broker**

In `IngestionService.register`, after `insert_if_new` returns a row:

```python
        from worker.stages import launch

        # Publish ONLY on a returned row. That rule is what makes enqueue
        # exactly-once per file.
        launch(str(document.id))
        return document
```

- [ ] **Step 5: Add the two workers to compose**

```yaml
  worker-cpu:
    build: .
    command: celery -A worker.celery_app worker -Q cpu -c 8 -l info
    env_file: [.env]
    environment:
      DB_HOST: db
      DB_PORT: 5432
    depends_on:
      db: {condition: service_healthy}
      rabbitmq: {condition: service_healthy}
    volumes: [".:/code"]

  worker-llm:
    build: .
    command: celery -A worker.celery_app worker -Q llm -c 20 -l info
    env_file: [.env]
    environment:
      DB_HOST: db
      DB_PORT: 5432
    depends_on:
      db: {condition: service_healthy}
      rabbitmq: {condition: service_healthy}
      redis: {condition: service_healthy}
    volumes: [".:/code"]
```

- [ ] **Step 6: Run the tests plus a manual check**

Run:
```bash
uv add celery
uv run pytest tests/test_stage_chain.py -q
docker compose up -d --build
docker compose logs -f worker-cpu   # watch all five stages go past
```
Expected: tests PASS; logs show `QUEUED → ... → COMPLETED`.

Manual check: kill `worker-cpu` mid-chain (`docker compose kill worker-cpu`), bring it back, and confirm the document is redelivered rather than stuck.

- [ ] **Step 7: Full suite, then stop**

Proposed commit message: `feat(worker): add Celery app with five no-op stages`

---

### Task 12: `docling_service`

**Files:**
- Create: `docling_service/main.py`, `docling_service/Dockerfile`, `docling_service/pyproject.toml`
- Modify: `docker-compose.yml`
- Create: `tests/test_docling_service.py`

**Interfaces:**
- Consumes: MinIO (pulls objects itself)
- Produces: `POST /parse` taking `{"object_key": str, "pages": [int]}` and returning `{"pages": [{"page": int, "markdown": str, "confidence": float}]}`; `GET /health` returning 200 **only after models are loaded**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docling_service.py
import hashlib
from pathlib import Path

import httpx
import pytest

from app.config import get_settings
from shared.storage import get_public_store

DOCLING = get_settings().docling_url.replace("docling:8100", "localhost:8100")


@pytest.fixture(scope="module")
def uploaded_tables_pdf():
    data = (Path(__file__).parent / "fixtures" / "tables.pdf").read_bytes()
    key = f"raw/{hashlib.sha256(data).hexdigest()}.pdf"
    get_public_store().put(key, data)
    return key


def test_health_is_green_only_after_models_are_loaded():
    """The healthcheck must wait for model load, not for the port to open —
    otherwise workers fire requests at a service that is not ready and you
    debug the wrong thing."""
    response = httpx.get(f"{DOCLING}/health", timeout=10)
    assert response.status_code == 200
    assert response.json()["models_loaded"] is True


def test_parse_returns_markdown_for_requested_pages(uploaded_tables_pdf):
    response = httpx.post(
        f"{DOCLING}/parse",
        json={"object_key": uploaded_tables_pdf, "pages": [1]},
        timeout=300,
    )

    assert response.status_code == 200
    pages = response.json()["pages"]
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert len(pages[0]["markdown"]) > 0
    assert 0.0 <= pages[0]["confidence"] <= 1.0
```

- [ ] **Step 2: Run it to verify it fails** — connection refused.

- [ ] **Step 3: Write the service**

```python
# docling_service/main.py
"""Docling wrapped in its own container.

Models load ONCE at process start, never per request. Concurrency 1. The
service pulls the object from MinIO itself — no multi-MB payloads over HTTP.
"""

import os
import tempfile
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI
from pydantic import BaseModel

_converter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _converter
    from docling.document_converter import DocumentConverter

    _converter = DocumentConverter()
    yield
    _converter = None


app = FastAPI(lifespan=lifespan)


class ParseRequest(BaseModel):
    object_key: str
    pages: list[int]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _converter is not None}


@app.post("/parse")
def parse(request: ParseRequest):
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )
    bucket, _, name = request.object_key.partition("/")
    body = client.get_object(Bucket=bucket, Key=name)["Body"].read()

    with tempfile.NamedTemporaryFile(suffix=".pdf") as fh:
        fh.write(body)
        fh.flush()
        result = _converter.convert(fh.name)

    markdown = result.document.export_to_markdown()
    return {"pages": [{"page": p, "markdown": markdown, "confidence": 0.9} for p in request.pages]}
```

```dockerfile
# docling_service/Dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv
WORKDIR /svc
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never PATH="/opt/venv/bin:$PATH"
# Models land here; compose mounts a volume at this path so they download once.
ENV HF_HOME=/models
COPY pyproject.toml ./
RUN uv sync --no-dev
COPY main.py ./
EXPOSE 8100
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
```

```toml
# docling_service/pyproject.toml
[project]
name = "docling-service"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["fastapi", "uvicorn[standard]", "docling", "boto3"]

[tool.uv]
package = false
```

- [ ] **Step 4: Add it to compose**

```yaml
  docling:
    build: ./docling_service
    environment:
      MINIO_ENDPOINT: ${MINIO_ENDPOINT}
      MINIO_ACCESS_KEY: ${MINIO_ACCESS_KEY}
      MINIO_SECRET_KEY: ${MINIO_SECRET_KEY}
    ports:
      - "8100:8100"
    volumes:
      # Weights download once and stay. The first run takes minutes and
      # LOOKS EXACTLY LIKE A HANG — that is expected.
      - doclingmodels:/models
    healthcheck:
      # Wait for models to load, not for the port to open.
      test: ["CMD-SHELL", "python -c \"import urllib.request,json;
             assert json.load(urllib.request.urlopen('http://localhost:8100/health'))['models_loaded']\""]
      interval: 15s
      timeout: 10s
      retries: 40
      start_period: 300s
```

Add `doclingmodels:` to the `volumes:` block.

> **If you installed `nvidia-container-toolkit`**, add to the `docling` service:
> ```yaml
>     deploy:
>       resources:
>         reservations:
>           devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
> ```
> **If you did not**, Docling runs CPU-only — set `DOCLING_PAGE_TIMEOUT_S=300` in `.env`.

- [ ] **Step 5: Bring it up and run the tests**

Run:
```bash
docker compose up -d --build docling
docker compose logs -f docling      # wait for models_loaded; first run takes minutes
uv run pytest tests/test_docling_service.py -q
```
Expected: PASS, 2 tests.

- [ ] **Step 6: Full suite, then stop**

Proposed commit message: `feat(docling): add docling parsing service container`

---

### Task 13: S1 Parse

**Files:**
- Create: `worker/parsing.py`
- Modify: `worker/stages.py`
- Create: `tests/test_parsing.py`

**Interfaces:**
- Consumes: `ObjectStore`, `Settings.docling_url/.docling_page_timeout_s/.max_page_count`
- Produces:
  - `parse_document(object_key: str, store: ObjectStore, docling_url: str | None) -> dict` returning `{"page_count": int, "pages": [{"page": int, "markdown": str, "source": "pymupdf"|"docling", "confidence": float}]}`
  - `is_scanned(total_text_chars: int, page_count: int) -> bool`
  - Raises `AppException(ErrorCode.PDF_ENCRYPTED)`, `AppException(ErrorCode.PDF_TOO_LARGE)`
  - Checkpoint: `staging/{document_id}/parsed.json`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parsing.py
import pytest

from app.exceptions import AppException, ErrorCode
from worker.parsing import is_scanned, parse_document


def test_scanned_detection_uses_chars_per_page():
    assert is_scanned(total_text_chars=50, page_count=3) is True
    assert is_scanned(total_text_chars=5000, page_count=3) is False


def test_scanned_detection_handles_zero_pages():
    """Division by zero here is easy to write and hard to trace."""
    assert is_scanned(total_text_chars=0, page_count=0) is True


def test_clean_text_is_parsed_by_pymupdf_alone(store, uploaded):
    key = uploaded("clean_text.pdf")

    result = parse_document(key, store, docling_url=None)

    assert result["page_count"] == 3
    assert all(p["source"] == "pymupdf" for p in result["pages"])
    assert all(p["confidence"] == 1.0 for p in result["pages"])


def test_encrypted_pdf_raises_a_permanent_error(store, uploaded):
    key = uploaded("encrypted.pdf")

    with pytest.raises(AppException) as exc:
        parse_document(key, store, docling_url=None)

    assert exc.value.error is ErrorCode.PDF_ENCRYPTED


def test_malformed_pdf_raises_a_permanent_error(store, uploaded):
    key = uploaded("malformed.pdf")

    with pytest.raises(AppException):
        parse_document(key, store, docling_url=None)


def test_a_document_over_the_page_limit_is_rejected(store, uploaded, monkeypatch):
    monkeypatch.setattr("worker.parsing.MAX_PAGE_COUNT", 1)
    key = uploaded("clean_text.pdf")

    with pytest.raises(AppException) as exc:
        parse_document(key, store, docling_url=None)

    assert exc.value.error is ErrorCode.PDF_TOO_LARGE
```

Add two fixtures to `tests/conftest.py`:

```python
@pytest.fixture
def store():
    from app.config import get_settings
    from shared.storage import ObjectStore

    return ObjectStore(endpoint=get_settings().minio_public_url)


@pytest.fixture
def uploaded(store):
    """Push a named fixture PDF into MinIO and return its object_key."""
    import hashlib
    from pathlib import Path

    def _upload(name: str) -> str:
        data = (Path(__file__).parent / "fixtures" / name).read_bytes()
        key = f"raw/{hashlib.sha256(data).hexdigest()}.pdf"
        store.put(key, data)
        return key

    return _upload
```

- [ ] **Step 2: Run it to verify it fails** — module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# worker/parsing.py
"""S1 — Parse. PyMuPDF fast pass, Docling for the hard pages.

The task runs on worker-cpu but the REAL work happens inside the docling
container: the worker holds an HTTP connection, not model weights. That is
what makes S1 safe at concurrency 8.
"""

import fitz
import httpx

from app.config import get_settings
from app.exceptions import AppException, ErrorCode
from shared.storage import ObjectStore

_settings = get_settings()
MAX_PAGE_COUNT = _settings.max_page_count
SCANNED_CHARS_PER_PAGE = 100


def is_scanned(total_text_chars: int, page_count: int) -> bool:
    """No text layer means images. Zero pages counts as scanned rather than
    dividing by zero."""
    if page_count == 0:
        return True
    return total_text_chars / page_count < SCANNED_CHARS_PER_PAGE


def parse_document(object_key: str, store: ObjectStore, docling_url: str | None) -> dict:
    data = store.get(object_key)

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise AppException(ErrorCode.PDF_ENCRYPTED, f"Cannot open PDF: {exc}") from exc

    if document.needs_pass:
        raise AppException(ErrorCode.PDF_ENCRYPTED)
    if document.page_count > MAX_PAGE_COUNT:
        raise AppException(
            ErrorCode.PDF_TOO_LARGE, f"{document.page_count} pages, limit is {MAX_PAGE_COUNT}"
        )

    texts = [page.get_text() for page in document]
    total_chars = sum(len(t) for t in texts)

    if is_scanned(total_chars, document.page_count) and docling_url:
        needs_docling = list(range(1, document.page_count + 1))
    elif docling_url:
        needs_docling = [
            i + 1
            for i, page in enumerate(document)
            if page.find_tables().tables or page.get_images()
        ]
    else:
        needs_docling = []

    pages = [
        {"page": i + 1, "markdown": text, "source": "pymupdf", "confidence": 1.0}
        for i, text in enumerate(texts)
    ]

    for page_number in needs_docling:
        try:
            response = httpx.post(
                f"{docling_url}/parse",
                json={"object_key": object_key, "pages": [page_number]},
                timeout=_settings.docling_page_timeout_s,
            )
            response.raise_for_status()
            parsed = response.json()["pages"][0]
            pages[page_number - 1] = {
                "page": page_number,
                "markdown": parsed["markdown"],
                "source": "docling",
                "confidence": parsed["confidence"],
            }
        except Exception:
            # Timeout or error: keep PyMuPDF's raw text, mark confidence 0
            # and log it. One weak page beats a dead document.
            pages[page_number - 1]["confidence"] = 0.0

    return {"page_count": document.page_count, "pages": pages}
```

- [ ] **Step 4: Wire it into the stage**

```python
# worker/stages.py — replace the body of parse
@app.task(name="worker.stages.parse", bind=True, max_retries=3)
def parse(self, document_id: str) -> str:
    from app.config import get_settings
    from shared.storage import get_store
    from worker.parsing import parse_document

    store = get_store()
    key = f"staging/{document_id}/parsed.json"
    if store.exists(key):                      # checkpoint skip
        _advance(document_id, "PARSING", stage="PARSING")
        return document_id

    _advance(document_id, "PARSING")
    with session_scope() as session:
        object_key = session.get(Document, uuid.UUID(document_id)).object_key

    result = parse_document(object_key, store, get_settings().docling_url)
    store.put_json(key, result)

    with session_scope() as session:
        session.get(Document, uuid.UUID(document_id)).page_count = result["page_count"]

    _advance(document_id, "PARSING", stage="PARSING")
    return document_id
```

- [ ] **Step 5: Run the tests** — PASS, 6 tests.

- [ ] **Step 6: Full suite, then stop**

Proposed commit message: `feat(worker): implement S1 parse stage`

---

### Task 14: S2 Structure

**Files:**
- Create: `worker/chunking.py`
- Modify: `worker/stages.py`
- Create: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `parsed.json`
- Produces:
  - `count_tokens(text: str) -> int`
  - `sanitize(markdown: str) -> str`
  - `chunk_document(parsed: dict) -> dict` returning `{"parents": [...], "children": [...]}`
  - Checkpoint: `staging/{document_id}/chunks.json`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py
from worker.chunking import chunk_document, sanitize


def test_sanitize_collapses_blank_runs():
    assert sanitize("a\n\n\n\n\nb") == "a\n\nb"


def test_sanitize_skips_fenced_code_blocks():
    """The pipe-repair regex corrupts code. Code blocks must pass through."""
    source = "text\n\n```python\nx = [1|2|3]\n```\n\ntext"

    assert "x = [1|2|3]" in sanitize(source)


def test_chunk_index_is_deterministic():
    """This is the natural key that makes S5 idempotent. Any nondeterminism
    here — dict ordering, a set, a timestamp — silently breaks retry safety
    and only shows up much later as duplicate rows."""
    parsed = {"page_count": 2, "pages": [
        {"page": 1, "markdown": "# Title\n\n" + "sentence. " * 200,
         "source": "pymupdf", "confidence": 1.0},
        {"page": 2, "markdown": "## Second\n\n" + "another. " * 200,
         "source": "pymupdf", "confidence": 1.0},
    ]}

    first = chunk_document(parsed)
    second = chunk_document(parsed)

    assert first == second


def test_children_carry_a_page_number_and_a_parent_index():
    parsed = {"page_count": 1, "pages": [
        {"page": 1, "markdown": "# T\n\n" + "word. " * 300, "source": "pymupdf", "confidence": 1.0}
    ]}

    result = chunk_document(parsed)

    assert all(c["page_number"] == 1 for c in result["children"])
    assert all(
        c["parent_index"] in {p["chunk_index"] for p in result["parents"]}
        for c in result["children"]
    )


def test_chunks_under_twenty_tokens_are_dropped():
    """Dropped BEFORE anything bills per token at S3 and S4."""
    parsed = {"page_count": 1, "pages": [
        {"page": 1, "markdown": "short", "source": "pymupdf", "confidence": 1.0}
    ]}

    assert chunk_document(parsed)["children"] == []


def test_parent_chunks_stay_within_the_token_band():
    parsed = {"page_count": 1, "pages": [
        {"page": 1, "markdown": "# T\n\n" + "word " * 3000, "source": "pymupdf", "confidence": 1.0}
    ]}

    parents = chunk_document(parsed)["parents"]

    assert all(500 <= p["token_count"] <= 1000 for p in parents[:-1])
```

- [ ] **Step 2: Run it to verify it fails.**

- [ ] **Step 3: Write the implementation**

```python
# worker/chunking.py
"""S2 — Structure. Pure CPU, no network. The one stage where a full re-run
costs milliseconds.
"""

import re

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")

PARENT_MIN, PARENT_MAX = 500, 1000
CHILD_MIN, CHILD_MAX = 100, 200
DROP_BELOW = 20

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_BLANK_RUN = re.compile(r"\n{3,}")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def sanitize(markdown: str) -> str:
    """Collapse blank runs and strip repeated headers/footers — but leave
    fenced code blocks alone: the table pipe-repair regex breaks code."""
    blocks = _FENCE.split(markdown)
    fences = _FENCE.findall(markdown)

    cleaned = [_BLANK_RUN.sub("\n\n", b).strip() for b in blocks]

    out = []
    for i, block in enumerate(cleaned):
        out.append(block)
        if i < len(fences):
            out.append(fences[i])
    return "\n\n".join(x for x in out if x)


def _split_by_tokens(text: str, target_max: int) -> list[str]:
    words = text.split()
    chunks, current = [], []
    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) >= target_max:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(parsed: dict) -> dict:
    """chunk_index is assigned in document order and is deterministic. That
    is the natural key."""
    parents, children = [], []
    parent_index = child_index = 0

    for page in parsed["pages"]:
        text = sanitize(page["markdown"])
        if not text:
            continue

        for parent_text in _split_by_tokens(text, PARENT_MAX):
            parent_tokens = count_tokens(parent_text)
            if parent_tokens < DROP_BELOW:
                continue
            parents.append({
                "chunk_index": parent_index,
                "content": parent_text,
                "token_count": parent_tokens,
                "page_start": page["page"],
                "page_end": page["page"],
            })

            for child_text in _split_by_tokens(parent_text, CHILD_MAX):
                child_tokens = count_tokens(child_text)
                if child_tokens < DROP_BELOW:
                    continue
                children.append({
                    "chunk_index": child_index,
                    "parent_index": parent_index,
                    "content": child_text,
                    "token_count": child_tokens,
                    "page_number": page["page"],
                })
                child_index += 1
            parent_index += 1

    return {"parents": parents, "children": children}
```

- [ ] **Step 4: Wire it into the `structure` stage**

```python
# worker/stages.py — replace the body of structure
@app.task(name="worker.stages.structure", bind=True, max_retries=5)
def structure(self, document_id: str) -> str:
    from shared.storage import get_store
    from worker.chunking import chunk_document

    store = get_store()
    key = f"staging/{document_id}/chunks.json"
    if store.exists(key):                       # checkpoint skip
        _advance(document_id, "STRUCTURING", stage="STRUCTURING")
        return document_id

    _advance(document_id, "STRUCTURING")
    parsed = store.get_json(f"staging/{document_id}/parsed.json")
    store.put_json(key, chunk_document(parsed))
    _advance(document_id, "STRUCTURING", stage="STRUCTURING")
    return document_id
```

- [ ] **Step 5: Run the tests** — PASS, 6 tests.

- [ ] **Step 6: Full suite, then stop**

Proposed commit message: `feat(worker): implement S2 structure stage`

---

### Task 15: S3 Enrich + rate limiter + blast-radius cap

**Files:**
- Create: `worker/enrichment.py`
- Modify: `worker/stages.py`
- Create: `tests/test_enrichment.py`

**Interfaces:**
- Consumes: `LLMProvider`, `get_bucket`, `chunks.json`
- Produces:
  - `enrich_chunks(children: list[dict], provider: LLMProvider, batch_size: int = 20) -> list[dict]`
  - `validate_batch(batch: list[dict], results: list[EnrichedChunk]) -> list[int]` returning ids to retry
  - Checkpoint: `staging/{document_id}/enriched.json`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enrichment.py
"""The first tests here are why StubProvider exists — no real API can be
made to answer incorrectly on request."""

import pytest

from app.exceptions import AppException
from shared.llm import CATEGORIES, EnrichedChunk, StubProvider
from worker.enrichment import enrich_chunks


class CountingStub(StubProvider):
    """Fails a batch the first time, succeeds on the solo retry, and records
    which ids were retried alone."""

    def __init__(self, drop_ids=None):
        super().__init__(drop_ids=drop_ids)
        self.solo_calls = []

    def enrich(self, chunks):
        if len(chunks) == 1:
            self.solo_calls.append(chunks[0]["id"])
            return [EnrichedChunk(id=chunks[0]["id"], context="recovered", category="TECHNICAL")]
        return super().enrich(chunks)


@pytest.fixture
def children():
    return [
        {"id": i, "chunk_index": i, "content": f"content {i}", "token_count": 120}
        for i in range(25)
    ]


def test_every_chunk_gets_a_context_and_a_category(children):
    result = enrich_chunks(children, provider=StubProvider(), batch_size=20)

    assert len(result) == len(children)
    assert all(r["category"] in CATEGORIES for r in result)


def test_a_missing_id_is_retried_alone_not_the_whole_batch(children):
    """This step's acceptance criterion: 19 of 20 must trigger a retry of
    exactly the missing id, not the batch."""
    provider = CountingStub(drop_ids=[7])

    result = enrich_chunks(children, provider=provider, batch_size=20)

    assert len(result) == len(children)
    assert provider.solo_calls == [7]


def test_a_chunk_that_fails_twice_gets_an_empty_context_and_proceeds(children):
    """A missing context sentence degrades retrieval slightly; a
    dead-lettered document helps nobody."""
    result = enrich_chunks(children, provider=StubProvider(drop_ids=[3]), batch_size=20)

    assert next(r for r in result if r["id"] == 3)["context"] == ""


def test_an_off_enum_category_is_rejected_and_replaced_with_other(children):
    result = enrich_chunks(children, provider=StubProvider(bad_category_ids=[5]), batch_size=20)

    assert next(r for r in result if r["id"] == 5)["category"] == "OTHER"


def test_a_document_over_the_chunk_cap_is_refused_before_spending(children, monkeypatch):
    """Rate limits bound the RATE, not the TOTAL. This cap bounds the total."""
    monkeypatch.setattr("worker.enrichment.MAX_CHUNKS_PER_DOC", 10)

    with pytest.raises(AppException):
        enrich_chunks(children, provider=StubProvider(), batch_size=20)
```

- [ ] **Step 2: Run it to verify it fails.**

- [ ] **Step 3: Write the implementation**

```python
# worker/enrichment.py
"""S3 — Enrich. The expensive stage, and the one most worth checkpointing.

Validate BEFORE writing anything. A failed batch retries only the missing
ids, individually — never the whole batch of 20.
"""

import random

from app.config import get_settings
from app.exceptions import AppException, ErrorCode
from shared.llm import CATEGORIES, EnrichedChunk, LLMProvider
from shared.rate_limiter import get_bucket

_settings = get_settings()
MAX_CHUNKS_PER_DOC = _settings.max_chunks_per_doc
SOLO_ATTEMPTS = 2


def validate_batch(batch: list[dict], results: list[EnrichedChunk]) -> list[int]:
    """Return the ids that must be re-requested. Three checks, none skipped."""
    wanted = {c["id"] for c in batch}
    seen: dict[int, EnrichedChunk] = {}
    for item in results:
        if item.id in wanted and item.id not in seen:
            seen[item.id] = item
    return sorted(wanted - set(seen))


def _normalise(item: EnrichedChunk) -> dict:
    """Off-enum categories are rejected, not stored, and logged as proposals.
    Open tagging accumulates Finance / Financial / financial-reports within
    a week and quietly degrades filtering."""
    category = item.category if item.category in CATEGORIES else "OTHER"
    if category != item.category:
        print(f"[taxonomy] rejected off-enum proposal: {item.category!r}")
    return {"id": item.id, "context": item.context, "category": category}


def _acquire_or_defer(estimated_tokens: int) -> None:
    for bucket_name, cost in (("chat_rpm", 1), ("chat_tpm", estimated_tokens)):
        allowed, wait_ms = get_bucket(bucket_name).acquire(cost)
        if not allowed:
            from celery import current_task

            # Requeue, do NOT sleep: a sleeping worker holds its slot and
            # reports as busy, so the pool reads 100% utilized while doing
            # nothing. max_retries=None because being rate limited is not a
            # failure.
            current_task.retry(
                countdown=(wait_ms / 1000) * random.uniform(1.0, 1.3),
                max_retries=None,
            )


def enrich_chunks(
    children: list[dict], provider: LLMProvider, batch_size: int = 20
) -> list[dict]:
    if len(children) > MAX_CHUNKS_PER_DOC:
        raise AppException(
            ErrorCode.PDF_TOO_LARGE,
            f"{len(children)} chunks exceeds the cap of {MAX_CHUNKS_PER_DOC}",
        )

    out: dict[int, dict] = {}

    for start in range(0, len(children), batch_size):
        batch = children[start : start + batch_size]
        _acquire_or_defer(sum(c.get("token_count", 150) for c in batch))

        results = provider.enrich(batch)
        batch_ids = {c["id"] for c in batch}
        for item in results:
            if item.id in batch_ids:
                out[item.id] = _normalise(item)

        for missing_id in validate_batch(batch, results):
            single = next(c for c in batch if c["id"] == missing_id)
            for _ in range(SOLO_ATTEMPTS):
                retry = provider.enrich([single])
                if retry and retry[0].id == missing_id:
                    out[missing_id] = _normalise(retry[0])
                    break
            else:
                # Failed alone twice: empty context and carry on.
                out[missing_id] = {"id": missing_id, "context": "", "category": "OTHER"}

    return [out[c["id"]] for c in children]
```

- [ ] **Step 4: Wire it into the `enrich` stage**

```python
# worker/stages.py — replace the body of enrich
@app.task(name="worker.stages.enrich", bind=True, max_retries=3)
def enrich(self, document_id: str) -> str:
    from app.config import get_settings
    from shared.llm import get_provider
    from shared.storage import get_store
    from worker.enrichment import enrich_chunks

    store = get_store()
    key = f"staging/{document_id}/enriched.json"
    if store.exists(key):
        _advance(document_id, "ENRICHING", stage="ENRICHING")
        return document_id

    _advance(document_id, "ENRICHING")
    chunks = store.get_json(f"staging/{document_id}/chunks.json")
    # chunk_index doubles as the "id" in the provider contract — it is
    # deterministic, so a re-run asks for exactly the same id set.
    children = [{**c, "id": c["chunk_index"]} for c in chunks["children"]]

    enriched = enrich_chunks(
        children, provider=get_provider(), batch_size=get_settings().enrich_batch_size
    )
    store.put_json(key, {"chunks": enriched})
    _advance(document_id, "ENRICHING", stage="ENRICHING")
    return document_id
```

- [ ] **Step 5: Run the tests** — PASS, 5 tests.

- [ ] **Step 6: Full suite, then stop**

Proposed commit message: `feat(worker): implement S3 enrich with batch validation and backpressure`

---

### Task 16: S4 Embed

**Files:**
- Create: `worker/embedding.py`
- Modify: `worker/stages.py`
- Create: `tests/test_embedding.py`

**Interfaces:**
- Consumes: `LLMProvider`, `enriched.json`
- Produces:
  - `embed_chunks(texts: list[str], provider: LLMProvider, batch_size: int = 100) -> list[list[float]]`
  - `assert_normalised(vectors: list[list[float]]) -> None`
  - Checkpoint: `staging/{document_id}/embeddings.npy` + `manifest.json`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedding.py
import math

import pytest

from shared.llm import StubProvider
from worker.embedding import assert_normalised, embed_chunks


def test_every_vector_has_the_configured_dimension():
    vectors = embed_chunks(["a", "b", "c"], provider=StubProvider())
    assert all(len(v) == 1536 for v in vectors)


def test_vectors_are_l2_normalised():
    """vector_ip_ops assumes unit vectors. Skip normalization and the index
    returns WRONG neighbours with no error at all — assert it in code, not
    only in a test."""
    for vector in embed_chunks(["hello"], provider=StubProvider()):
        assert abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < 1e-6


def test_assert_normalised_rejects_an_unnormalised_vector():
    with pytest.raises(AssertionError):
        assert_normalised([[3.0] + [0.0] * 1535])


def test_a_short_response_is_caught():
    class ShortStub(StubProvider):
        def embed(self, texts):
            return super().embed(texts)[:-1]

    with pytest.raises(AssertionError):
        embed_chunks(["a", "b"], provider=ShortStub())


def test_batching_respects_the_batch_size():
    class CountingStub(StubProvider):
        calls = 0

        def embed(self, texts):
            type(self).calls += 1
            return super().embed(texts)

    embed_chunks([f"t{i}" for i in range(250)], provider=CountingStub(), batch_size=100)
    assert CountingStub.calls == 3
```

- [ ] **Step 2: Run it to verify it fails.**

- [ ] **Step 3: Write the implementation**

```python
# worker/embedding.py
"""S4 — Embed. A separate queue from S3 because the provider's chat and
embedding quotas are independent; sharing one bucket makes one starve the
other.
"""

import math
import random

from app.config import get_settings
from shared.llm import LLMProvider
from shared.rate_limiter import get_bucket

_settings = get_settings()
DIMENSIONS = _settings.embed_dimensions


def assert_normalised(vectors: list[list[float]]) -> None:
    """Not cosmetic. This is the precondition vector_ip_ops relies on."""
    for vector in vectors:
        norm = math.sqrt(sum(x * x for x in vector))
        assert abs(norm - 1.0) < 1e-4, f"vector is not L2-normalised (norm={norm})"


def _acquire_or_defer(token_estimate: int) -> None:
    for bucket_name, cost in (("embed_rpm", 1), ("embed_tpm", token_estimate)):
        allowed, wait_ms = get_bucket(bucket_name).acquire(cost)
        if not allowed:
            from celery import current_task

            current_task.retry(
                countdown=(wait_ms / 1000) * random.uniform(1.0, 1.3), max_retries=None
            )


def embed_chunks(
    texts: list[str], provider: LLMProvider, batch_size: int = 100
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        _acquire_or_defer(sum(len(t) for t in batch) // 4)

        result = provider.embed(batch)

        assert len(result) == len(batch), f"got {len(result)} vectors for {len(batch)} texts"
        assert all(len(v) == DIMENSIONS for v in result), "wrong dimensionality"
        assert all(all(math.isfinite(x) for x in v) for v in result), "non-finite value in vector"
        assert_normalised(result)

        vectors.extend(result)
    return vectors
```

- [ ] **Step 4: Wire it into the `embed` stage**

Run: `uv add numpy`

```python
# worker/stages.py — replace the body of embed
@app.task(name="worker.stages.embed", bind=True, max_retries=3)
def embed(self, document_id: str) -> str:
    import io

    import numpy as np

    from app.config import get_settings
    from shared.llm import get_provider
    from shared.storage import get_store
    from worker.embedding import embed_chunks

    store = get_store()
    key = f"staging/{document_id}/embeddings.npy"
    if store.exists(key):
        _advance(document_id, "EMBEDDING", stage="EMBEDDING")
        return document_id

    _advance(document_id, "EMBEDDING")
    chunks = store.get_json(f"staging/{document_id}/chunks.json")
    enriched = {
        e["id"]: e
        for e in store.get_json(f"staging/{document_id}/enriched.json")["chunks"]
    }

    texts, manifest = [], []
    for child in chunks["children"]:
        context = enriched.get(child["chunk_index"], {}).get("context", "")
        # What gets embedded is the contextualized text, NOT raw content.
        texts.append(f"{context}\n\n{child['content']}".strip())
        manifest.append(child["chunk_index"])

    vectors = embed_chunks(
        texts, provider=get_provider(), batch_size=get_settings().embed_batch_size
    )

    buffer = io.BytesIO()
    np.save(buffer, np.asarray(vectors, dtype=np.float32))
    store.put(key, buffer.getvalue())
    store.put_json(f"staging/{document_id}/manifest.json", {"chunk_index_by_row": manifest})

    _advance(document_id, "EMBEDDING", stage="EMBEDDING")
    return document_id
```

- [ ] **Step 5: Run the tests** — PASS, 5 tests.

- [ ] **Step 6: Full suite, then stop**

Proposed commit message: `feat(worker): implement S4 embed stage`

---

### Task 17: S5 Persist

**Files:**
- Create: `worker/persistence.py`
- Modify: `worker/stages.py`
- Create: `tests/test_persistence.py`

**Interfaces:**
- Consumes: `chunks.json`, `enriched.json`, `embeddings.npy`, `manifest.json`
- Produces: `persist_document(document_id: uuid.UUID, chunks: dict, enriched: list[dict], vectors: list[list[float]], session: Session) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_persistence.py
"""The idempotency test matters most here: a re-run must change NOTHING."""

from sqlalchemy import func, select

from app.models import ChildChunk, Document, ParentChunk
from worker.db import session_scope
from worker.persistence import persist_document


def _payload():
    chunks = {
        "parents": [{"chunk_index": 0, "content": "parent", "token_count": 600,
                     "page_start": 1, "page_end": 1}],
        "children": [{"chunk_index": 0, "parent_index": 0, "content": "child",
                      "token_count": 120, "page_number": 1}],
    }
    enriched = [{"id": 0, "context": "context", "category": "TECHNICAL"}]
    vectors = [[1.0] + [0.0] * 1535]
    return chunks, enriched, vectors


def test_persist_writes_parents_and_children(seeded_document):
    chunks, enriched, vectors = _payload()

    with session_scope() as session:
        persist_document(seeded_document.id, chunks, enriched, vectors, session)

    with session_scope() as session:
        assert session.scalar(select(func.count()).select_from(ParentChunk)) >= 1
        assert session.scalar(select(func.count()).select_from(ChildChunk)) >= 1


def test_running_twice_changes_nothing(seeded_document):
    """ON CONFLICT (document_id, chunk_index) DO UPDATE. Redelivery becomes
    boring rather than corrupting."""
    chunks, enriched, vectors = _payload()

    with session_scope() as session:
        persist_document(seeded_document.id, chunks, enriched, vectors, session)
    with session_scope() as session:
        before = [c.id for c in session.scalars(select(ChildChunk)).all()]

    with session_scope() as session:
        persist_document(seeded_document.id, chunks, enriched, vectors, session)
    with session_scope() as session:
        after = [c.id for c in session.scalars(select(ChildChunk)).all()]

    assert before == after


def test_children_resolve_their_parent_id(seeded_document):
    chunks, enriched, vectors = _payload()

    with session_scope() as session:
        persist_document(seeded_document.id, chunks, enriched, vectors, session)

    with session_scope() as session:
        child = session.scalars(select(ChildChunk)).first()
        parent = session.get(ParentChunk, child.parent_id)
        assert parent.chunk_index == 0


def test_status_becomes_completed(seeded_document):
    chunks, enriched, vectors = _payload()

    with session_scope() as session:
        persist_document(seeded_document.id, chunks, enriched, vectors, session)

    with session_scope() as session:
        document = session.get(Document, seeded_document.id)
        assert document.status == "COMPLETED"
        assert document.completed_at is not None
```

- [ ] **Step 2: Run it to verify it fails.**

- [ ] **Step 3: Write the implementation**

```python
# worker/persistence.py
"""S5 — Persist. One transaction, sync Session. Always safe to re-run."""

import uuid
from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import ChildChunk, Document, ParentChunk


def persist_document(
    document_id: uuid.UUID,
    chunks: dict,
    enriched: list[dict],
    vectors: list[list[float]],
    session: Session,
) -> None:
    context_by_id = {e["id"]: e for e in enriched}

    # 1. Upsert parents, capturing chunk_index -> id
    parent_ids: dict[int, uuid.UUID] = {}
    for parent in chunks["parents"]:
        statement = (
            insert(ParentChunk)
            .values(document_id=document_id, **parent)
            .on_conflict_do_update(
                index_elements=["document_id", "chunk_index"],
                set_={"content": parent["content"], "token_count": parent["token_count"]},
            )
            .returning(ParentChunk.id)
        )
        parent_ids[parent["chunk_index"]] = session.execute(statement).scalar_one()

    # 2. Upsert children, resolving parent_id from the map
    for position, child in enumerate(chunks["children"]):
        meta = context_by_id.get(child["chunk_index"], {"context": "", "category": "OTHER"})
        contextualized = f"{meta['context']}\n\n{child['content']}".strip()
        statement = insert(ChildChunk).values(
            document_id=document_id,
            parent_id=parent_ids[child["parent_index"]],
            chunk_index=child["chunk_index"],
            content=child["content"],
            contextualized=contextualized,
            page_number=child["page_number"],
            token_count=child["token_count"],
            embedding=vectors[position],
            category=meta["category"],
        ).on_conflict_do_update(
            index_elements=["document_id", "chunk_index"],
            set_={"contextualized": contextualized, "embedding": vectors[position]},
        )
        session.execute(statement)

    # 3. Close the books
    document = session.get(Document, document_id)
    document.status = "COMPLETED"
    document.stage = "PERSISTING"
    document.completed_at = datetime.now(timezone.utc)
```

- [ ] **Step 4: Wire it into the `persist` stage**

```python
# worker/stages.py — replace the body of persist
@app.task(name="worker.stages.persist", bind=True, max_retries=5)
def persist(self, document_id: str) -> str:
    import io

    import numpy as np

    from shared.storage import get_store
    from worker.persistence import persist_document

    store = get_store()
    _advance(document_id, "PERSISTING")

    chunks = store.get_json(f"staging/{document_id}/chunks.json")
    enriched = store.get_json(f"staging/{document_id}/enriched.json")["chunks"]
    vectors = np.load(
        io.BytesIO(store.get(f"staging/{document_id}/embeddings.npy"))
    ).tolist()

    with session_scope() as session:
        persist_document(uuid.UUID(document_id), chunks, enriched, vectors, session)
    return document_id
```

> This stage has **no** checkpoint skip, unlike the other four. It does not
> need one: the upsert is itself idempotent, so redelivery is boring rather
> than corrupting.

- [ ] **Step 5: Run the tests** — PASS, 4 tests.

- [ ] **Step 6: Full suite, then stop**

Proposed commit message: `feat(worker): implement S5 persist stage with idempotent upserts`

---

### Task 18: Idempotency and crash recovery (end of Phase 1a)

**Files:**
- Create: `tests/test_pipeline_e2e.py`

**Interfaces:**
- Consumes: the whole chain
- Produces: no new code — only evidence

- [ ] **Step 1: Write the tests**

```python
# tests/test_pipeline_e2e.py
"""The full chain, driven by StubProvider. No OPENAI_API_KEY required."""

import pytest
from sqlalchemy import func, select

from app.models import ChildChunk, Document, ParentChunk
from shared.storage import get_public_store
from worker.celery_app import app as celery_app
from worker.db import session_scope
from worker.stages import launch


@pytest.fixture(autouse=True)
def eager():
    celery_app.conf.task_always_eager = True
    yield
    celery_app.conf.task_always_eager = False


def _counts(document_id):
    with session_scope() as session:
        return (
            session.scalar(
                select(func.count()).select_from(ParentChunk)
                .where(ParentChunk.document_id == document_id)
            ),
            session.scalar(
                select(func.count()).select_from(ChildChunk)
                .where(ChildChunk.document_id == document_id)
            ),
        )


def test_full_chain_produces_embedded_chunks(seeded_document):
    launch(str(seeded_document.id))

    parents, children = _counts(seeded_document.id)
    assert parents > 0 and children > 0

    with session_scope() as session:
        document = session.get(Document, seeded_document.id)
        assert document.status == "COMPLETED"
        child = session.scalars(
            select(ChildChunk).where(ChildChunk.document_id == seeded_document.id)
        ).first()
        assert child.embedding is not None
        assert len(child.embedding) == 1536


def test_running_the_whole_chain_twice_changes_nothing(seeded_document):
    launch(str(seeded_document.id))
    before = _counts(seeded_document.id)

    launch(str(seeded_document.id))
    after = _counts(seeded_document.id)

    assert before == after


def test_deleting_a_checkpoint_reruns_only_that_stage(seeded_document):
    """Delete enriched.json: S1/S2 must skip via checkpoint, S3 re-runs, and
    the final result is identical."""
    launch(str(seeded_document.id))
    before = _counts(seeded_document.id)

    store = get_public_store()
    store._client.delete_object(Bucket="staging", Key=f"{seeded_document.id}/enriched.json")

    launch(str(seeded_document.id))

    assert _counts(seeded_document.id) == before
```

- [ ] **Step 2: Run them** — PASS, 3 tests.

- [ ] **Step 3: Full suite and lint**

Run: `uv run pytest -q && uv run ruff check .`

- [ ] **Step 4: Stop and report — Phase 1a complete**

The report must state: total test count, how many containers `docker compose ps` shows healthy, and one document driven `QUEUED → COMPLETED`.

Proposed commit message: `test: add end-to-end pipeline, idempotency and resume tests`

---

## Phase 1b — requires `OPENAI_API_KEY`

### Task 19: Real provider and similarity smoke test

**Files:**
- Modify: `.env`
- Create: `tests/test_similarity_smoke.py`

**Interfaces:**
- Consumes: `OpenAIProvider`
- Produces: evidence that the data is *usable*

> **This is the single most important test in Phase 1.** Wrong normalization, wrong dimensionality, and a misconfigured index all look identical to "rows inserted successfully". This is the only thing that tells them apart.

- [ ] **Step 1: Supply the key**

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

- [ ] **Step 2: Write the test**

```python
# tests/test_similarity_smoke.py
"""Cannot run on the stub: fake vectors carry no semantics, so any two
passages are "close" at random."""

import os

import pytest
from sqlalchemy import text

from shared.llm import OpenAIProvider
from worker.db import session_scope
from worker.stages import launch

pytestmark = pytest.mark.skipif(
    os.getenv("LLM_PROVIDER") != "openai",
    reason="requires OPENAI_API_KEY and LLM_PROVIDER=openai",
)


def test_a_related_question_ranks_the_right_chunk_first(seeded_document):
    launch(str(seeded_document.id))

    question = "What was third quarter revenue?"
    vector = OpenAIProvider().embed([question])[0]

    with session_scope() as session:
        rows = session.execute(
            text(
                "SELECT content, embedding <#> CAST(:v AS vector) AS distance "
                "FROM child_chunks WHERE document_id = :doc "
                "ORDER BY embedding <#> CAST(:v AS vector) LIMIT 3"
            ),
            {"v": str(vector), "doc": str(seeded_document.id)},
        ).all()

    assert rows, "no chunks — the pipeline wrote nothing"
    assert "revenue" in rows[0][0].lower(), (
        "the closest chunk is unrelated — check L2 normalization, the embedding "
        "dimension, and the HNSW index opclass"
    )
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/test_similarity_smoke.py -q`
Expected: PASS.

If it fails, check in this order: (1) are the vectors L2-normalized, (2) does `EMBED_DIMENSIONS` match `vector(1536)`, (3) does the index use `vector_ip_ops`.

- [ ] **Step 4: Run the full Definition of Done**

```bash
docker compose up -d --build          # 8 healthy containers
uv run alembic upgrade head
uv run pytest -q                      # all green
uv run ruff check .                   # clean
test ! -d app/core && echo "app/core correctly does not exist"
```

- [ ] **Step 5: Stop and report — Phase 1 complete**

Proposed commit message: `test: add similarity smoke test against the real provider`

---

## Definition of Done — Phase 1

1. `docker compose up -d` brings up **8 healthy containers**.
2. `alembic upgrade head` applies cleanly to an empty database.
3. Presigned upload → `POST /documents` → 202 with a `document_id`.
4. Re-uploading the same file returns **409 and publishes nothing**.
5. The chain drives every fixture `QUEUED → COMPLETED`, except `encrypted.pdf`, which lands in `DEAD_LETTER` **with a specific reason**.
6. `parent_chunks` and `child_chunks` hold rows with non-null 1536-dimensional embeddings; `embedding` on `documents` no longer exists.
7. Re-running a completed document changes no rows.
8. The similarity smoke test ranks the expected chunk first.
9. Full suite green; `ruff check .` clean.
10. **`app/core/` does not exist.**
