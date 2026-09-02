# Phase 1 — Đường ghi (ingestion) · Kế hoạch thực thi

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload một file PDF và biến nó thành các chunk đã nhúng, tìm kiếm được, nằm trong `parent_chunks` và `child_chunks`.

**Architecture:** Client PUT thẳng PDF vào MinIO bằng presigned URL; API chỉ đối chiếu hash rồi publish sang RabbitMQ. Năm Celery stage (S1→S5) chạy nối tiếp, mỗi stage ghi một checkpoint bền vào MinIO trước khi chuyển tiếp, nên crash chỉ tốn một lần chạy lại của đúng một stage. S1 gọi sang container `docling` qua HTTP; S3/S4 rút token từ một shared bucket trên Redis trước mỗi lần gọi OpenAI.

**Tech Stack:** FastAPI (async) · Celery + RabbitMQ · SQLAlchemy 2 (async cho API, **sync** cho worker) · Postgres 16 + pgvector · MinIO (S3 API qua boto3) · Redis · PyMuPDF + Docling · OpenAI `gpt-4o-mini` + `text-embedding-3-small` · pytest · uv · ruff

**Spec:** [`docs/system-design.md`](../system-design.md) — bản thiết kế chốt. Khi plan này và spec bất đồng, **spec đúng**. Hai tài liệu `docs/ingestion-architecture.md` và `docs/ingestion-plan.md` là bản cũ, đã bị thay ở nhiều chỗ; đừng làm theo.

---

## Global Constraints

Mọi task đều ngầm chịu các ràng buộc dưới đây. Giá trị chép nguyên từ spec.

- **Python** `>=3.12`; quản lý phụ thuộc bằng `uv`; thêm gói bằng `uv add` / `uv add --dev`, không sửa tay `pyproject.toml`.
- **Branch:** toàn bộ Phase 1 làm trên branch `phase-1`.
- **Git:** KHÔNG chạy `git commit`, `git push`, `git merge`. Kết thúc mỗi task thì dừng lại, báo cáo, chờ lệnh.
- **Suite phải xanh ở cuối MỌI task.** `uv run pytest -q` và `uv run ruff check .` đều sạch. Pre-commit hook sẽ chặn nếu không.
- **Embedding:** `text-embedding-3-small`, `dimensions=1536`, **L2-normalize mọi vector** — không tuỳ chọn.
- **Chat:** `gpt-4o-mini`.
- **Chunk:** parent 500–1000 token, child 100–200 token, **drop chunk dưới 20 token**. Tokenizer `cl100k_base` qua `tiktoken`.
- **Batch:** enrich 20 child chunk mỗi request, embed 100 text mỗi request.
- **Docling:** per-page timeout `DOCLING_PAGE_TIMEOUT_S=90` khi có GPU; **300** khi chạy CPU-only. Concurrency 1.
- **Phát hiện bản scan:** `total_text_chars / page_count < 100` → định tuyến **toàn bộ** tài liệu sang Docling.
- **Giới hạn:** `MAX_FILE_SIZE_MB=50`, `MAX_PAGE_COUNT=500`, `MAX_CHUNKS_PER_DOC=5000`.
- **Category — enum đóng, 8 giá trị:** `FINANCIAL` `LEGAL` `TECHNICAL` `MARKETING` `HR` `RESEARCH` `OPERATIONS` `OTHER`. Giá trị ngoài enum bị **từ chối và ghi log**, không lưu.
- **Queue:** đúng hai — `cpu` (S1, S2, S5) và `llm` (S3, S4).
- **Celery:** `result_backend=None`, `task_acks_late=True`, `task_reject_on_worker_lost=True`, `worker_prefetch_multiplier=1`.
- **DB driver:** API `postgresql+asyncpg`, worker `postgresql+psycopg` (**sync**). Task worker là `def` thường — **không `asyncio.run()` ở bất kỳ đâu trong `worker/`**.
- **MinIO:** bucket `raw` (giữ vĩnh viễn) và `staging` (lifecycle 7 ngày).
- **`app/core/` KHÔNG được tồn tại.** Đó là Phase 2. Không thư mục rỗng, không router placeholder.
- **ruff:** `line-length = 100`, ruleset đã cấu hình trong `pyproject.toml`.

---

## Phân chia 1a / 1b

| | Task | Cần `OPENAI_API_KEY`? |
| --- | --- | --- |
| **Phase 1a** | 1 → 18 | **Không.** Chạy hết bằng `StubProvider` |
| **Phase 1b** | 19 | **Có** |

Phase 1a kết thúc khi: chain chạy trọn vẹn với stub, `parent_chunks`/`child_chunks` có row, chạy lại không đổi gì. Phase 1b đổi sang provider thật và chạy similarity smoke test — bằng chứng duy nhất rằng dữ liệu *dùng được* chứ không chỉ đúng định dạng.

---

## Cấu trúc file

```
app/
  config.py                    SỬA — thêm ~20 setting mới
  models/
    document.py                THAY HẲN — từ title/content thành thực thể file
    parent_chunk.py            MỚI
    child_chunk.py             MỚI
  controllers/
    document_controller.py     XOÁ (Task 7)
    ingestion_controller.py    MỚI — 4 endpoint
  services/
    document_service.py        XOÁ (Task 7)
    ingestion_service.py       MỚI — dedup + publish
  repositories/
    document_repository.py     XOÁ rồi VIẾT LẠI cho shape mới
  schemas/
    document.py                THAY HẲN
shared/                        MỚI — anh em của app/, worker KHÔNG được import app/
  storage.py                   client MinIO/S3
  rate_limiter.py              token bucket Redis (Lua)
  llm.py                       Protocol + StubProvider + OpenAIProvider + factory
worker/                        MỚI — Celery, sync
  celery_app.py                cấu hình + task_routes
  db.py                        engine sync
  stages.py                    5 @task, chain
  parsing.py                   S1
  chunking.py                  S2
  enrichment.py                S3
  embedding.py                 S4
  persistence.py               S5
docling_service/               MỚI — container riêng
  main.py                      FastAPI: POST /parse
  Dockerfile
  pyproject.toml
tests/
  fixtures/generate.py         MỚI — sinh 6 PDF
  fixtures/*.pdf               MỚI — sinh ra, có commit
alembic/versions/              MỚI — 1 revision
docker-compose.yml             SỬA — 2 service thành 8
.env.example                   SỬA — thêm nhóm biến mới
```

**`shared/` nằm ngang hàng `app/`, không nằm trong.** Trong `worker/`, mọi dòng bắt đầu bằng `from app.` — trừ `from app.models` — đều là lỗi. Đặt như vậy để vi phạm ranh giới lộ ra ngay trên dòng import chứ không phụ thuộc vào việc ai đó nhớ luật.

---

## Phase 1a

### Task 1: Fixture corpus

**Files:**
- Create: `tests/fixtures/generate.py`
- Create: `tests/test_fixtures.py`
- Sinh ra: `tests/fixtures/{clean_text,tables,multi_column,scanned,encrypted,malformed}.pdf`

**Interfaces:**
- Consumes: không
- Produces: `tests/fixtures/generate.py::main() -> None` sinh 6 file. Các task sau đọc file theo đường dẫn `tests/fixtures/<tên>.pdf`.

- [ ] **Step 1: Thêm dev dependency**

```bash
uv add --dev reportlab pypdf
uv add pymupdf
```

- [ ] **Step 2: Viết test đỏ**

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
    assert chars / doc.page_count >= 100, "clean_text phải vượt ngưỡng phát hiện bản scan"


def test_scanned_has_no_text_layer():
    """Đây là fixture ép nhánh 'định tuyến toàn bộ tài liệu sang OCR'."""
    doc = fitz.open(FIXTURES / "scanned.pdf")
    chars = sum(len(page.get_text()) for page in doc)
    assert chars / doc.page_count < 100


def test_encrypted_is_actually_encrypted():
    assert PdfReader(FIXTURES / "encrypted.pdf").is_encrypted


def test_malformed_cannot_be_opened():
    with pytest.raises(Exception):
        fitz.open(FIXTURES / "malformed.pdf")
```

- [ ] **Step 3: Chạy để xác nhận đỏ**

Run: `uv run pytest tests/test_fixtures.py -q`
Expected: FAIL — thư mục `tests/fixtures/` chưa có file nào.

- [ ] **Step 4: Viết generator**

```python
# tests/fixtures/generate.py
"""Sinh 6 PDF mẫu. Chạy: uv run python tests/fixtures/generate.py

Sinh bằng script chứ không dùng PDF thật: tái lập được, nhỏ, không dính
bản quyền, và thêm case mới thì sửa script chứ không đi xin file.
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
    "Doanh thu quy 3 nam 2024 dat 41.7 ty dong, tang 12 phan tram so voi cung ky. "
    "Chi phi van hanh giu nguyen o muc 8.2 ty dong. "
)


def _simple(path: Path, flowables, page_size=A4):
    doc = BaseDocTemplate(str(path), pagesize=page_size)
    frame = Frame(50, 50, page_size[0] - 100, page_size[1] - 100, id="f")
    doc.addPageTemplates([PageTemplate(id="t", frames=[frame])])
    doc.build(flowables)


def clean_text():
    _simple(HERE / "clean_text.pdf", [Paragraph(BODY * 6, STYLES["Normal"]) for _ in range(3)])


def tables():
    data = [["Quy", "Doanh thu", "Chi phi"], ["Q1", "38.1", "8.0"], ["Q2", "39.4", "8.1"]]
    _simple(HERE / "tables.pdf", [Paragraph(BODY, STYLES["Normal"]), Table(data)])


def multi_column():
    path = HERE / "multi_column.pdf"
    doc = BaseDocTemplate(str(path), pagesize=A4)
    left = Frame(50, 50, 230, 700, id="l")
    right = Frame(300, 50, 230, 700, id="r")
    doc.addPageTemplates([PageTemplate(id="two", frames=[left, right])])
    doc.build([Paragraph(BODY * 4, STYLES["Normal"]) for _ in range(4)])


def scanned():
    """Render clean_text thành ảnh rồi bọc lại — mất hoàn toàn text layer."""
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
    """Cắt cụt còn 60% số byte — header còn nguyên, phần thân hỏng."""
    data = (HERE / "clean_text.pdf").read_bytes()
    (HERE / "malformed.pdf").write_bytes(data[: int(len(data) * 0.6)])


def main() -> None:
    clean_text()
    tables()
    multi_column()
    scanned()
    encrypted()
    malformed()
    print("6 fixture đã sinh tại", HERE)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Sinh file và chạy lại test**

Run:
```bash
uv run python tests/fixtures/generate.py
uv run pytest tests/test_fixtures.py -q
```
Expected: PASS, 5 test.

- [ ] **Step 6: Dừng và báo cáo**

Báo: 6 fixture đã sinh, suite 60 → 65 test, xanh. Commit message đề xuất khi được lệnh:
`test: add generated PDF fixture corpus`

---

### Task 2: Config cho toàn bộ Phase 1

**Files:**
- Modify: `app/config.py`
- Modify: `.env.example`, `.env`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: `app.config.Settings` hiện có (5 biến `DB_*` + override `DATABASE_URL`)
- Produces: `Settings.rabbitmq_url`, `.redis_url`, `.minio_endpoint`, `.minio_public_endpoint`, `.minio_access_key`, `.minio_secret_key`, `.minio_bucket_raw`, `.minio_bucket_staging`, `.openai_api_key`, `.openai_chat_model`, `.openai_embed_model`, `.embed_dimensions`, `.docling_url`, `.docling_page_timeout_s`, `.llm_provider`, `.max_file_size_mb`, `.max_page_count`, `.max_chunks_per_doc`, `.enrich_batch_size`, `.embed_batch_size`, `.rl_chat_rpm`, `.rl_chat_tpm`, `.rl_embed_rpm`, `.rl_embed_tpm`, và property `.worker_database_url -> str`

- [ ] **Step 1: Viết test đỏ**

```python
# thêm vào tests/test_config.py
def test_worker_database_url_uses_the_sync_driver(db_env):
    """Worker dùng psycopg (sync); API dùng asyncpg. Hai engine, một bộ model."""
    settings = Settings(_env_file=None)

    assert settings.worker_database_url.startswith("postgresql+psycopg://")
    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.worker_database_url.endswith(settings.database_url.split("/")[-1])


def test_llm_provider_defaults_to_stub(db_env):
    """Mặc định phải là stub: quên cắm key thì test chạy được, không phải nổ."""
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "stub"


def test_openai_key_is_not_required_when_provider_is_stub(db_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None


def test_minio_public_url_falls_back_to_the_internal_endpoint(db_env, monkeypatch):
    """Presigned URL ký cho một host cụ thể — URL ký bằng tên nội bộ thì
    trình duyệt không dùng được. Hai endpoint tách nhau vì lý do đó."""
    monkeypatch.setenv("MINIO_ENDPOINT", "http://minio:9000")
    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    settings = Settings(_env_file=None)
    assert settings.minio_public_url == "http://minio:9000"
```

- [ ] **Step 2: Chạy để xác nhận đỏ**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'worker_database_url'`

- [ ] **Step 3: Thêm setting**

```python
# app/config.py — thêm vào class Settings, sau các trường DB_*

    # --- Hạ tầng ---
    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672//"
    redis_url: str = "redis://redis:6379/0"

    # Endpoint nội bộ (container → container) và endpoint công khai (trình
    # duyệt → MinIO). Presigned URL được ký cho đúng một host, nên URL ký
    # bằng "minio:9000" là vô dụng với client bên ngoài.
    minio_endpoint: str = "http://minio:9000"
    minio_public_endpoint: str | None = None
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_raw: str = "raw"
    minio_bucket_staging: str = "staging"

    docling_url: str = "http://docling:8100"
    docling_page_timeout_s: int = 90

    # --- LLM ---
    # "stub" là mặc định có chủ ý: thiếu key thì suite vẫn chạy được.
    llm_provider: str = "stub"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    embed_dimensions: int = 1536

    # --- Giới hạn ---
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
        """Cùng một database, driver đồng bộ. Worker không có event loop."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")

    @property
    def minio_public_url(self) -> str:
        return self.minio_public_endpoint or self.minio_endpoint
```

> Trường lưu là `minio_public_endpoint` (đọc từ env `MINIO_PUBLIC_ENDPOINT`, cho phép `None`); thứ mọi nơi khác dùng là property `minio_public_url` đã có fallback. Đừng đọc thẳng trường — nó có thể là `None`.

- [ ] **Step 4: Chạy test**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Cập nhật `.env.example` và `.env`**

Thêm vào cuối `.env.example` (rồi `cp` các dòng tương ứng sang `.env`):

```bash
# --- Hạ tầng ---
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
# stub = không gọi mạng, kết quả tất định. Đổi thành openai khi có key.
LLM_PROVIDER=stub
# OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
EMBED_DIMENSIONS=1536

# --- Giới hạn ---
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

- [ ] **Step 6: Suite đầy đủ + lint, rồi dừng**

Run: `uv run pytest -q && uv run ruff check .`
Expected: PASS, 69 test.

Commit message đề xuất: `feat(config): add infrastructure, LLM and limit settings`

---

### Task 3: Compose — rabbitmq, redis, minio

**Files:**
- Modify: `docker-compose.yml`
- Create: `tests/test_infrastructure.py`

**Interfaces:**
- Consumes: `Settings.rabbitmq_url`, `.redis_url`, `.minio_endpoint`
- Produces: ba service chạy được, dùng cho Task 4–6

- [ ] **Step 1: Thêm dependency**

```bash
uv add boto3 redis kombu
```

- [ ] **Step 2: Viết test đỏ**

```python
# tests/test_infrastructure.py
"""Ba container hạ tầng phải chạm được từ host trước khi viết client cho chúng."""

import boto3
import pytest
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

- [ ] **Step 3: Chạy để xác nhận đỏ**

Run: `uv run pytest tests/test_infrastructure.py -q`
Expected: FAIL — connection refused, chưa có service nào.

- [ ] **Step 4: Thêm service vào compose**

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

  # Tạo bucket rồi thoát. Không dùng entrypoint dài hạn — bucket là việc
  # một lần, và gắn nó vào lifecycle của minio làm minio khởi động chậm hơn.
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

Thêm `miniodata:` vào khối `volumes:` cuối file.

- [ ] **Step 5: Bật lên và chạy test**

Run:
```bash
docker compose up -d rabbitmq redis minio minio-init
uv run pytest tests/test_infrastructure.py -q
```
Expected: PASS, 3 test.

- [ ] **Step 6: Suite đầy đủ, rồi dừng**

Run: `uv run pytest -q && uv run ruff check .`
Expected: 72 test xanh.

Commit message đề xuất: `feat(infra): add rabbitmq, redis and minio services`

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
  - `get_store() -> ObjectStore`

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_storage.py
import json
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
    """URL này đi tới trình duyệt, nên phải mang host công khai chứ không
    phải tên service nội bộ — nếu không client bên ngoài không resolve được."""
    url = store.presigned_put(key)
    assert urlparse(url).netloc == urlparse(settings.minio_public_url).netloc
    assert "X-Amz-Signature" in url
```

- [ ] **Step 2: Chạy để xác nhận đỏ**

Run: `uv run pytest tests/test_storage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared'`

- [ ] **Step 3: Viết implementation**

```python
# shared/storage.py
"""Client object storage dùng chung cho cả API và worker.

Dùng boto3 chứ không phải SDK riêng của MinIO: MinIO nói S3 API, nên đổi
sang S3 thật sau này chỉ là đổi endpoint, không phải viết lại client.
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
        """'raw/abc.pdf' -> ('raw', 'abc.pdf'). Bucket nằm trong key để
        mọi tầng trên chỉ phải nhớ một chuỗi."""
        bucket, _, rest = key.partition("/")
        if bucket not in (self._bucket_raw, self._bucket_staging):
            raise ValueError(f"bucket không hợp lệ trong key: {key!r}")
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
    """Store nội bộ, dùng trong container."""
    return ObjectStore()


@lru_cache
def get_public_store() -> ObjectStore:
    """Chỉ để ký presigned URL cho client bên ngoài."""
    return ObjectStore(endpoint=get_settings().minio_public_url)
```

- [ ] **Step 4: Chạy test**

Run: `uv run pytest tests/test_storage.py -q`
Expected: PASS, 5 test.

- [ ] **Step 5: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(shared): add S3-compatible object store client`

---

### Task 5: `shared/rate_limiter.py`

**Files:**
- Create: `shared/rate_limiter.py`
- Create: `tests/test_rate_limiter.py`

**Interfaces:**
- Consumes: `Settings.redis_url`, `.rl_*`
- Produces: `TokenBucket(key: str, capacity: int, refill_per_sec: float)`, phương thức `acquire(tokens: int = 1) -> tuple[bool, int]` trả `(allowed, wait_ms)`; và `get_bucket(name: Literal["chat_rpm","chat_tpm","embed_rpm","embed_tpm"]) -> TokenBucket`

- [ ] **Step 1: Viết test đỏ**

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
    assert 0 < wait_ms <= 1000, "wait phải là thời gian tới khi có đủ token"


def test_acquiring_more_than_one_token_at_a_time(bucket):
    assert bucket.acquire(tokens=3)[0] is True
    assert bucket.acquire(tokens=1)[0] is False


def test_a_request_larger_than_capacity_is_refused_not_hung(bucket):
    """Một request đòi nhiều hơn sức chứa sẽ không bao giờ thoả được —
    phải từ chối dứt khoát chứ không trả về wait vô hạn."""
    allowed, wait_ms = bucket.acquire(tokens=99)
    assert allowed is False
    assert wait_ms == -1


def test_two_buckets_do_not_share_state():
    a = TokenBucket(key=f"a:{uuid.uuid4()}", capacity=1, refill_per_sec=1.0)
    b = TokenBucket(key=f"b:{uuid.uuid4()}", capacity=1, refill_per_sec=1.0)
    a.acquire()
    assert b.acquire()[0] is True
```

- [ ] **Step 2: Chạy để xác nhận đỏ**

Run: `uv run pytest tests/test_rate_limiter.py -q`
Expected: FAIL — module chưa tồn tại.

- [ ] **Step 3: Viết implementation**

```python
# shared/rate_limiter.py
"""Token bucket dùng chung, đặt trên Redis.

Vì sao ở Redis chứ không đếm trong worker: hạn ngạch là tổng của cả hệ
thống. 20 worker mỗi cái tự đếm "tôi mới gọi 10 lần" thì tổng đã 200 mà
không ai biết.

Vì sao Lua: "kiểm tra còn token" và "trừ token" phải không thể chen ngang.
Tách làm hai lệnh thì hai worker cùng đọc thấy còn 1 token rồi cùng trừ.
"""

from functools import lru_cache

import redis as redis_lib

from app.config import get_settings

# KEYS[1] = khoá bucket; ARGV = capacity, refill_per_sec, tokens, now_ms
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
        url = settings.redis_url
        self._redis = redis_lib.from_url(url)
        self._script = self._redis.register_script(_SCRIPT)
        self.key = key
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec

    def acquire(self, tokens: int = 1) -> tuple[bool, int]:
        """Trả (allowed, wait_ms). wait_ms == -1 nghĩa là không bao giờ
        thoả được vì request lớn hơn sức chứa."""
        now_ms = int(self._redis.time()[0] * 1000 + self._redis.time()[1] / 1000)
        allowed, wait_ms = self._script(
            keys=[self.key],
            args=[self.capacity, self.refill_per_sec, tokens, now_ms],
        )
        return bool(allowed), int(wait_ms)


@lru_cache
def get_bucket(name: str) -> TokenBucket:
    """Bốn bucket, vì request và token là hai hạn ngạch riêng, và chat với
    embedding cũng là hai hạn ngạch riêng. Nhốt chung thì một cái bỏ đói
    cái kia dù nhà cung cấp không hề giới hạn như vậy."""
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

- [ ] **Step 4: Chạy test**

Run: `uv run pytest tests/test_rate_limiter.py -q`
Expected: PASS, 5 test.

> Nếu `test_refuses_once_exhausted_and_reports_a_wait` đỏ vì `wait_ms == 0`: kiểm tra `redis.time()` trả về `(giây, micro-giây)` và phép tính `now_ms` cộng đúng hai thành phần.

- [ ] **Step 5: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(shared): add Redis token bucket rate limiter`

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

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_llm_provider.py
"""Provider có hai bản. Stub tồn tại KHÔNG phải để tiết kiệm tiền — chi phí
thật chỉ vài xu — mà vì tiêu chí nghiệm thu của S3 là "trả về 19 trên 20 thì
phải retry đúng id thiếu", và không cách nào bắt API thật làm vậy theo yêu cầu.
"""

import pytest

from shared.llm import CATEGORIES, StubProvider, get_provider


@pytest.fixture
def chunks():
    return [{"id": i, "content": f"đoạn văn số {i}"} for i in range(5)]


def test_stub_returns_one_result_per_chunk(chunks):
    assert len(StubProvider().enrich(chunks)) == len(chunks)


def test_stub_is_deterministic(chunks):
    assert StubProvider().enrich(chunks) == StubProvider().enrich(chunks)


def test_stub_only_emits_categories_from_the_closed_enum(chunks):
    assert all(c.category in CATEGORIES for c in StubProvider().enrich(chunks))


def test_stub_can_be_told_to_drop_ids(chunks):
    """Đây là lý do stub tồn tại: dựng đúng kịch bản hỏng mà S3 phải chịu."""
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
        assert abs(norm - 1.0) < 1e-6, "stub cũng phải chuẩn hoá — nếu không nó che mất bug thật"


def test_get_provider_returns_stub_by_default(db_env):
    assert isinstance(get_provider(), StubProvider)
```

- [ ] **Step 2: Chạy để xác nhận đỏ**

Run: `uv run pytest tests/test_llm_provider.py -q`
Expected: FAIL — module chưa tồn tại.

- [ ] **Step 3: Viết implementation**

```python
# shared/llm.py
"""Đường nối provider: một Protocol, hai implementation.

Chọn implementation theo hai cách, cố ý:
  - env var LLM_PROVIDER — cho lúc chạy tay `docker compose up`
  - truyền thẳng vào hàm (dependency injection) — cho test, vì env var chỉ
    bật/tắt được stub chứ không điều khiển được stub TRẢ VỀ GÌ.
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
    """Vector giả tất định, đã chuẩn hoá L2.

    Chuẩn hoá cả ở stub là có chủ ý: nếu stub trả vector chưa chuẩn hoá thì
    assert chuẩn hoá ở S4 sẽ đỏ khi chạy bằng stub, và người ta sẽ gỡ assert
    đó đi — đúng cái assert phải sống sót tới production.
    """
    digest = hashlib.sha256(text.encode()).digest()
    raw = [(digest[i % len(digest)] - 128) / 128 for i in range(dimensions)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


class StubProvider:
    """Không gọi mạng. Có thể ra lệnh cho nó hỏng theo đúng cách mình cần."""

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
                EnrichedChunk(id=cid, context=f"Đoạn này nói về nội dung số {cid}.", category=category)
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
            raise RuntimeError("LLM_PROVIDER=openai nhưng OPENAI_API_KEY chưa được đặt")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._chat_model = settings.openai_chat_model
        self._embed_model = settings.openai_embed_model
        self._dimensions = settings.embed_dimensions

    def enrich(self, chunks: list[dict]) -> list[EnrichedChunk]:
        import json

        prompt = (
            "Với mỗi chunk, viết MỘT câu ngữ cảnh và chọn đúng một category trong: "
            + ", ".join(sorted(CATEGORIES))
            + ". Trả về JSON {\"chunks\":[{\"id\":int,\"context\":str,\"category\":str}]}."
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

- [ ] **Step 4: Thêm dependency và chạy test**

Run:
```bash
uv add openai
uv run pytest tests/test_llm_provider.py -q
```
Expected: PASS, 7 test.

- [ ] **Step 5: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(shared): add LLM provider protocol with stub and OpenAI implementations`

---

### Task 7: Gỡ bề mặt CRUD demo

**Files:**
- Delete: `app/controllers/document_controller.py`, `app/services/document_service.py`, `app/repositories/document_repository.py`, `app/schemas/document.py`
- Delete: `tests/test_documents.py`, `tests/test_document_service.py`, `tests/test_document_repository.py`
- Modify: `app/controllers/__init__.py`, `tests/test_transaction_boundary.py`

**Interfaces:**
- Consumes: không
- Produces: không. Task này chỉ **gỡ**.

> **Vì sao đây là một task riêng.** `Document` hiện nghĩa là "một cặp title + content"; sau Task 8 nó nghĩa là "một file PDF đã upload". 28 test dưới đây không hỏng — chúng mô tả đúng một khái niệm sắp thôi tồn tại. Tách "gỡ cái cũ" khỏi "dựng cái mới" thành hai commit làm diff đọc được, và giữ suite **xanh** ở cả hai — điều kiện để TDD có nghĩa ở các task sau.

- [ ] **Step 1: Gỡ router khỏi đăng ký**

```python
# app/controllers/__init__.py
from fastapi import FastAPI

from app.controllers import health_controller

__all__ = ["register_controllers"]


def register_controllers(app: FastAPI) -> None:
    app.include_router(health_controller.router)
```

- [ ] **Step 2: Xoá 4 module và 3 file test**

```bash
rm app/controllers/document_controller.py
rm app/services/document_service.py
rm app/repositories/document_repository.py
rm app/schemas/document.py
rm tests/test_documents.py tests/test_document_service.py tests/test_document_repository.py
```

- [ ] **Step 3: Gỡ `test_transaction_boundary.py` khỏi repository**

Ba test này kiểm chứng hợp đồng của `get_session` (commit khi thành công, rollback khi có exception). Hợp đồng đó **không liên quan gì** tới repository — dùng `DocumentRepository` chỉ là tiện tay. Thay bằng thao tác trực tiếp trên model:

```python
# tests/test_transaction_boundary.py — thay 3 chỗ gọi DocumentRepository
# XOÁ: from app.repositories.document_repository import DocumentRepository
from app.models.document import Document

# ... trong mỗi test, thay
#   await DocumentRepository(session).create(title=title, content="x")
# bằng
        session.add(Document(title=title, content="x"))
        await session.flush()
```

- [ ] **Step 4: Sửa một assert trong `test_exception_handlers.py`**

Test `test_app_exception_renders_the_envelope` khẳng định message của `DOCUMENT_TITLE_EXISTS`. `title` sắp hết unique nên ErrorCode đó sẽ đổi ở Task 8 — nhưng **chưa đổi ở task này**, nên để nguyên. Chỉ ghi chú lại để Task 8 nhớ.

- [ ] **Step 5: Chạy suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: **PASS, 51 test** (79 − 28). Xanh, không đỏ chỗ nào.

- [ ] **Step 6: Dừng và báo cáo**

Báo rõ: xoá 28 test, sửa 3 test, suite còn 51 và xanh.

Commit message đề xuất: `refactor!: remove the demo document CRUD surface`

---

### Task 8: Schema mới — models + migration

**Files:**
- Rewrite: `app/models/document.py`
- Create: `app/models/parent_chunk.py`, `app/models/child_chunk.py`
- Modify: `app/models/__init__.py`, `app/exceptions/error_codes.py`
- Create: `alembic/versions/<hash>_phase1_schema.py` (sinh bằng CLI)
- Create: `tests/test_schema.py`
- Modify: `tests/test_transaction_boundary.py`, `tests/test_exception_handlers.py`

**Interfaces:**
- Consumes: `app.models.base.Base`
- Produces:
  - `Document` với `id: uuid.UUID`, `sha256_hash: str`, `filename: str`, `object_key: str`, `size_bytes: int`, `page_count: int | None`, `status: str`, `stage: str | None`, `attempts: int`, `failed_stage: str | None`, `last_error: str | None`, `created_at`, `updated_at`, `completed_at`
  - `ParentChunk` với `id`, `document_id`, `chunk_index`, `content`, `token_count`, `page_start`, `page_end`
  - `ChildChunk` với `id`, `document_id`, `parent_id`, `chunk_index`, `content`, `contextualized`, `page_number`, `token_count`, `embedding`, `category`
  - `ErrorCode.DOCUMENT_ALREADY_INGESTED = (409, "Document already ingested")`, `ErrorCode.PDF_ENCRYPTED`, `ErrorCode.PDF_TOO_LARGE`

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_schema.py
"""Schema là hợp đồng mà cả API và worker cùng dựa vào. Kiểm ở mức DB thật
chứ không chỉ ở mức model — index và constraint không tồn tại trong Python."""

import uuid

import pytest
from sqlalchemy import text

from app.models import ChildChunk, Document, ParentChunk

pytestmark = pytest.mark.asyncio


async def test_document_is_a_file_entity_not_a_title_content_pair(db_session):
    doc = Document(
        sha256_hash="a" * 64,
        filename="bao-cao.pdf",
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
    """page_number nằm ở child, không ở parent — đó là thứ quyết định độ
    chính xác của trích dẫn ở Phase 2."""
    doc = Document(sha256_hash="c" * 64, filename="x.pdf", object_key="raw/x.pdf", size_bytes=1)
    db_session.add(doc)
    await db_session.flush()

    parent = ParentChunk(
        document_id=doc.id, chunk_index=0, content="cha", token_count=600, page_start=1, page_end=2
    )
    db_session.add(parent)
    await db_session.flush()

    child = ChildChunk(
        document_id=doc.id,
        parent_id=parent.id,
        chunk_index=0,
        content="con",
        contextualized="ngữ cảnh\n\ncon",
        page_number=2,
        token_count=120,
        embedding=[0.0] * 1536,
    )
    db_session.add(child)
    await db_session.flush()

    assert child.page_number == 2


async def test_hnsw_index_uses_inner_product(db_session):
    """vector_ip_ops giả định vector đơn vị. Nếu index bị tạo bằng
    vector_cosine_ops thì L2-normalize ở S4 trở thành công cốc."""
    result = await db_session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'child_chunks'")
    )
    defs = " ".join(row[0] for row in result)
    assert "hnsw" in defs.lower()
    assert "vector_ip_ops" in defs
```

- [ ] **Step 2: Chạy để xác nhận đỏ**

Run: `uv run pytest tests/test_schema.py -q`
Expected: FAIL — `ImportError: cannot import name 'ChildChunk'`

- [ ] **Step 3: Viết model**

```python
# app/models/document.py — THAY TOÀN BỘ FILE
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Document(Base):
    """Một file PDF đã upload. Text sống trong chunk, không ở đây."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Unique ở đây là thứ đóng cuộc đua dedup: INSERT ... ON CONFLICT
    # DO NOTHING RETURNING id, publish chỉ khi có row trả về.
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(Text, nullable=False, default="QUEUED")
    # Con trỏ resume: stage VỪA HOÀN THÀNH, không phải stage đang chạy.
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
    """Đơn vị ngữ cảnh để sinh câu trả lời. Không được nhúng."""

    __tablename__ = "parent_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Tất định từ output của parse — đây là khoá tự nhiên làm S5 idempotent.
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
    """Đơn vị truy hồi. Cái được nhúng là `contextualized`, không phải `content`."""

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
    # Độ chính xác của trích dẫn nằm ở ĐÂY, không ở parent.
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

- [ ] **Step 4: Sinh migration bằng CLI và thêm index HNSW bằng tay**

Run: `uv run alembic revision --autogenerate -m "phase 1 schema"`

Autogenerate **không** suy ra được index HNSW. Thêm vào cuối `upgrade()` của file vừa sinh:

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

và vào đầu `downgrade()`:

```python
    op.execute("DROP INDEX IF EXISTS ix_child_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_documents_active_status")
```

- [ ] **Step 5: Cập nhật ErrorCode**

```python
# app/exceptions/error_codes.py — thay DOCUMENT_TITLE_EXISTS
    DOCUMENT_NOT_FOUND = (404, "Document not found")
    DOCUMENT_ALREADY_INGESTED = (409, "Document already ingested")
    PDF_ENCRYPTED = (422, "PDF is encrypted and cannot be parsed")
    PDF_TOO_LARGE = (413, "PDF exceeds the size or page limit")
    HASH_MISMATCH = (400, "Uploaded object does not match the supplied hash")
    VALIDATION_FAILED = (422, "Validation failed")
    INTERNAL_ERROR = (500, "Internal server error")
    DATABASE_ERROR = (500, "Database operation failed")
```

Sửa assert tương ứng trong `tests/test_exception_handlers.py` từ `"A document with this title already exists"` sang `"Document already ingested"`.

- [ ] **Step 6: Sửa `test_transaction_boundary.py` cho model mới**

```python
# thay Document(title=title, content="x") bằng
        session.add(
            Document(
                sha256_hash=marker, filename="x.pdf", object_key=f"raw/{marker}.pdf", size_bytes=1
            )
        )
# và raw SQL: SELECT 1 FROM documents WHERE sha256_hash = :marker
```
(`marker` là một chuỗi 64 ký tự duy nhất mỗi test, ví dụ `uuid.uuid4().hex * 2`.)

- [ ] **Step 7: Migrate và chạy suite**

Run:
```bash
uv run alembic upgrade head
uv run pytest -q && uv run ruff check .
```
Expected: PASS, 55 test.

- [ ] **Step 8: Dừng và báo cáo**

Commit message đề xuất: `feat(db): replace CRUD schema with file entity and chunk tables`

---

### Task 9: API — presigned upload + đăng ký

**Files:**
- Create: `app/schemas/ingestion.py`, `app/repositories/document_repository.py`, `app/services/ingestion_service.py`, `app/controllers/ingestion_controller.py`
- Modify: `app/controllers/__init__.py`
- Create: `tests/test_ingestion_api.py`

**Interfaces:**
- Consumes: `ObjectStore`, `Document`, `ErrorCode`
- Produces:
  - `DocumentRepository.insert_if_new(sha256_hash, filename, object_key, size_bytes) -> Document | None` — trả `None` khi đã tồn tại
  - `IngestionService.create_upload_url(filename) -> UploadTarget`
  - `IngestionService.register(payload: DocumentRegister) -> Document`
  - `POST /documents/upload-url`, `POST /documents`

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_ingestion_api.py
import hashlib
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
    """Tin hash của client cho phép một client đầu độc dedup entry của
    client khác. API phải tự tính lại trên object đã lưu."""
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

Thêm fixture vào `tests/conftest.py`:

```python
@pytest.fixture
def uploaded_pdf():
    """Đẩy thật một fixture PDF lên MinIO rồi trả payload đăng ký."""
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

- [ ] **Step 2: Chạy để xác nhận đỏ**

Run: `uv run pytest tests/test_ingestion_api.py -q`
Expected: FAIL — 404 trên mọi route.

- [ ] **Step 3: Viết schema, repository, service, controller**

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
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.utils.database import get_session


class DocumentRepository:
    """Không có business rule; không bao giờ commit — session sở hữu transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert_if_new(
        self, sha256_hash: str, filename: str, object_key: str, size_bytes: int
    ) -> Document | None:
        """INSERT ... ON CONFLICT DO NOTHING RETURNING id.

        Một câu lệnh, nguyên tử. Trả None nghĩa là file đã tồn tại. Caller
        publish sang broker CHỈ khi nhận được row — luật đó làm enqueue
        exactly-once mỗi file mà không cần khoá phân tán.
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
        from sqlalchemy import func

        query = select(Document)
        if status:
            query = query.where(Document.status == status)
        total = await self.session.scalar(
            select(func.count()).select_from(query.subquery())
        )
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
        # Key tạm bằng uuid: hash thật chỉ biết sau khi client upload xong.
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

        # Tính lại hash trên object đã lưu. Tin hash của client cho phép một
        # client đầu độc dedup entry của client khác.
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

Đăng ký router trong `app/controllers/__init__.py`.

- [ ] **Step 4: Chạy test**

Run: `uv run pytest tests/test_ingestion_api.py -q`
Expected: PASS, 5 test.

- [ ] **Step 5: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(api): add presigned upload and document registration`

---

### Task 10: API — status, list, delete

**Files:**
- Modify: `app/controllers/ingestion_controller.py`, `app/services/ingestion_service.py`
- Modify: `tests/test_ingestion_api.py`

**Interfaces:**
- Consumes: `DocumentRepository.get_by_id`, `.list`
- Produces: `GET /documents/{id}/status`, `GET /documents`, `DELETE /documents/{id}`

- [ ] **Step 1: Viết test đỏ**

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
    """Trước đây guard này là int4 range. Giờ id là uuid nên FastAPI lo,
    nhưng phải xác nhận nó ra 422 chứ không phải nổ ở driver."""
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

- [ ] **Step 2: Chạy để xác nhận đỏ** — Run: `uv run pytest tests/test_ingestion_api.py -q` → FAIL 404.

- [ ] **Step 3: Thêm ba endpoint**

```python
# app/services/ingestion_service.py — thêm vào IngestionService
    async def get(self, document_id: uuid.UUID) -> Document:
        document = await self.repository.get_by_id(document_id)
        if document is None:
            raise AppException(ErrorCode.DOCUMENT_NOT_FOUND, f"Document {document_id} not found")
        return document

    async def list(self, limit: int, offset: int, status: str | None):
        return await self.repository.list(limit=limit, offset=offset, status=status)

    async def delete(self, document_id: uuid.UUID) -> None:
        """Cascade xuống chunks qua ON DELETE CASCADE; raw/ giữ nguyên —
        đó là nguồn sự thật, và là cách duy nhất ingest lại."""
        document = await self.get(document_id)
        await self.repository.session.delete(document)
```

```python
# app/controllers/ingestion_controller.py — thêm
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

- [ ] **Step 4: Chạy test** → PASS, 10 test trong file.

- [ ] **Step 5: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(api): add document status, list and delete endpoints`

---

### Task 11: Celery skeleton với 5 stage no-op

**Files:**
- Create: `worker/__init__.py`, `worker/celery_app.py`, `worker/db.py`, `worker/stages.py`
- Modify: `docker-compose.yml`, `app/services/ingestion_service.py`
- Create: `tests/test_stage_chain.py`

**Interfaces:**
- Consumes: `Settings.rabbitmq_url`, `.worker_database_url`
- Produces:
  - `worker.celery_app.app` (Celery instance)
  - `worker.db.session_scope()` context manager, sync
  - `worker.stages.{parse,structure,enrich,embed,persist}` — 5 task
  - `worker.stages.launch(document_id: str) -> None` — dựng chain và gửi đi
  - `worker.stages.STAGES = ["PARSING","STRUCTURING","ENRICHING","EMBEDDING","PERSISTING"]`

> **Vì sao task này cố ý không xử lý gì thật.** Nó chứng minh routing, chain, `acks_late`, retry và state machine chạy đúng trong lúc mọi stage còn tầm thường để debug. Bỏ qua nó nghĩa là sau này bạn phải debug ngữ nghĩa RabbitMQ và inference Docling **cùng lúc**.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_stage_chain.py
"""Chạy chain trong tiến trình bằng task_always_eager — kiểm logic chain và
state machine, không kiểm việc phân phối qua broker (cái đó kiểm bằng tay)."""

import uuid

import pytest

from worker.celery_app import app as celery_app
from worker.stages import STAGES, launch


@pytest.fixture(autouse=True)
def eager():
    celery_app.conf.task_always_eager = True
    yield
    celery_app.conf.task_always_eager = False


def test_task_routes_split_cpu_and_llm():
    routes = celery_app.conf.task_routes
    assert routes["worker.stages.parse"]["queue"] == "cpu"
    assert routes["worker.stages.structure"]["queue"] == "cpu"
    assert routes["worker.stages.enrich"]["queue"] == "llm"
    assert routes["worker.stages.embed"]["queue"] == "llm"
    assert routes["worker.stages.persist"]["queue"] == "cpu"


def test_acks_late_is_on():
    """Crash phải dẫn tới giao lại, không phải mất việc."""
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_no_result_backend():
    """documents.status trong Postgres là trạng thái bền, không phải backend."""
    assert celery_app.conf.result_backend is None


def test_chain_drives_a_document_to_completed(seeded_document):
    launch(str(seeded_document.id))

    refreshed = reload(seeded_document.id)
    assert refreshed.status == "COMPLETED"
    assert refreshed.stage == STAGES[-1]
    assert refreshed.completed_at is not None
```

Thêm helper vào cùng file:

```python
def reload(document_id):
    from worker.db import session_scope
    from app.models.document import Document

    with session_scope() as session:
        return session.get(Document, uuid.UUID(str(document_id)))
```

và fixture `seeded_document` vào `tests/conftest.py` — chèn một `Document` bằng engine **sync** rồi commit thật (chain chạy ngoài transaction của test):

```python
@pytest.fixture
def seeded_document(uploaded_pdf):
    import uuid as _uuid
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

- [ ] **Step 2: Chạy để xác nhận đỏ** → `ModuleNotFoundError: No module named 'worker'`

- [ ] **Step 3: Viết worker**

```python
# worker/db.py
"""Engine ĐỒNG BỘ, cố ý khác với engine async của API.

Hai engine, một bộ model. Task Celery là `def` thường — không có
asyncio.run() ở bất kỳ đâu trong worker/.
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
    result_backend=None,           # Postgres là nguồn sự thật cho trạng thái
    task_acks_late=True,           # crash → giao lại, không mất
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # đừng ôm task ở stage chậm
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
"""Năm stage. Ở task này tất cả đều no-op — chỉ đổi state machine.

documents.stage giữ stage VỪA HOÀN THÀNH, và chỉ tiến sau khi artifact đã
ghi bền. Nên cửa sổ crash luôn quy về "artifact đã ghi, con trỏ chưa tiến",
tức chạy lại đúng một stage, vô hại.
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
    _advance(document_id, "PARSING")
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
    chain(
        parse.s(document_id), structure.s(), enrich.s(), embed.s(), persist.s()
    ).apply_async()
```

- [ ] **Step 3b: Viết test đỏ cho đường hỏng**

```python
# thêm vào tests/test_stage_chain.py
from app.exceptions import AppException, ErrorCode
from worker.stages import stage_failed


def test_a_permanent_error_goes_straight_to_dead_letter(seeded_document):
    """PDF mã hoá sẽ không bao giờ parse được. Đốt 5 lần retry cho nó là
    lãng phí, và tệ hơn là che mất lý do thật."""
    stage_failed(
        str(seeded_document.id), "PARSING", AppException(ErrorCode.PDF_ENCRYPTED)
    )

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

- [ ] **Step 3c: Viết `stage_failed` và nối vào từng task**

```python
# worker/stages.py — thêm

MAX_ATTEMPTS = 3

# Những lỗi này là VĨNH VIỄN: file sẽ không bao giờ parse được, nên retry
# chỉ đốt thời gian và che mất lý do thật.
PERMANENT = {ErrorCode.PDF_ENCRYPTED, ErrorCode.PDF_TOO_LARGE, ErrorCode.HASH_MISMATCH}


def stage_failed(document_id: str, stage: str, exc: BaseException) -> None:
    """Ghi lại thất bại. DEAD_LETTER phải luôn kèm failed_stage và
    last_error — dead-letter im lặng là thứ không ai debug được."""
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

Bọc mỗi task bằng cùng một khuôn:

```python
@app.task(name="worker.stages.parse", bind=True, max_retries=3)
def parse(self, document_id: str) -> str:
    try:
        ...                                    # thân stage
    except AppException as exc:
        stage_failed(document_id, "PARSING", exc)
        raise                                  # KHÔNG retry lỗi vĩnh viễn
    except Exception as exc:
        stage_failed(document_id, "PARSING", exc)
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

> `AppException` là lỗi **vĩnh viễn** nên nằm ngoài `autoretry_for` — nó đi thẳng tới `DEAD_LETTER` thay vì đốt năm lần thử trên một file không bao giờ parse được.

- [ ] **Step 4: Nối API vào broker**

Trong `IngestionService.register`, sau khi `insert_if_new` trả về row:

```python
        from worker.stages import launch

        # Publish CHỈ khi có row trả về. Đó là luật làm enqueue exactly-once.
        launch(str(document.id))
        return document
```

- [ ] **Step 5: Thêm hai worker vào compose**

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

- [ ] **Step 6: Chạy test + kiểm tra bằng tay**

Run:
```bash
uv add celery
uv run pytest tests/test_stage_chain.py -q
docker compose up -d --build
docker compose logs -f worker-cpu   # xem 5 stage chạy qua
```
Expected: test PASS; log cho thấy `QUEUED → ... → COMPLETED`.

Kiểm tra thêm bằng tay: giết `worker-cpu` giữa chain (`docker compose kill worker-cpu`), bật lại, tài liệu phải được giao lại chứ không mắc kẹt.

- [ ] **Step 7: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(worker): add Celery app with five no-op stages`

---

### Task 12: `docling_service`

**Files:**
- Create: `docling_service/main.py`, `docling_service/Dockerfile`, `docling_service/pyproject.toml`
- Modify: `docker-compose.yml`
- Create: `tests/test_docling_service.py`

**Interfaces:**
- Consumes: MinIO (tự kéo object)
- Produces: `POST /parse` nhận `{"object_key": str, "pages": [int]}`, trả `{"pages": [{"page": int, "markdown": str, "confidence": float}]}`; `GET /health` chỉ trả 200 **sau khi model đã nạp xong**

- [ ] **Step 1: Viết test đỏ**

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
    """Healthcheck phải chờ model nạp xong, không phải chờ cổng mở — nếu
    không worker sẽ bắn request vào service chưa sẵn sàng và bạn đi debug
    nhầm chỗ."""
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

- [ ] **Step 2: Chạy để xác nhận đỏ** → connection refused.

- [ ] **Step 3: Viết service**

```python
# docling_service/main.py
"""Docling gói trong một container riêng.

Model nạp MỘT LẦN lúc process khởi động, không phải mỗi request. Concurrency
1. Service tự kéo object từ MinIO — không đẩy payload vài MB qua HTTP.
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
    return {
        "pages": [
            {"page": p, "markdown": markdown, "confidence": 0.9} for p in request.pages
        ]
    }
```

```dockerfile
# docling_service/Dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv
WORKDIR /svc
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never PATH="/opt/venv/bin:$PATH"
# Model tải vào đây; compose mount volume lên đường dẫn này nên chỉ tải một lần.
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

- [ ] **Step 4: Thêm vào compose**

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
      # Weights tải một lần rồi dùng mãi. Lần chạy đầu mất vài phút và
      # TRÔNG Y HỆT TREO MÁY — đó là bình thường.
      - doclingmodels:/models
    healthcheck:
      # Chờ model nạp xong, không phải chờ cổng mở.
      test: ["CMD-SHELL", "python -c \"import urllib.request,json;
             assert json.load(urllib.request.urlopen('http://localhost:8100/health'))['models_loaded']\""]
      interval: 15s
      timeout: 10s
      retries: 40
      start_period: 300s
```

Thêm `doclingmodels:` vào khối `volumes:`.

> **Nếu bạn đã cài `nvidia-container-toolkit`**, thêm vào service `docling`:
> ```yaml
>     deploy:
>       resources:
>         reservations:
>           devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
> ```
> **Nếu chưa**, Docling chạy CPU-only — đặt `DOCLING_PAGE_TIMEOUT_S=300` trong `.env`.

- [ ] **Step 5: Bật lên và chạy test**

Run:
```bash
docker compose up -d --build docling
docker compose logs -f docling      # chờ "models_loaded": lần đầu vài phút
uv run pytest tests/test_docling_service.py -q
```
Expected: PASS, 2 test.

- [ ] **Step 6: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(docling): add docling parsing service container`

---

### Task 13: S1 Parse

**Files:**
- Create: `worker/parsing.py`
- Modify: `worker/stages.py`
- Create: `tests/test_parsing.py`

**Interfaces:**
- Consumes: `ObjectStore`, `Settings.docling_url/.docling_page_timeout_s/.max_page_count`
- Produces:
  - `parse_document(object_key: str, store: ObjectStore, docling_url: str) -> dict` trả `{"page_count": int, "pages": [{"page": int, "markdown": str, "source": "pymupdf"|"docling", "confidence": float}]}`
  - `is_scanned(total_text_chars: int, page_count: int) -> bool`
  - Ngoại lệ: `AppException(ErrorCode.PDF_ENCRYPTED)`, `AppException(ErrorCode.PDF_TOO_LARGE)`
  - Checkpoint: `staging/{document_id}/parsed.json`

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_parsing.py
from pathlib import Path

import pytest

from app.exceptions import AppException, ErrorCode
from worker.parsing import is_scanned, parse_document

FIXTURES = Path(__file__).parent / "fixtures"


def test_scanned_detection_uses_chars_per_page():
    assert is_scanned(total_text_chars=50, page_count=3) is True
    assert is_scanned(total_text_chars=5000, page_count=3) is False


def test_scanned_detection_handles_zero_pages():
    """Chia cho 0 ở đây là một crash rất dễ xảy ra và rất khó truy."""
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

Thêm hai fixture vào `tests/conftest.py`:

```python
@pytest.fixture
def store():
    from shared.storage import ObjectStore
    from app.config import get_settings

    return ObjectStore(endpoint=get_settings().minio_public_url)


@pytest.fixture
def uploaded(store):
    """Đẩy một fixture PDF theo tên lên MinIO, trả về object_key."""
    import hashlib
    from pathlib import Path

    def _upload(name: str) -> str:
        data = (Path(__file__).parent / "fixtures" / name).read_bytes()
        key = f"raw/{hashlib.sha256(data).hexdigest()}.pdf"
        store.put(key, data)
        return key

    return _upload
```

- [ ] **Step 2: Chạy để xác nhận đỏ** → module chưa tồn tại.

- [ ] **Step 3: Viết implementation**

```python
# worker/parsing.py
"""S1 — Parse. PyMuPDF quét nhanh, Docling cho trang khó.

Task chạy trên worker-cpu nhưng CÔNG VIỆC THẬT xảy ra trong container
docling: worker chỉ giữ một HTTP connection, không giữ model weights. Đó là
lý do S1 an toàn ở concurrency 8.
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
    """Không có text layer nghĩa là ảnh. page_count == 0 tính là scan chứ
    không phải chia cho 0."""
    if page_count == 0:
        return True
    return total_text_chars / page_count < SCANNED_CHARS_PER_PAGE


def parse_document(object_key: str, store: ObjectStore, docling_url: str | None) -> dict:
    data = store.get(object_key)

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise AppException(ErrorCode.PDF_ENCRYPTED, f"Không mở được PDF: {exc}") from exc

    if document.needs_pass:
        raise AppException(ErrorCode.PDF_ENCRYPTED)
    if document.page_count > MAX_PAGE_COUNT:
        raise AppException(
            ErrorCode.PDF_TOO_LARGE, f"{document.page_count} trang, trần là {MAX_PAGE_COUNT}"
        )

    texts = [page.get_text() for page in document]
    total_chars = sum(len(t) for t in texts)

    if is_scanned(total_chars, document.page_count) and docling_url:
        needs_docling = list(range(1, document.page_count + 1))
    else:
        needs_docling = [
            i + 1
            for i, page in enumerate(document)
            if page.find_tables().tables or page.get_images()
        ] if docling_url else []

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
            # Hết giờ hoặc lỗi: giữ text thô của PyMuPDF, đánh dấu confidence 0
            # và ghi log. Một trang kém còn hơn cả tài liệu chết.
            pages[page_number - 1]["confidence"] = 0.0

    return {"page_count": document.page_count, "pages": pages}
```

- [ ] **Step 4: Nối vào stage**

```python
# worker/stages.py — thay thân task parse
@app.task(name="worker.stages.parse", bind=True, max_retries=3)
def parse(self, document_id: str) -> str:
    from shared.storage import get_store
    from worker.parsing import parse_document

    store = get_store()
    key = f"staging/{document_id}/parsed.json"
    if store.exists(key):                      # checkpoint skip
        _advance(document_id, "PARSING", stage="PARSING")
        return document_id

    _advance(document_id, "PARSING")
    with session_scope() as session:
        document = session.get(Document, uuid.UUID(document_id))
        object_key = document.object_key

    result = parse_document(object_key, store, get_settings().docling_url)
    store.put_json(key, result)

    with session_scope() as session:
        session.get(Document, uuid.UUID(document_id)).page_count = result["page_count"]

    _advance(document_id, "PARSING", stage="PARSING")
    return document_id
```

- [ ] **Step 5: Chạy test** → PASS, 6 test.

- [ ] **Step 6: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(worker): implement S1 parse stage`

---

### Task 14: S2 Structure

**Files:**
- Create: `worker/chunking.py`
- Modify: `worker/stages.py`
- Create: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `parsed.json`
- Produces:
  - `sanitize(markdown: str) -> str`
  - `chunk_document(parsed: dict) -> dict` trả `{"parents": [...], "children": [...]}`
  - Checkpoint: `staging/{document_id}/chunks.json`

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_chunking.py
from worker.chunking import chunk_document, count_tokens, sanitize


def test_sanitize_collapses_blank_runs():
    assert sanitize("a\n\n\n\n\nb") == "a\n\nb"


def test_sanitize_skips_fenced_code_blocks():
    """Regex vá pipe sẽ phá code. Code block phải đi qua nguyên vẹn."""
    source = "văn bản\n\n```python\nx = [1|2|3]\n```\n\nvăn bản"

    assert "x = [1|2|3]" in sanitize(source)


def test_chunk_index_is_deterministic():
    """Đây là khoá tự nhiên làm S5 idempotent. Bất kỳ tính bất định nào ở
    đây — thứ tự dict, set, timestamp — sẽ âm thầm phá retry safety và chỉ
    lộ ra rất lâu sau, dưới dạng row trùng."""
    parsed = {"page_count": 2, "pages": [
        {"page": 1, "markdown": "# Tiêu đề\n\n" + "câu văn. " * 200, "source": "pymupdf", "confidence": 1.0},
        {"page": 2, "markdown": "## Mục hai\n\n" + "câu khác. " * 200, "source": "pymupdf", "confidence": 1.0},
    ]}

    first = chunk_document(parsed)
    second = chunk_document(parsed)

    assert [c["chunk_index"] for c in first["children"]] == [
        c["chunk_index"] for c in second["children"]
    ]
    assert first == second


def test_children_carry_a_page_number_and_a_parent_index():
    parsed = {"page_count": 1, "pages": [
        {"page": 1, "markdown": "# T\n\n" + "câu. " * 300, "source": "pymupdf", "confidence": 1.0}
    ]}

    result = chunk_document(parsed)

    assert all(c["page_number"] == 1 for c in result["children"])
    assert all(c["parent_index"] in {p["chunk_index"] for p in result["parents"]}
               for c in result["children"])


def test_chunks_under_twenty_tokens_are_dropped():
    """Drop TRƯỚC khi có gì tính tiền theo token ở S3/S4."""
    parsed = {"page_count": 1, "pages": [
        {"page": 1, "markdown": "ngắn", "source": "pymupdf", "confidence": 1.0}
    ]}

    assert chunk_document(parsed)["children"] == []


def test_parent_chunks_stay_within_the_token_band():
    parsed = {"page_count": 1, "pages": [
        {"page": 1, "markdown": "# T\n\n" + "từ " * 3000, "source": "pymupdf", "confidence": 1.0}
    ]}

    parents = chunk_document(parsed)["parents"]

    assert all(500 <= p["token_count"] <= 1000 for p in parents[:-1])
```

- [ ] **Step 2: Chạy để xác nhận đỏ.**

- [ ] **Step 3: Viết implementation**

```python
# worker/chunking.py
"""S2 — Structure. Thuần CPU, không gọi mạng. Stage duy nhất mà chạy lại
tốn vài mili giây.
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
    """Gộp dòng trống, bỏ header/footer lặp — nhưng KHÔNG đụng vào fenced
    code block: regex vá pipe của bảng sẽ phá cú pháp code."""
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
    """chunk_index gán theo thứ tự tài liệu, tất định. Đó là khoá tự nhiên."""
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

- [ ] **Step 4: Nối vào stage `structure`**

```python
# worker/stages.py — thay thân task structure
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

- [ ] **Step 5: Chạy test** → PASS, 6 test.

- [ ] **Step 6: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(worker): implement S2 structure stage`

---

### Task 15: S3 Enrich + rate limiter + trần blast-radius

**Files:**
- Create: `worker/enrichment.py`
- Modify: `worker/stages.py`
- Create: `tests/test_enrichment.py`

**Interfaces:**
- Consumes: `LLMProvider`, `get_bucket`, `chunks.json`
- Produces:
  - `enrich_chunks(children: list[dict], provider: LLMProvider, batch_size: int = 20) -> list[dict]`
  - `validate_batch(batch: list[dict], results: list[EnrichedChunk]) -> list[int]` trả danh sách id thiếu/không hợp lệ
  - Checkpoint: `staging/{document_id}/enriched.json`

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_enrichment.py
"""Ba test đầu là lý do StubProvider tồn tại — không cách nào bắt API thật
trả về sai theo yêu cầu."""

import pytest

from app.exceptions import AppException
from shared.llm import CATEGORIES, EnrichedChunk, StubProvider
from worker.enrichment import enrich_chunks


class CountingStub(StubProvider):
    """Hỏng cả lô lần đầu, thành công khi được gọi lại đơn lẻ, và ghi lại
    những id nào đã bị gọi lại một mình."""

    def __init__(self, drop_ids=None):
        super().__init__(drop_ids=drop_ids)
        self.solo_calls = []

    def enrich(self, chunks):
        if len(chunks) == 1:
            self.solo_calls.append(chunks[0]["id"])
            return [EnrichedChunk(id=chunks[0]["id"], context="đã khôi phục", category="TECHNICAL")]
        return super().enrich(chunks)


@pytest.fixture
def children():
    return [
        {"id": i, "chunk_index": i, "content": f"nội dung {i}", "token_count": 120}
        for i in range(25)
    ]


def test_every_chunk_gets_a_context_and_a_category(children):
    result = enrich_chunks(children, provider=StubProvider(), batch_size=20)

    assert len(result) == len(children)
    assert all(r["context"] for r in result)
    assert all(r["category"] in CATEGORIES for r in result)


def test_a_missing_id_is_retried_alone_not_the_whole_batch(children):
    """Tiêu chí nghiệm thu của bước này: 19 trên 20 phải kích hoạt retry
    ĐÚNG id thiếu, không phải cả lô."""
    provider = CountingStub(drop_ids=[7])

    result = enrich_chunks(children, provider=provider, batch_size=20)

    assert len(result) == len(children)
    assert provider.solo_calls == [7], "phải gọi lại đúng một id, một mình"


def test_a_chunk_that_fails_twice_gets_an_empty_context_and_proceeds(children):
    """Một câu ngữ cảnh thiếu làm retrieval kém đi một chút; một tài liệu
    dead-letter thì không giúp ai."""
    provider = StubProvider(drop_ids=[3])

    result = enrich_chunks(children, provider=provider, batch_size=20)

    stubborn = next(r for r in result if r["id"] == 3)
    assert stubborn["context"] == ""


def test_an_off_enum_category_is_rejected_and_replaced_with_other(children):
    result = enrich_chunks(children, provider=StubProvider(bad_category_ids=[5]), batch_size=20)

    assert next(r for r in result if r["id"] == 5)["category"] == "OTHER"


def test_a_document_over_the_chunk_cap_is_refused_before_spending(children, monkeypatch):
    """Rate limit chặn TỐC ĐỘ, không chặn TỔNG. Trần này là thứ chặn tổng."""
    monkeypatch.setattr("worker.enrichment.MAX_CHUNKS_PER_DOC", 10)

    with pytest.raises(AppException):
        enrich_chunks(children, provider=StubProvider(), batch_size=20)
```

- [ ] **Step 2: Chạy để xác nhận đỏ.**

- [ ] **Step 3: Viết implementation**

```python
# worker/enrichment.py
"""S3 — Enrich. Stage đắt nhất và đáng checkpoint nhất.

Validate TRƯỚC khi ghi bất cứ thứ gì. Batch hỏng thì thử lại đúng những id
thiếu, từng cái một — không bao giờ cả lô 20.
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
    """Trả về id cần gọi lại. Ba kiểm tra, không bỏ cái nào."""
    wanted = {c["id"] for c in batch}
    seen: dict[int, EnrichedChunk] = {}
    for item in results:
        if item.id in wanted and item.id not in seen:
            seen[item.id] = item
    return sorted(wanted - set(seen))


def _normalise(item: EnrichedChunk) -> dict:
    """Category ngoài enum bị từ chối, không lưu, và ghi log như đề xuất.
    Gắn tag tự do sẽ tích Finance/Financial/financial-reports trong một tuần."""
    category = item.category if item.category in CATEGORIES else "OTHER"
    if category != item.category:
        print(f"[taxonomy] đề xuất ngoài enum bị từ chối: {item.category!r}")
    return {"id": item.id, "context": item.context, "category": category}


def _acquire_or_defer(estimated_tokens: int) -> None:
    for bucket_name, cost in (("chat_rpm", 1), ("chat_tpm", estimated_tokens)):
        allowed, wait_ms = get_bucket(bucket_name).acquire(cost)
        if not allowed:
            from celery import current_task

            # Requeue, KHÔNG sleep: worker đang ngủ vẫn giữ slot và báo cáo
            # là bận, làm pool hiện 100% utilized trong khi không làm gì.
            # max_retries=None vì bị rate limit không phải là lỗi.
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
            f"{len(children)} chunk vượt trần {MAX_CHUNKS_PER_DOC}",
        )

    out: dict[int, dict] = {}

    for start in range(0, len(children), batch_size):
        batch = children[start : start + batch_size]
        _acquire_or_defer(sum(c["token_count"] for c in batch) if "token_count" in batch[0] else len(batch) * 150)

        results = provider.enrich(batch)
        for item in results:
            if item.id in {c["id"] for c in batch}:
                out[item.id] = _normalise(item)

        for missing_id in validate_batch(batch, results):
            single = next(c for c in batch if c["id"] == missing_id)
            for _ in range(SOLO_ATTEMPTS):
                retry = provider.enrich([single])
                if retry and retry[0].id == missing_id:
                    out[missing_id] = _normalise(retry[0])
                    break
            else:
                # Thất bại đơn lẻ hai lần: context rỗng và đi tiếp.
                out[missing_id] = {"id": missing_id, "context": "", "category": "OTHER"}

    return [out[c["id"]] for c in children]
```

- [ ] **Step 4: Nối vào stage `enrich`**

```python
# worker/stages.py — thay thân task enrich
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
    # chunk_index đóng vai "id" trong hợp đồng với provider — tất định, nên
    # gọi lại cũng ra đúng bộ id đó.
    children = [{**c, "id": c["chunk_index"]} for c in chunks["children"]]

    enriched = enrich_chunks(
        children, provider=get_provider(), batch_size=get_settings().enrich_batch_size
    )
    store.put_json(key, {"chunks": enriched})
    _advance(document_id, "ENRICHING", stage="ENRICHING")
    return document_id
```

- [ ] **Step 5: Chạy test** → PASS, 5 test.

- [ ] **Step 6: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(worker): implement S3 enrich with batch validation and backpressure`

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
  - `assert_normalised(vectors) -> None`
  - Checkpoint: `staging/{document_id}/embeddings.npy` + `manifest.json`

- [ ] **Step 1: Viết test đỏ**

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
    """vector_ip_ops giả định vector đơn vị. Bỏ chuẩn hoá thì index trả về
    láng giềng SAI mà không có lỗi nào — assert trong code, không chỉ trong test."""
    for vector in embed_chunks(["xin chào"], provider=StubProvider()):
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

- [ ] **Step 2: Chạy để xác nhận đỏ.**

- [ ] **Step 3: Viết implementation**

```python
# worker/embedding.py
"""S4 — Embed. Queue riêng với S3 vì hạn ngạch chat và embedding của nhà
cung cấp là hai thứ riêng; nhốt chung một bucket thì một cái bỏ đói cái kia.
"""

import math
import random

from app.config import get_settings
from shared.llm import LLMProvider
from shared.rate_limiter import get_bucket

_settings = get_settings()
DIMENSIONS = _settings.embed_dimensions


def assert_normalised(vectors: list[list[float]]) -> None:
    """Không phải mỹ phẩm. Đây là điều kiện để vector_ip_ops đúng."""
    for vector in vectors:
        norm = math.sqrt(sum(x * x for x in vector))
        assert abs(norm - 1.0) < 1e-4, f"vector chưa chuẩn hoá L2 (norm={norm})"


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

        assert len(result) == len(batch), f"nhận {len(result)} vector cho {len(batch)} text"
        assert all(len(v) == DIMENSIONS for v in result), "sai số chiều"
        assert all(all(math.isfinite(x) for x in v) for v in result), "vector có giá trị vô hạn"
        assert_normalised(result)

        vectors.extend(result)
    return vectors
```

- [ ] **Step 4: Nối vào stage `embed`**

Run: `uv add numpy`

```python
# worker/stages.py — thay thân task embed
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
        # Thứ được nhúng là contextualized, KHÔNG phải content thô.
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

- [ ] **Step 5: Chạy test** → PASS, 5 test.

- [ ] **Step 6: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(worker): implement S4 embed stage`

---

### Task 17: S5 Persist

**Files:**
- Create: `worker/persistence.py`
- Modify: `worker/stages.py`
- Create: `tests/test_persistence.py`

**Interfaces:**
- Consumes: `chunks.json`, `enriched.json`, `embeddings.npy`, `manifest.json`
- Produces: `persist_document(document_id: uuid.UUID, chunks: dict, enriched: list[dict], vectors: list[list[float]], session: Session) -> None`

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_persistence.py
"""Idempotency test là quan trọng nhất ở đây: chạy lại phải KHÔNG đổi gì."""

import uuid

from sqlalchemy import func, select

from app.models import ChildChunk, Document, ParentChunk
from worker.db import session_scope
from worker.persistence import persist_document


def _payload():
    chunks = {
        "parents": [{"chunk_index": 0, "content": "cha", "token_count": 600,
                     "page_start": 1, "page_end": 1}],
        "children": [{"chunk_index": 0, "parent_index": 0, "content": "con",
                      "token_count": 120, "page_number": 1}],
    }
    enriched = [{"id": 0, "context": "ngữ cảnh", "category": "TECHNICAL"}]
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
    """ON CONFLICT (document_id, chunk_index) DO UPDATE. Chạy lại là chuyện
    nhàm chán, không phải chuyện làm hỏng dữ liệu."""
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

- [ ] **Step 2: Chạy để xác nhận đỏ.**

- [ ] **Step 3: Viết implementation**

```python
# worker/persistence.py
"""S5 — Persist. Một transaction, sync Session. Luôn an toàn khi chạy lại."""

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

    # 1. Upsert parent, lấy map chunk_index -> id
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

    # 2. Upsert child, giải parent_id từ map
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

    # 3. Đóng sổ
    document = session.get(Document, document_id)
    document.status = "COMPLETED"
    document.stage = "PERSISTING"
    document.completed_at = datetime.now(timezone.utc)
```

- [ ] **Step 4: Nối vào stage `persist`**

```python
# worker/stages.py — thay thân task persist
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

> Stage này **không** có checkpoint skip, khác bốn stage kia. Không cần: bản
> thân upsert đã idempotent, nên chạy lại là chuyện nhàm chán chứ không phải
> chuyện làm hỏng dữ liệu.

- [ ] **Step 5: Chạy test** → PASS, 4 test.

- [ ] **Step 6: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(worker): implement S5 persist stage with idempotent upserts`

---

### Task 18: Idempotency và crash recovery (kết thúc Phase 1a)

**Files:**
- Create: `tests/test_pipeline_e2e.py`

**Interfaces:**
- Consumes: toàn bộ chain
- Produces: không có code mới — chỉ bằng chứng

- [ ] **Step 1: Viết test**

```python
# tests/test_pipeline_e2e.py
"""Chain đầy đủ, chạy bằng StubProvider. Không cần OPENAI_API_KEY."""

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
    """Xoá enriched.json: S1/S2 phải bỏ qua nhờ checkpoint skip, S3 chạy
    lại, kết quả cuối giống hệt."""
    launch(str(seeded_document.id))
    before = _counts(seeded_document.id)

    store = get_public_store()
    store._client.delete_object(
        Bucket="staging", Key=f"{seeded_document.id}/enriched.json"
    )

    launch(str(seeded_document.id))

    assert _counts(seeded_document.id) == before
```

- [ ] **Step 2: Chạy** → PASS, 3 test.

- [ ] **Step 3: Chạy toàn bộ suite và lint**

Run: `uv run pytest -q && uv run ruff check .`

- [ ] **Step 4: Dừng và báo cáo — Phase 1a xong**

Báo cáo phải nêu: tổng số test, `docker compose ps` cho thấy mấy container healthy, và một tài liệu đi trọn `QUEUED → COMPLETED`.

Commit message đề xuất: `test: add end-to-end pipeline, idempotency and resume tests`

---

## Phase 1b — cần `OPENAI_API_KEY`

### Task 19: Provider thật + similarity smoke test

**Files:**
- Modify: `.env`
- Create: `tests/test_similarity_smoke.py`

**Interfaces:**
- Consumes: `OpenAIProvider`
- Produces: bằng chứng rằng dữ liệu *dùng được*

> **Đây là test quan trọng nhất trong toàn bộ Phase 1.** Normalize sai, số chiều sai, và index cấu hình sai — cả ba trông y hệt "đã insert thành công". Đây là thứ duy nhất phân biệt được.

- [ ] **Step 1: Cắm key**

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

- [ ] **Step 2: Viết test**

```python
# tests/test_similarity_smoke.py
"""KHÔNG chạy được bằng stub: vector giả không mang ngữ nghĩa, nên hai đoạn
văn bất kỳ đều "gần nhau" một cách ngẫu nhiên."""

import os

import pytest
from sqlalchemy import text

from shared.llm import OpenAIProvider
from worker.db import session_scope
from worker.stages import launch

pytestmark = pytest.mark.skipif(
    os.getenv("LLM_PROVIDER") != "openai", reason="cần OPENAI_API_KEY và LLM_PROVIDER=openai"
)


def test_a_related_question_ranks_the_right_chunk_first(seeded_document):
    launch(str(seeded_document.id))

    question = "Doanh thu quý 3 là bao nhiêu?"
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

    assert rows, "không có chunk nào — pipeline chưa ghi được gì"
    assert "doanh thu" in rows[0][0].lower(), (
        "đoạn khớp nhất không chứa nội dung liên quan — kiểm tra L2-normalize, "
        "số chiều, và opclass của index HNSW"
    )
```

- [ ] **Step 3: Chạy**

Run: `uv run pytest tests/test_similarity_smoke.py -q`
Expected: PASS.

Nếu đỏ, kiểm theo thứ tự: (1) vector đã chuẩn hoá L2 chưa, (2) `EMBED_DIMENSIONS` có khớp `vector(1536)` không, (3) index có dùng `vector_ip_ops` không.

- [ ] **Step 4: Chạy trọn Definition of Done**

```bash
docker compose up -d --build          # 8 container healthy
uv run alembic upgrade head
uv run pytest -q                      # toàn bộ xanh
uv run ruff check .                   # sạch
test ! -d app/core && echo "app/core đúng là không tồn tại"
```

- [ ] **Step 5: Dừng và báo cáo — Phase 1 xong**

Commit message đề xuất: `test: add similarity smoke test against the real provider`

---

## Definition of Done — Phase 1

1. `docker compose up -d` cho **8 container healthy**.
2. `alembic upgrade head` áp sạch lên database rỗng.
3. Presigned upload → `POST /documents` → 202 kèm `document_id`.
4. Upload lại cùng file → **409, không publish message thứ hai**.
5. Chain đưa mọi fixture `QUEUED → COMPLETED`, trừ `encrypted.pdf` vào `DEAD_LETTER` **có lý do cụ thể**.
6. `parent_chunks` và `child_chunks` có row với embedding 1536 chiều non-null; cột `embedding` trên `documents` không còn.
7. Chạy lại một tài liệu đã COMPLETED → **không đổi row nào**.
8. Similarity smoke test xếp đúng chunk lên đầu.
9. Toàn bộ suite xanh; `ruff check .` sạch.
10. **`app/core/` không tồn tại.**
