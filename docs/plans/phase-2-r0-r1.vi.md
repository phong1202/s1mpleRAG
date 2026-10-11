# Phase 2 · R0 + R1 — Đường nối, Basic RAG, và cái thước đo · Implementation Plan

> **Cho agentic worker:** REQUIRED SUB-SKILL: dùng superpowers:executing-plans để làm plan này theo từng task. Các bước dùng cú pháp checkbox (`- [ ]`) để theo dõi.

**Mục tiêu:** Trả lời một câu hỏi từ corpus kèm trích dẫn chỉ đúng tài liệu và số trang — và có khả năng chứng minh bằng một con số rằng bất kỳ thay đổi nào sau này có làm retrieval tốt lên hay không.

**Kiến trúc:** Một vòng `for` trong `app/core/pipeline.py` với tám chỗ nối. Ba chỗ là thật ở R0 (truy hồi, dựng ngữ cảnh, sinh câu trả lời); năm chỗ là no-op (viết lại, lập kế hoạch, gộp, cổng, phản tư) để các bậc sau thay thế mà không phải cấu trúc lại gì. Trần vòng lặp là một hằng số config mà R6 sẽ nâng từ `1` lên `3`. R1 thêm một hàng trace cho mỗi truy vấn và một bộ câu hỏi có nhãn — đó là thứ mọi bậc từ R2 trở đi phải cãi dựa vào.

**Tech Stack:** FastAPI (async) · SQLAlchemy 2 async + `asyncpg` · Postgres 16 + pgvector (HNSW, `vector_ip_ops`) · OpenAI `text-embedding-3-large` (cắt còn 1536 chiều) / `gpt-6-luna` sau một Protocol · pytest · uv · ruff

**Spec:** [`docs/retrieval-design.md`](../retrieval-design.md) — thiết kế đã chốt. Chỗ nào plan này và spec mâu thuẫn, **spec thắng**.

---

## Đồng bộ với `main` — 2026-10-10

Phase 1 đã merge vào `main` (PR #1). Ngày 2026-10-10 `phase-2` được rebase lên `main`: 13 commit của nhánh giờ nằm trên 46 commit mà `main` có thêm. Phần này ghi lại những gì lần rebase đó làm thay đổi trong plan. Chỗ nào phần này mâu thuẫn với một task viết trước ngày đó, **phần này thắng**.

### Trạng thái

Cập nhật 2026-10-10, sau lượt thực thi đầu. Chưa có gì được commit; mỗi task có một snapshot riêng chờ duyệt.

| Task | Trạng thái |
| --- | --- |
| 1 → 9 (R0) | Đã commit trên `phase-2`. Tóm tắt ở mục R0. |
| 9b, 9c | Xong. |
| 9d | **Chờ**: gọi OpenAI (3 embed + 3 chat), cần được cho phép. |
| 10, 11 | Xong. |
| 12 | Step 1 → 7 xong: runner, test, 28 mục nháp, máy báo 20 nhãn trúng được. **Chờ**: người duyệt `eval/dataset.yaml` (Step 8), rồi đo con số nền bằng OpenAI (Step 9). |
| 13 | README viết lại và các dữ kiện trong spec đã sửa (Step 1 → 3, 5). **Còn**: chạy thử các lệnh README đưa ra trên stack đã tắt (Step 4), và con số nền ở spec §8. |
| 14 | Xong: khuyến nghị không dùng DeepSeek. |

Suite hiện tại: 318 test xanh (`--ignore=tests/test_ocr_service.py`), `ruff check .` và `ruff format --check .` sạch.

### Xung đột đã gỡ khi rebase

- **`Dockerfile`, `pyproject.toml`** (commit context builder). `main` đã có `tiktoken` và đã đổ sẵn bảng BPE vào `/opt/tiktoken-cache`. Giữ nguyên bản của `main`; commit đó giờ chỉ thêm `app/core/context_builder.py` và test của nó. Đoạn Dockerfile ở Task 6 Step 1 không còn áp dụng.
- **`app/controllers/__init__.py`** (commit `POST /query`). Đăng ký cả `ingestion_controller` lẫn `query_controller`.

Nhánh local giờ lệch `origin/phase-2`. Đẩy lên cần `git push --force-with-lease`, và chỉ khi được bảo.

### `main` đã đổi gì mà plan này phải theo

| Chuyện | Trên `main` | Hệ quả ở đây |
| --- | --- | --- |
| Model | `text-embedding-3-large` cắt còn 1536 chiều; chat `gpt-6-luna` với `OPENAI_REASONING_EFFORT=low` | Dòng Tech Stack đã sửa. `AsyncOpenAIProvider` dùng đúng model và số chiều, nhưng chưa truyền `reasoning_effort` → Task 9c |
| Provider đồng bộ đã được gia cố | Structured outputs `strict`, báo lỗi rõ khi thiếu key | Bản async vẫn gọi JSON mode và không đưa schema cho model → Task 9c |
| CI (`.github/workflows/ci.yml`) | PR vào `main` chạy `ruff check`, `ruff format --check`, pytest và build `web/` | 14 file của nhánh này chưa format → Task 9b. "Suite xanh" giờ gồm cả format |
| Test cần dịch vụ thật | Postgres, RabbitMQ, Redis, MinIO | `docker compose up -d --wait db rabbitmq redis minio` trước khi chạy suite |
| Schema có phần database tự giữ | Index HNSW, cột generated `child_chunks.tsv`, hai index và một check constraint trên `documents` cố ý không map trên model | `alembic --autogenerate` đề xuất **xoá** hết chúng → Task 10 Step 4 |
| Corpus thật | 10 tài liệu, khoảng 2.960 child chunk, luật giao thông tiếng Việt; PDF gốc nằm ở `data/` (không commit) | Các ví dụ hoá đơn điện tử ở Task 12 chỉ còn là hình dạng mẫu; nhãn thật gán trên corpus này |
| `.env` của máy dev | `LLM_PROVIDER=openai` | Mọi lần chạy thật (DoD 1, Task 12 Step 6) đều tốn tiền OpenAI → ràng buộc mới ở Global Constraints |
| Fixture test | `client` của `tests/conftest.py` không thay provider của đường đọc | Test của Task 11 viết lại theo `query_client` |
| FE (`web/`) | Có khung chat và trình xem PDF; kiểu `SourceReference` còn ghi là tạm | Xem mục dưới |

### Ngoài plan này

- **Nối FE với `POST /query`.** Một session FE lên plan và làm riêng, theo [`docs/fe-query-handoff.md`](../fe-query-handoff.md). Plan này không đụng tới `web/`.
- **`locations` cho trích dẫn** (`docs/system-design.md` §5.3: format đã chốt, ghi là "built in Phase 2"). Cần S1 giữ lại bbox, S2 gắn vào child, một migration, và **nạp lại corpus**. Cần một plan riêng. Cho tới khi có, FE lùi về `page_number`, thứ R0 đã trả.
- **Đường đọc không đi qua rate limiter của worker** (`rl:openai:*`). Một lỗi 429 từ OpenAI ở `retrieve` hay `generate` hiện ra thành 500. Chấp nhận ở R0/R1; xét lại khi R4 → R7 thêm lời gọi LLM vào mỗi request.
- **`/query` không có xác thực**, như mọi endpoint khác (rủi ro đã ghi ở plan Phase 1).

---

## Global Constraints

Mọi task đều ngầm mang những ràng buộc này. Giá trị chép nguyên văn từ spec.

- **Python** `>=3.12`; dependency thêm bằng `uv add` / `uv add --dev`, không bao giờ sửa tay `pyproject.toml`.
- **Branch:** toàn bộ R0 và R1 nằm trên nhánh `phase-2`.
- **Git:** KHÔNG chạy `git commit`, `git push`, `git merge`. Kết thúc mỗi task bằng dừng và báo cáo; chờ chỉ dẫn tường minh.
- **Điểm cắt nhánh: `phase-2` nằm trên `main`** kể từ lần rebase 2026-10-10, và `main` đã mang toàn bộ Phase 1. **Mọi test trong plan này đều tự gieo dữ liệu của nó**, nên chúng cần schema và các dịch vụ đang chạy, không cần corpus. Chỉ mục 1 của Definition of Done và Task 12 Step 6 mới cần corpus đã nạp thật. Rebase lại mỗi khi `main` tiến.
- **Suite phải xanh ở cuối MỌI task.** `uv run pytest -q --ignore=tests/test_ocr_service.py`, `uv run ruff check .` và `uv run ruff format --check .` đều sạch: đó là ba thứ CI chạy cho PR vào `main`. Suite cần Postgres, RabbitMQ, Redis và MinIO đang chạy (`docker compose up -d --wait db rabbitmq redis minio`). `tests/test_ocr_service.py` cần `chandra` và GPU, và không liên quan tới đường đọc.
- **Phía API là async.** Mọi thứ dưới `app/core/` là `async def`. `LLMProvider` đồng bộ của worker không bị đụng vào; R0 thêm một Protocol thứ hai đứng cạnh.
- **`ef_search` được đặt tường minh cho mỗi session** và phải `>= 2 x RETRIEVAL_OVER_FETCH`. Mặc định của pgvector là 40; xin 50 hàng xóm từ một hàng đợi rộng 40 sẽ làm phần đuôi kém đi trong im lặng.
- **`<#>` là tích vô hướng ÂM.** `ORDER BY ... ASC` là gần nhất trước. `DESC` đảo ngược toàn bộ kết quả và không ném lỗi nào.
- **`DISTINCT ON (parent_id)` đòi `parent_id` đứng đầu `ORDER BY`.** Nếu không, Postgres giữ một child tuỳ ý trong mỗi nhóm, không báo lỗi.
- **`Candidate.rank` là thứ hạng trong chính nguồn của nó**, ghi lại ngay từ R0 dù chỉ có một nguồn. RRF của R2 đọc nó.
- **`Context.untrusted` là một list riêng, không bao giờ là một cờ trên passage.**
- **`ref` là `DocRef | WebRef` ngay từ dòng code đầu tiên**, dù R0 chỉ bao giờ dựng `DocRef`.
- **UUID không bao giờ đi vào prompt.** Nguồn được đánh số `[1]`, `[2]`; code ánh xạ số đó ngược lại.
- **Chính sách khi hỏng:** node nào parse lỗi hoặc sai schema thì lui về no-op của chính nó. Mất một năng lực; không mất request.
- **Cờ giai đoạn mặc định tắt**, và `RETRIEVAL_MAX_ITERATIONS` mặc định `1`. Tắt hết cờ thì hệ thống phải chạy đúng như R0 — và phải có test chứng minh điều đó.
- **Mọi lần gọi LLM trong test đi qua `AsyncStubProvider`.** Suite chạy không cần API key và không tốn xu nào.
- **Gọi OpenAI thật là việc có chủ đích.** `.env` của máy dev đặt `LLM_PROVIDER=openai`, nên `POST /query` trên API đang chạy và `python -m eval.run` đều tốn tiền. Plan này chỉ có hai chỗ cần provider thật: mục 1 của Definition of Done và Task 12 Step 6. Dừng lại và hỏi trước khi chạy một trong hai.

---

## Chia R0 / R1

| | Task | Sản phẩm |
| --- | --- | --- |
| **R0** | 1 → 9 *(xong)* | `POST /query` trả câu trả lời có trích dẫn; đủ tám chỗ nối, năm chỗ là no-op |
| **R0** | 9b, 9c, 9d | Nhánh qua được CI của `main`; provider thật trả về đúng schema; một câu hỏi thật có trích dẫn chỉ đúng trang |
| **R1** | 10 → 12 | Mỗi truy vấn ghi một hàng trace; `eval/run.py` in `recall@10` và `MRR` trên bộ có nhãn |
| **Đóng** | 13 | `README.md` và spec mô tả đúng hệ thống đang có |
| **Riêng** | 14 | Khuyến nghị có dùng DeepSeek hay không, cho từng lời gọi LLM |

R0 xong khi một câu hỏi về một PDF đã nạp thật trả về kèm trích dẫn chỉ đúng trang. R1 xong khi bạn có **một con số nền được ghi lại** — thứ mà R2 đến R8 phải vượt qua.

---

## Cấu trúc file

```
app/core/                          ★ MỚI — đường đọc, chỉ phía API
  contracts.py                     Query, Candidate, Context, Answer, Trace, ref (Task 1)
  pipeline.py                      vòng lặp và tám chỗ nối (Task 8)
  deps.py                          ráp các stage từ cờ config (Task 8)
  retrievers/
    base.py                        Retriever Protocol (Task 5)
    vector.py                      R0 — embed, search, xếp hạng child (Task 5)
  stages/
    noops.py                       năm chỗ nối rỗng (Task 8)
  parent_expander.py               R0 — gom child về parent (Task 4b)
  context_builder.py               cộng dồn, bỏ trùng, ngân sách token, tách untrusted (Task 6)
  generator.py                     nguồn đánh số, ánh xạ trích dẫn (Task 7)
app/repositories/
  chunk_repository.py              ★ MỚI — tìm ở mức child + lấy parent (Task 4)
app/schemas/query.py               ★ MỚI — request/response (Task 9)
app/services/query_service.py      ★ MỚI — điều phối app/core/ (Task 9)
app/controllers/query_controller.py ★ MỚI — POST /query (Task 9)
app/models/query_trace.py          ★ MỚI — R1 (Task 10)
shared/llm.py                      SỬA — thêm AsyncLLMProvider cạnh bản sync (Task 2)
app/config.py                      SỬA — settings retrieval và cờ giai đoạn (Task 3)
README.md                          SỬA — viết lại theo hệ thống đang có (Task 13)
eval/
  dataset.yaml                     ★ MỚI — 20 + 8 mục có nhãn (Task 12)
  run.py                           ★ MỚI — mặc định chế độ retrieval-only (Task 12)
alembic/versions/                  ★ MỘT revision mới (Task 10)
tests/
  test_contracts.py                Task 1
  test_async_llm_provider.py       Task 2
  helpers.py                       dữ liệu seed dùng chung (Task 4)
  test_chunk_repository.py         Task 4
  test_parent_expander.py          Task 4b
  test_vector_retriever.py         Task 5
  test_context_builder.py          Task 6
  test_generator.py                Task 7
  test_pipeline.py                 Task 8  ← test quan trọng nhất của plan này
  test_query_api.py                Task 9
  test_query_trace.py              Task 10, 11
  test_eval_harness.py             Task 12
```

**`app/core/` không bao giờ import từ `worker/`, và `worker/` không bao giờ import từ `app/core/`.** Hai đường chỉ chia sẻ `app/models`, `app/config` và `shared/` — đúng ranh giới Phase 1 đã dựng.

---

# R0 — Đường nối và Basic RAG

## Task 1 → 9: đã xong

Chín task đầu đã được commit trên `phase-2`. Code và test của chúng nằm trong git và không chép lại ở đây. Bản plan đầy đủ từng bước của chín task này còn trong lịch sử: `git log -p -- docs/plans/phase-2-r0-r1.vi.md`.

| Task | Sản phẩm | Commit |
| --- | --- | --- |
| 1 | `app/core/contracts.py` | `feat(core): add the read path contracts` |
| 2 | `AsyncLLMProvider`, `AsyncStubProvider`, `AsyncOpenAIProvider` trong `shared/llm.py` | `feat(shared): add the async llm provider for the read path` |
| 3 | Settings `RETRIEVAL_*` và các cờ giai đoạn | `feat(config): add retrieval settings and stage flags` |
| 4, 5 | `ChunkRepository` và `VectorRetriever`: tìm và xếp hạng ở mức child | `feat(core): add the parent-collapsing retrieval query`, `feat(core): retrieve and rank children instead of parents` |
| 4b | `expand_parents` | `feat(core): add the parent expansion stage` |
| 6 | `build_context` | `feat(core): add the context builder` |
| 7 | `generate`, `map_citations` | `feat(core): add the answer generator and citation mapping` |
| 8 | `Stages`, `answer`, năm no-op, `build_stages` | `feat(core): add the retrieval pipeline and its no-op seams` |
| 9 | `POST /query` | `feat(api): add POST /query` |

**Interface mà các task còn lại dùng.** Chữ ký lấy từ code hiện tại, không phải từ plan cũ:

- `Answer(text, citations, trace)`; `Trace.nodes` là list `TraceNode(node, ms, detail)`, cùng `Trace.iterations` và `Trace.total_ms`
- `DocCitation(ordinal, document_id, filename, page_number, quote)`; `Candidate.ref` là `DocRef(document_id, parent_id, child_id, page_number, filename)`
- `VectorRetriever(session, provider).search(plan) -> list[Candidate]`, sâu `retrieval_over_fetch`, không cắt về `top_k`
- `build_stages(session, provider, settings) -> Stages` và `answer(query, stages, max_iterations) -> Answer` trong `app/core/pipeline.py`
- `QueryService(session, provider).ask(payload) -> QueryResponse`; provider tới qua `Depends(get_async_provider)`
- `AsyncStubProvider.script(schema, *responses)`; một schema chưa được kịch bản hoá thì `complete()` ném `KeyError`
- Trong test: `tests/helpers.seed_chunks(session)` gieo 4 parent, mỗi parent 3 child (child `i` nằm ở trang `i // 3 + 1`); fixture `query_client` trả về `(client, provider)`

---

## Task 9b: Format lại 14 file của đường đọc

CI trên `main` chạy `ruff format --check .` cho mọi PR. Chín task đầu được commit khi chưa có CI và không qua pre-commit hook (hook nằm ở `.git/hooks`, không đi theo clone), nên 14 file xuống dòng ở cột 88 trong khi project đặt `line-length = 100`. Không dòng logic nào đổi.

**Files:**
- Modify: `app/config.py`, `app/core/context_builder.py`, `app/core/generator.py`, `app/core/pipeline.py`, `app/core/retrievers/vector.py`, `app/services/query_service.py`
- Modify: `tests/test_async_llm_provider.py`, `tests/test_chunk_repository.py`, `tests/test_context_builder.py`, `tests/test_generator.py`, `tests/test_parent_expander.py`, `tests/test_pipeline.py`, `tests/test_query_api.py`, `tests/test_vector_retriever.py`

- [x] **Step 1: Xác nhận đỏ**

```bash
uv run ruff format --check .
```

Expected: `14 files would be reformatted`.

- [x] **Step 2: Format**

```bash
uv run ruff format .
```

- [x] **Step 3: Kiểm tra diff** — `git diff --stat` liệt kê đúng 14 file ở trên; đọc lướt diff, chỉ có gộp và tách dòng.

- [x] **Step 4: Chạy đủ ba kiểm tra của CI, rồi dừng**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest -q --ignore=tests/test_ocr_service.py
```

Expected: sạch, 304 test PASS. Cài lại pre-commit hook theo README mục "Linting" để chuyện này không lặp lại.

Commit message đề xuất: `style: format the read path to the project line length`

---

## Task 9c: `AsyncOpenAIProvider` ngang hàng với bản đồng bộ trên `main`

`complete()` hiện gọi JSON mode (`response_format={"type": "json_object"}`) và không đưa schema cho model. Với OpenAI thật, điều đó hỏng theo hai cách. JSON mode đòi prompt phải nhắc tới chữ "JSON", nếu không API trả 400, và prompt của generator không nhắc. Kể cả khi qua được, model không biết phải trả key `answer`, nên `schema.model_validate` fail. Không test nào bắt được, vì mọi test đi qua `AsyncStubProvider` và nhánh này chưa gọi provider thật lần nào.

`main` đã giải đúng bài này cho `enrich()` bằng structured outputs `strict`. Đường đọc làm theo, qua `chat.completions.parse` của SDK: nó tự dựng JSON schema `strict` từ model Pydantic, nên mọi schema của R4 → R7 đi cùng một đường mà không ai phải viết schema tay. Kèm hai chỗ lệch nhỏ: `reasoning_effort` chưa được truyền (mặc định của API là `medium`, và reasoning token tính tiền như output), và thiếu key thì lỗi hiện ra ở request đầu tiên thay vì lúc khởi tạo.

**Files:**
- Modify: `shared/llm.py` (class `AsyncOpenAIProvider`)
- Modify: `tests/test_async_llm_provider.py`

**Interfaces:**
- Consumes: `Settings.openai_api_key`, `Settings.openai_chat_model`, `Settings.openai_reasoning_effort`
- Produces: `AsyncOpenAIProvider.complete(messages, schema) -> T`, chữ ký không đổi; `RuntimeError` khi thiếu key, và khi model không trả về object nào

- [x] **Step 1: Viết test đỏ**

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

- [x] **Step 2: Chạy để xác nhận đỏ**

```bash
uv run pytest -q tests/test_async_llm_provider.py
```

Expected: 3 FAIL, 5 PASS. Hai test đầu với `AttributeError: ... has no attribute 'create'`; test thứ ba với `openai.OpenAIError: Missing credentials`, tức lỗi của SDK chứ chưa phải lỗi của ta.

- [x] **Step 3: Sửa provider**

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

`json` vẫn được nửa đồng bộ của file dùng; đừng xoá import đó.

- [x] **Step 4: Chạy test** — PASS, 8 test trong file này.

- [x] **Step 5: Suite đầy đủ, rồi dừng**

Task này không gọi OpenAI. Lần gọi thật đầu tiên là mục 1 của Definition of Done.

Commit message đề xuất: `fix(llm): bind read-path completions to their schema`

---

## Task 9d: Câu hỏi thật đầu tiên

Chín task của R0 chưa từng chạy với embedding thật, model thật hay corpus thật. Làm việc đó ngay bây giờ, trước khi R1 dựng cái thước đo lên trên: một lỗi ở đường đọc mà tới Task 12 mới lộ ra thì con số nền đo một hệ thống hỏng.

**Files:** không có. Task này chỉ chạy và ghi lại kết quả. Nếu lộ ra lỗi, phần sửa là một task riêng với test của nó.

- [ ] **Step 1: Hỏi trước khi chạy.** Task này gọi OpenAI: 3 lần embed và 3 lần chat (Global Constraints).

- [ ] **Step 2: Bật stack với code của nhánh này**

```bash
docker compose up -d --wait db rabbitmq redis minio
docker compose up -d --wait --force-recreate api
curl -s localhost:8000/openapi.json | grep -c '"/query"'
docker compose exec api sh -c 'echo $LLM_PROVIDER'
```

Expected: `1` (API đã có `/query`) rồi `openai`. Source được bind-mount và nhánh này không thêm dependency nào, nên tạo lại container là đủ, không cần build lại image.

- [ ] **Step 3: Hỏi ba câu**

```bash
ask() {
  curl -s -D - localhost:8000/query -H 'content-type: application/json' \
    -d "$(python3 -c 'import json, sys; print(json.dumps({"question": sys.argv[1]}))' "$1")"
}

ask "Người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu tiền?"
ask "Phương tiện thủy nội địa phải đáp ứng những điều kiện gì để được hoạt động?"
ask "Tỷ giá USD hôm nay là bao nhiêu?"
```

Hai câu đầu có đáp án trong corpus, ở hai tài liệu khác nhau. Câu thứ ba thì không.

- [ ] **Step 4: Kiểm từng câu trả lời**

| Kiểm | Đạt khi |
| --- | --- |
| HTTP | 200, envelope `{code, message, data}` |
| Ngôn ngữ | Trả lời bằng tiếng Việt |
| Trích dẫn | Mỗi ý trong câu trả lời mang `[n]`, và mỗi `[n]` có một mục trong `citations` |
| Đúng trang | Mở PDF ở `data/` (hoặc trình xem của FE) tới `filename` + `page_number` của từng trích dẫn: đoạn được dùng thật sự nằm ở trang đó |
| Câu ngoài corpus | Nói rõ nguồn không trả lời được; không bịa ra con số nào |
| Độ trễ | Ghi lại `latency_ms` của từng câu |

- [ ] **Step 5: Ghi kết quả, rồi dừng**

Báo cáo mỗi câu một dòng: câu hỏi, tài liệu và trang được trích, đúng hay sai, `latency_ms`. Câu nào hỏng thì kèm header `X-Request-ID` và `docker compose logs api` quanh request đó, rồi dừng: không sửa trong task này.

Kết quả đạt ở đây cũng là mục 1 của Definition of Done.

---

# R1 — Cái thước đo

## Task 10: Bảng `query_traces`

**Files:**
- Create: `app/models/query_trace.py`
- Modify: `app/models/__init__.py`
- Create: `alembic/versions/<hash>_query_traces.py` (sinh bằng CLI)
- Create: `tests/test_query_trace.py`

**Interfaces:**
- Produces: `QueryTrace(id, question, standalone, strategy, iterations, total_ms, total_tokens, citation_count, nodes: JSONB, created_at)`

- [x] **Step 1: Viết test đỏ**

```python
# tests/test_query_trace.py
import uuid

import pytest
from sqlalchemy import select

from app.models import QueryTrace

pytestmark = pytest.mark.asyncio


async def test_a_trace_row_stores_the_per_node_detail(db_session):
    """Không có phần bóc tách theo node, một câu trả lời tệ chỉ mô tả được là
    "nó tìm dở". Có nó thì bạn thấy chính xác: cổng đòi thêm một vòng, và
    rerank đã vứt mất parent đúng ở vị trí 11."""
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
    """`standalone` xuất hiện ở R4 và `strategy` ở R5. R1 không được đòi hỏi
    chúng, nếu không R1 không thể ship trước chúng."""
    trace = QueryTrace(question="q", iterations=1, total_ms=1, citation_count=0, nodes=[])
    db_session.add(trace)
    await db_session.flush()

    assert trace.standalone is None and trace.strategy is None
```

- [x] **Step 2: Chạy để xác nhận đỏ** — `ImportError: cannot import name 'QueryTrace'`

- [x] **Step 3: Viết model**

```python
# app/models/query_trace.py
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class QueryTrace(Base):
    """Một hàng cho mỗi câu hỏi đã trả lời. Đây là thứ R2 đến R8 lập luận dựa
    vào: không có nó, một hồi quy chỉ mô tả được chứ không định vị được."""

    __tablename__ = "query_traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # Ghi từ R4 trở đi; null trước khi node viết lại tồn tại.
    standalone: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Ghi từ R5 trở đi. Phân bố của cột này là cách duy nhất bắt được một router
    # đã âm thầm trở thành một hằng số.
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

- [x] **Step 4: Sinh migration, rồi cắt nó về đúng một bảng**

```bash
uv run alembic revision --autogenerate -m "query traces"
```

**Đừng `upgrade` ngay.** Trên schema của `main`, autogenerate nhìn thấy những thứ database tự giữ mà model cố ý không map, và đề xuất xoá hết:

- `ix_child_chunks_embedding_hnsw` — index mà mọi truy vấn của R0 dựa vào
- `ix_child_chunks_tsv` và cột `child_chunks.tsv` — nền của BM25 ở R2
- `ix_documents_active_status`, `ix_documents_title_fts`
- `ck_documents_status`

Không test nào bắt được nếu chúng lọt vào: mất index HNSW thì truy vấn vẫn trả đúng kết quả, chỉ chậm đi. Mở file vừa sinh và xoá mọi op không thuộc `query_traces`. Thứ còn lại: `down_revision = "4af5845c2429"`, `upgrade()` là đúng một lời gọi `op.create_table("query_traces", ...)` với mười cột của model, và `downgrade()` là đúng một lời gọi `op.drop_table("query_traces")`.

Rồi mới chạy, và xác nhận ba thứ kia còn nguyên:

```bash
uv run alembic upgrade head
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d child_chunks"' \
  | grep -E 'ix_child_chunks_embedding_hnsw|ix_child_chunks_tsv| tsv '
```

Expected: ba dòng, cột `tsv`, index `hnsw` và index `gin (tsv)`.

`uv run alembic check` vẫn sẽ báo các mục "removed" ở trên. Đó là trạng thái có sẵn của `main`, không phải việc của task này; nó chỉ không được nhắc tới `query_traces`.

- [x] **Step 5: Chạy test, rồi dừng** — PASS, 2 test.

Commit message đề xuất: `feat(db): add the query trace table`

---

## Task 11: Lưu trace cho mỗi truy vấn

**Files:**
- Modify: `app/services/query_service.py`
- Modify: `tests/test_query_trace.py`
- Modify: `tests/conftest.py`, `tests/test_query_api.py` (chuyển fixture)

**Interfaces:**
- Consumes: `QueryTrace`, `Answer.trace`
- Produces: một hàng `query_traces` cho mỗi `POST /query`

- [x] **Step 0: Đưa hai fixture của đường đọc về `tests/conftest.py`**

`query_client` và `seeded_corpus` hiện nằm trong `tests/test_query_api.py`, nên file test khác không thấy chúng. Fixture `client` của `conftest.py` không dùng thay được: nó không override `get_async_provider`, nên `POST /query` chạy tới một `AsyncStubProvider` chưa được kịch bản hoá, ném `KeyError`, và test đỏ vì lý do sai.

Cắt nguyên hai fixture đó sang cuối `tests/conftest.py`, mang theo các import chúng cần (`ASGITransport`, `AsyncClient`, `create_app`, `get_session`, `AsyncStubProvider`, `get_async_provider`, `seed_chunks`), rồi bỏ các import thừa khỏi `tests/test_query_api.py`.

```bash
uv run ruff check . && uv run pytest -q tests/test_query_api.py
```

Expected: sạch, 4 test PASS như trước.

- [x] **Step 1: Viết test đỏ**

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

- [x] **Step 2: Chạy để xác nhận đỏ** — test đầu: không có hàng nào được ghi. Hai test sau xanh ngay vì chưa có gì để hỏng; chúng chỉ có nghĩa sau Step 3, và test thứ ba là thứ sẽ đỏ nếu ai đó đổi savepoint về một `flush()` trần.

- [x] **Step 3: Nối vào service**

```python
# app/services/query_service.py — trong QueryService.ask, trước khi return

        await _persist_trace(self._session, payload.question, result)

# và ở mức module

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

- [x] **Step 4: Chạy test** — PASS, 5 test trong file này.

- [x] **Step 5: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(api): persist a trace row for every query`

---

## Task 12: `eval/dataset.yaml` và `eval/run.py`

Con số nền. Mọi bậc từ R2 trở đi phải vượt qua thứ mà cái này in ra hôm nay.

**Files:**
- Create: `eval/__init__.py`, `eval/dataset.yaml`, `eval/run.py`
- Create: `tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `ChunkRepository`, `AsyncLLMProvider`, `build_stages`
- Produces:
  - `load_dataset(path) -> Dataset` với `set_a: list[LabelledQuestion]`, `set_b: list[BehaviourQuestion]`
  - `recall_at_k(hits, gold, k) -> float`, `reciprocal_rank(hits, gold) -> float`
  - `python -m eval.run` → in `recall@10`, `MRR`, `n`; `--full` thêm độ chính xác trích dẫn và số vòng trung bình
  - `unreachable_labels(session, questions) -> list[str]`: id của các nhãn không lần truy hồi nào trúng được
  - `python -m eval.run --check-labels` → chỉ kiểm nhãn rồi dừng; không gọi provider nào

- [x] **Step 1: Thêm dependency**

```bash
uv add --dev pyyaml
```

- [x] **Step 2: Viết test đỏ**

```python
# tests/test_eval_harness.py
"""Các chỉ số chỉ là ba dòng số học và mỗi dòng đều dễ sai một cách tinh vi,
nên chúng được unit test tách khỏi database."""

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
    """Một chunk đúng ở vị trí 11 không nằm trong ngữ cảnh và không được
    tính."""
    hits = [_Hit("x", i) for i in range(1, 12)]

    assert recall_at_k(hits, gold=("x", 11), k=10) == 0.0


def test_reciprocal_rank_rewards_position():
    hits = [_Hit("a", 1), _Hit("a", 2), _Hit("a", 3)]

    assert reciprocal_rank(hits, gold=("a", 1)) == 1.0
    assert reciprocal_rank(hits, gold=("a", 2)) == 0.5
    assert reciprocal_rank(hits, gold=("a", 99)) == 0.0


def test_the_dataset_is_anchored_to_sha256_and_page_not_to_ids():
    """Chunk id được sinh lại sau mỗi lần chunk lại. Nhãn neo vào chúng sẽ chết
    ngay lần đầu bạn đổi chiến lược chunking -- mà đó là nút đầu tiên ai cũng
    vặn."""
    dataset = load_dataset("eval/dataset.yaml")

    for question in dataset.set_a:
        assert len(question.gold_sha256) == 64
        assert question.gold_page >= 1
        assert question.quote, "một đoạn trích làm nhãn sai lộ ra ngay khi nhìn"


def test_set_a_covers_every_stage_that_will_be_measured_against_it():
    """Một bậc không có câu hỏi nào chạm vào chế độ hỏng của nó thì không thể
    chứng minh là có tác dụng. Các con số này là những gì spec đã lập luận."""
    dataset = load_dataset("eval/dataset.yaml")
    kinds = [q.kind for q in dataset.set_a]

    assert len(dataset.set_a) == 20
    assert kinds.count("baseline") == 8
    assert kinds.count("code") == 4  # R2 -- vector trượt chuỗi chính xác
    assert kinds.count("followup") == 4  # R4 -- điểm 0 cho tới khi có viết lại
    assert kinds.count("summarize") == 4  # R5 -- vector search không làm được
    assert len(dataset.set_b) == 8
```

Bốn test trên là số học thuần. Test thứ năm cần database, vì nó kiểm chính điều kiện để một nhãn có nghĩa:

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

- [x] **Step 3: Chạy để xác nhận đỏ** — module chưa tồn tại.

- [x] **Step 4: Viết khung dataset**

Hai mươi mục ở Set A và tám ở Set B. Bốn mục dưới đây là hình dạng mẫu; hai
mươi tư mục còn lại được soạn ở Step 6. Một cái nhãn là hai khẳng định: đoạn
trích nằm ở trang này, và trang này trả lời câu hỏi này. Máy kiểm được khẳng
định thứ nhất; khẳng định thứ hai vẫn cần một người đọc và gật.

```yaml
# eval/dataset.yaml
# Neo vào (sha256, page): sống sót qua mọi lần chunk lại. Không bao giờ neo vào chunk id.
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

> **Các `gold_sha256` toàn số 0 là chỗ giữ chỗ cho lượt gán nhãn, và test ở
> Step 2 sẽ fail cho tới khi chúng được thay bằng hash thật.** Cái fail đó là
> cố ý: nó là thứ ngăn harness được tuyên bố hoàn thành với một dataset rỗng.

> **Bốn mục mẫu hỏi về hoá đơn điện tử; corpus thật thì không.** Chúng được viết
> trước khi có corpus và chỉ còn là hình dạng. Corpus trên `main` là mười tài
> liệu luật giao thông (đường bộ, đường thuỷ nội địa, hàng hải, hàng không) cùng
> một file tiếng Nhật ba trang; PDF gốc nằm ở `data/`. Cả 28 mục phải hỏi về
> corpus đó. Lấy hash và số trang để gán nhãn:

```bash
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
  "select sha256_hash, filename, page_count from documents where completed_at is not null order by filename"'
```

> `gold_page` là số trang 1-based, đúng thứ `child_chunks.page_number` lưu, và
> điểm được chấm bằng cách so khớp đúng số đó. Một đoạn trích nằm trong một
> child bắt đầu từ trang trước thì mang số của trang trước: nhãn như thế không
> bao giờ trúng được. `--check-labels` ở Step 5 tồn tại để bắt đúng chuyện này.

- [x] **Step 5: Viết runner**

```python
# eval/run.py
"""Mặc định chế độ retrieval-only: recall@10 và MRR không cần lần gọi chat nào,
chỉ một lần embed cho mỗi câu hỏi, nên chỉ số bạn kiểm tra sau mỗi lần vặn nút
gần như không tốn gì. `--full` chạy nguyên
đường ống và chỉ cần cho độ chính xác trích dẫn và tỉ lệ bịa.
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
    """Trang đúng có lọt vào ngữ cảnh không? Đây là trần cứng: một chunk không
    bao giờ tới nơi thì không prompt nào cứu được."""
    return float(any((h.sha256, h.page_number) == gold for h in hits[:k]))


def reciprocal_rank(hits, gold: tuple[str, int]) -> float:
    """Nó nằm ở hạng mấy? Rerank không đổi recall -- nó không mang gì mới vào
    -- nên đây là chỉ số cho thấy nó có xứng đáng được nuôi hay không."""
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
        # Nhãn neo vào (sha256, page); candidate mang document_id. Một lần tra
        # bắc cầu giữa hai thứ mà không phải mở rộng DocRef, vốn đã đóng băng.
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

- [x] **Step 6: Soạn nháp 28 mục từ corpus**

Agent soạn, bằng cách đọc `child_chunks.content` qua SQL. Không gọi OpenAI và **không chạy retrieval**: chọn câu hỏi từ kết quả tìm kiếm là lấy đầu ra của hệ thống làm đáp án cho chính nó.

| Quy tắc | Lý do |
| --- | --- |
| Đúng số lượng từng `kind` mà test ở Step 2 đòi: 8 `baseline`, 4 `code`, 4 `followup`, 4 `summarize`; Set B: 5 `outside`, 3 `web-tempting` | Mỗi bậc sau cần câu hỏi đo đúng điểm yếu của nó |
| 8 câu `baseline` trải trên ít nhất 6 tài liệu | Một bộ chỉ hỏi về một nghị định đo nghị định đó, không đo corpus |
| Câu hỏi viết như người dùng sẽ hỏi, không chép lời của chunk | Câu hỏi lặp lại nguyên văn chunk làm vector search trông giỏi hơn thực tế |
| `quote` là một đoạn nguyên văn, tối đa khoảng 15 từ, nằm gọn trong **một** child, không dính markup Markdown hay HTML | `--check-labels` so khớp trong từng child; một đoạn vắt qua hai child sẽ bị coi là không trúng được |
| `code`: câu hỏi chứa số hiệu văn bản, số điều khoản hoặc mã mẫu biểu | Đây là thứ vector hay trượt và R2 phải cứu |
| `followup`: có `turns`; `question` là lượt cuối và không tự đứng được | Điểm bằng 0 cho tới R4 là đúng thiết kế |
| `summarize`: `gold_page` là trang mang phạm vi điều chỉnh của văn bản | Vector search không làm được việc này cho tới R5 |

- [x] **Step 7: Cho máy kiểm nhãn**

```bash
uv run python -m eval.run --check-labels
```

Expected: `labels           20 reachable`. Lệnh này không gọi provider nào. Nhãn nào bị báo thì sửa `quote` hoặc `gold_page` cho tới khi sạch.

- [ ] **Step 8: Dừng, chờ người duyệt bộ câu hỏi**

Máy đã chứng minh đoạn trích nằm ở trang đó. Nó không chứng minh được trang đó **trả lời** câu hỏi, cũng không biết câu hỏi có giống thứ người dùng thật sẽ hỏi hay không. Người duyệt đọc cả 28 mục, sửa hoặc thay, rồi chạy lại Step 7. Sau lượt duyệt này bộ câu hỏi **đóng băng** (spec §8.4): thêm hay bớt một câu là vô hiệu mọi phép so sánh trước/sau.

- [ ] **Step 9: Ghi lại con số nền**

```bash
uv run python -m eval.run
```

Lệnh này **gọi OpenAI**: 20 lần embed, một cho mỗi câu hỏi Set A, vì corpus được embed bằng `text-embedding-3-large` và một vector stub so với nó chỉ là nhiễu. `--full` thêm 20 lần gọi chat. Hỏi trước khi chạy (Global Constraints). Với `LLM_PROVIDER=stub` runner từ chối chạy thay vì in ra một con số vô nghĩa.

**Ghi `recall@10` và `MRR` in ra được vào `docs/retrieval-design.md` §8 làm con
số nền của R1.** Một con số không được ghi lại thì không phải là con số nền.

- [ ] **Step 10: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(eval): add the retrieval evaluation harness`

---

## Task 13: README và spec mô tả đúng hệ thống đang có

`README.md` trên `main` vẫn là README của khung ban đầu: tên `rag-beginner`, câu "There is no RAG logic here yet", một bảng endpoint CRUD đã không còn tồn tại, và mục "Deliberate gaps" nói `embedding` luôn `NULL`. Phase 1 đã merge mà nó chưa được sửa. Task này viết lại cho đúng với code sau R1, và nằm sau cùng để không phải sửa hai lần. `docs/retrieval-design.md` cũng còn vài dữ kiện của thời trước Phase 1; chúng được sửa cùng lượt, vì plan này tuyên bố spec thắng khi hai bên mâu thuẫn.

**Files:**
- Modify: `README.md`
- Modify: `docs/retrieval-design.md`, `docs/system-design.md`

**Ràng buộc:**
- Giữ nguyên tên hai mục `## Linting` và `## Testing`: `.github/workflows/ci.yml` trỏ tới chúng bằng tên.
- README viết bằng tiếng Anh, như hiện tại.
- README là lối vào, không phải bản sao của thiết kế: mỗi mục tóm tắt rồi link sang `docs/system-design.md` hoặc `docs/retrieval-design.md`.
- Không chép giá trị nào từ `.env`. Bảng cấu hình lấy tên biến và mặc định từ `.env.example` và `app/config.py`.

- [x] **Step 1: Lấy sự thật từ code, không từ trí nhớ**

```bash
grep -n '@router\.' app/controllers/*.py                  # endpoints
grep -v '^#' .env.example | grep -v '^$' | cut -d= -f1    # configuration variables
grep -n '^  [a-z-]*:$' docker-compose.yml                 # services
```

- [x] **Step 2: Viết lại từng mục**

| Mục | Việc phải làm |
| --- | --- |
| Tiêu đề và mở đầu | Đổi `rag-beginner` thành `s1mpleRAG`. Hai, ba câu: nạp PDF, hỏi đáp có trích dẫn tới trang; gồm API, ba worker, OCR và FE. Bỏ câu "There is no RAG logic here yet". |
| *(mới)* Architecture | Năm stage S1 → S5 và đường đọc, mỗi thứ một đoạn ngắn. Bảng service của `docker-compose.yml` kèm vai trò từng cái: `db`, `rabbitmq`, `redis`, `minio`, `minio-init`, `chandra`, `api`, `worker-ocr`, `worker-cpu`, `worker-llm`. Link sang hai file thiết kế và `docs/diagrams/`. |
| Layout | Thêm `app/core/`, `worker/`, `shared/`, `web/`, `eval/`, `tests/`, `docs/`, `alembic/`. Giữ câu về luồng controller → service → repository → model. |
| *(mới)* Requirements | Docker và Compose, `uv`, `pnpm` cho `web/`. GPU NVIDIA 16 GB cho `chandra`, và không dùng chung GPU đó với việc khác. Trên WSL phải nới giới hạn RAM (link sang mục host RAM của `docs/system-design.md`). |
| Running locally (uv) | Giữ đoạn về `uv`. Sửa: API cần cả RabbitMQ, Redis và MinIO chứ không chỉ `db`. `LLM_PROVIDER=stub` là mặc định; `openai` thì tốn tiền. Thêm cách chạy `web/` (`pnpm install`, `pnpm dev`). |
| Running with Docker | `docker compose up -d --build` giờ kéo cả `chandra`: tải model và cần GPU. Thêm cách chạy không có GPU: bỏ `chandra` nhưng vẫn chạy `worker-ocr` bằng `--no-deps`, vì mọi tài liệu đều bắt đầu ở hàng đợi `ocr`. Nhắc: đổi `task_routes` thì phải restart `api`, vì bên publish mới là bên quyết định queue. |
| Response envelope | Giữ. Đổi ví dụ sang payload thật: `id` là UUID, ví dụ 404 lấy từ `/documents/{id}/status`. |
| Endpoints | Thay cả bảng bằng bảng bên dưới. |
| Linting | Giữ nguyên. Thêm một câu: CI chạy đúng hai lệnh này cho mọi PR vào `main`. |
| Testing | Sửa lệnh khởi động thành bốn dịch vụ, đúng như CI. Thêm: `tests/test_ocr_service.py` cần `chandra` và GPU; marker `real_llm` cần key và tốn tiền; mọi test khác dùng stub. |
| Migrations | Giữ. Thêm cảnh báo của Task 10 Step 4: autogenerate đề xuất xoá index HNSW, cột `tsv` cùng các index và constraint database tự giữ; phải xoá các op đó khỏi file sinh ra. |
| Configuration | Thay bảng bằng toàn bộ biến của `.env.example`, nhóm theo: database, app, RabbitMQ, Redis, OCR, MinIO, LLM, giới hạn, rate limit, retrieval. Giữ hai đoạn giải thích về `DB_*` và `DATABASE_URL`. |
| *(mới)* Evaluation | `uv run python -m eval.run` và `--full`; cần `LLM_PROVIDER=openai`; con số nền nằm ở `docs/retrieval-design.md` §8. |
| Deliberate gaps | Thay toàn bộ bằng các khoảng trống thật: chưa có xác thực; Redis và RabbitMQ không có volume; `locations` chưa được sinh ra; mọi cờ `RETRIEVAL_*_ENABLED` đang tắt vì R2 → R8 chưa làm; FE chưa nối với `POST /query`. |

Bảng endpoint, theo code hiện tại:

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | Liveness. |
| POST | `/documents/upload-url` | Presigned PUT cho một PDF; 409 nếu file có SHA-256 đó đã tồn tại. |
| POST | `/documents` | 202; đăng ký file đã upload và đưa nó vào pipeline. |
| GET | `/documents` | Danh sách có phân trang, kèm tiến độ đang chạy. |
| GET | `/documents/{id}/status` | Trạng thái, metadata và tiến độ của một tài liệu; 404 nếu không có. |
| GET | `/documents/{id}/file-url` | URL ngắn hạn (300 giây) để trình xem PDF tải file. |
| DELETE | `/documents/{id}` | 200 với `data: null`; 404 nếu không có. |
| POST | `/query` | Câu trả lời kèm trích dẫn. |

- [x] **Step 3: Kiểm tra README khớp code**

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

Expected: hai danh sách endpoint khớp nhau; không dòng `missing:` nào; lệnh grep cuối không ra gì.

- [ ] **Step 4: Chạy thử các lệnh README đưa ra** — mọi khối lệnh trong "Running", "Linting", "Testing" và "Migrations" được chạy một lần trên một clone sạch, hoặc trên máy dev với stack đã tắt. Lệnh nào không chạy được thì sửa README, không sửa code.

- [x] **Step 5: Sửa các dữ kiện lỗi thời trong spec**

Chỉ sửa dữ kiện. Không viết lại lập luận nào.

| File, chỗ | Đang ghi | Sửa thành |
| --- | --- | --- |
| `retrieval-design.md` §2 (quyết định 7), §5.4, §9 | reranker "same pattern as `docling`", `docling_service/` | `chandra` — container OCR hiện có; Docling đã bị thay ở Phase 1 |
| `retrieval-design.md` §6.1 | "`text-embedding-3-small` aligns Vietnamese and English weakly" | Model giờ là `text-embedding-3-large`. Thêm một câu ghi rõ: lập luận cho sub-query song ngữ dựa trên điểm yếu của `-small`, và phải được đo lại bằng bộ eval trước khi làm R4. **Không** xoá lập luận |
| `retrieval-design.md` §8 | chưa có con số nền | Đã ghi ở Task 12 Step 9; kiểm lại là có |
| `retrieval-design.md` §11 | "What Phase 1 must deliver", ở thì tương lai, trỏ tới `78bfb53` | Đổi thành đã giao: năm mục đều có trên `main`. Giữ bảng, vì nó vẫn nói R2 và R5 đứng trên cột nào |
| `retrieval-design.md` §12, dòng §2 quyết định 3 | "`text-embedding-3-small` … Model unchanged" | Phase 1 đã chuyển sang `-large` cắt 1536 chiều (MIRACL 54,9 so với 44,0); corpus hiện là tiếng Việt cùng một file tiếng Nhật |
| `retrieval-design.md` §14 | tách `shared/llm.py` "after Phase 1 merges" | Điều kiện đã thoả; việc tách vẫn chưa lên lịch |
| `system-design.md`, mục "Phase 2 — Retrieval (deferred)" | "deferred", liệt kê `retriever.py` | R0 đã có trên `phase-2`; trỏ sang `retrieval-design.md` và plan này |

```bash
grep -n -i 'docling\|embedding-3-small\|78bfb53' docs/retrieval-design.md
```

Expected: chỉ còn những dòng nói về lịch sử (bảng §12 kể lại quyết định cũ), không dòng nào mô tả hiện tại.

- [ ] **Step 6: Dừng và báo cáo**

Commit message đề xuất: `docs: describe the system as built`

---

## Task 14: Đánh giá DeepSeek cho các lời gọi LLM

Thêm ngày 2026-10-10 theo yêu cầu. Câu hỏi: có nên dùng DeepSeek trong hệ thống không, và cho lời gọi nào. Task này là nghiên cứu và một khuyến nghị; **không đổi code**. Nếu câu trả lời là có, việc thêm provider là một plan riêng. Task này không thuộc Definition of Done của R0 + R1.

**Files:** không có code. Kết quả ghi vào mục "Kết quả" ngay dưới.

Hệ thống gọi LLM ở năm chỗ, và mỗi chỗ đòi một thứ khác nhau:

| Lời gọi | Ở đâu | Đòi gì ở provider |
| --- | --- | --- |
| S3 enrich | worker, `OpenAIProvider.enrich` | Structured output `strict` với enum đóng; tiếng Việt; hàng nghìn chunk mỗi tài liệu; giới hạn tốc độ rõ ràng |
| S4 embed | worker, `OpenAIProvider.embed` | Một API embedding; đổi model hay số chiều là nạp lại toàn bộ corpus |
| Embed câu hỏi | API, `AsyncOpenAIProvider.embed_query` | Đúng model của S4 |
| Sinh câu trả lời | API, `AsyncOpenAIProvider.complete` | Trả về đúng schema Pydantic; độ trễ thấp; tiếng Việt |
| Các node quyết định R4 → R7 | API, chưa làm | Nhiều lời gọi nhỏ có schema cho mỗi câu hỏi: giá và độ trễ mỗi lời gọi là thứ quyết định |

- [x] **Step 1: Lấy dữ kiện của hôm nay.** Model, cửa sổ ngữ cảnh, giá (cache hit, cache miss, output), có API embedding hay không, hỗ trợ structured output tới mức nào (`json_schema` `strict` hay chỉ JSON mode), tương thích OpenAI SDK, giới hạn tốc độ, chính sách dữ liệu. Mọi con số kèm URL của trang đã đọc; thứ gì không kiểm được thì ghi "chưa kiểm".

- [x] **Step 2: Lấy giá hiện tại của hai model đang dùng** (`OPENAI_CHAT_MODEL` và `text-embedding-3-large`) để so.

- [x] **Step 3: Ước tính chi phí trên từng provider**, có phép tính: enrich một tài liệu 600 chunk; 1.000 câu hỏi chỉ có bước sinh; 1.000 câu hỏi khi đã có bốn node quyết định.

- [x] **Step 4: Ghi khuyến nghị cho từng lời gọi, rồi dừng.** Dùng, giữ OpenAI, hoặc không làm được; kèm rủi ro và những gì `shared/llm.py` phải đổi.

**Kết quả** (tra cứu ngày 2026-10-10, từ tài liệu và bảng giá chính thức của hai bên; chưa gọi thử API nào):

**Khuyến nghị: không chuyển sang DeepSeek, ở cả năm lời gọi.**

| Lời gọi | Kết luận | Lý do chính |
| --- | --- | --- |
| S3 enrich | Giữ OpenAI | Đắt hơn 1,3 đến 2,6 lần trên khối lượng này; Chat Completions của DeepSeek chỉ nhận `text` hoặc `json_object`, không nhận `json_schema` `strict` mà `_ENRICHED_SCHEMA` đang dùng |
| S4 embed, embed câu hỏi | Không làm được | Tài liệu API của DeepSeek không có endpoint embedding nào |
| Sinh câu trả lời | Giữ OpenAI | Đắt hơn 1,4 đến 2,8 lần; giờ hành chính ở Việt Nam rơi vào khung giá cao của DeepSeek; dữ liệu được xử lý và lưu ở Trung Quốc |
| Node quyết định R4 → R7 | Giữ OpenAI | Cùng chênh lệch giá; `chat.completions.parse()` (Task 9c) không chạy được với DeepSeek |

Giá cho mỗi 1 triệu token, theo trang giá của từng bên:

| Model | Input | Output |
| --- | --- | --- |
| `gpt-6-luna` (đang dùng) | $0,10 | $0,50 |
| `deepseek-flash`, giờ cao điểm | $0,30 | $1,20 |
| `deepseek-flash`, ngoài giờ cao điểm | $0,15 | $0,60 |
| `deepseek-v4-pro`, giờ cao điểm | $1,32 | $3,96 |

Ước tính, chưa tính cache:

| Việc | `gpt-6-luna` | `deepseek-flash` cao điểm | `deepseek-flash` ngoài cao điểm |
| --- | --- | --- | --- |
| Enrich một tài liệu 600 chunk (99K vào, 45K ra) | $0,032 | $0,084 | $0,042 |
| 1.000 câu hỏi, chỉ sinh câu trả lời (3,0M vào, 0,30M ra) | $0,45 | $1,26 | $0,63 |
| 1.000 câu hỏi khi có bốn node quyết định (6,2M vào, 0,62M ra) | $0,93 | $2,60 | $1,30 |

Giá output của DeepSeek, kể cả ngoài giờ cao điểm, vẫn cao hơn `gpt-6-luna`, nên nó chỉ rẻ hơn được khi phần lớn input là cache hit.

Chỗ duy nhất đáng thử là so chất lượng câu trả lời: trên các benchmark pháp lý tiếng Anh của Vals, `deepseek-flash` hơn `gpt-6-luna`. Đó chưa phải bằng chứng cho tiếng Việt; muốn biết thì đo bằng `eval.run --full` sau R1.

Chưa kiểm được: chất lượng tiếng Việt (không tìm thấy benchmark nào); độ trễ ở mức suy luận thấp; chính sách lưu giữ và huấn luyện trên dữ liệu gửi qua API (hai văn bản điều khoản của DeepSeek không nói); Responses API của DeepSeek có ép schema thật hay không.

Nguồn: `api-docs.deepseek.com` (các trang `quick_start/pricing`, `api/create-chat-completion`, `guides/json_mode`, `quick_start/rate_limit`), `developers.openai.com/api/docs/pricing`, `vals.ai`.

---

## Definition of Done — R0 + R1

1. `POST /query` trả lời một câu hỏi về một PDF đã nạp thật, kèm trích dẫn chỉ đúng tài liệu và một trang thật sự chứa đoạn được trích (Task 9d).
2. Đủ tám chỗ nối; năm chỗ là no-op; `tests/test_pipeline.py` chứng minh vòng lặp lặp được, vào lại ở `plan`, dùng truy vấn tinh chỉnh, và cộng dồn ngữ cảnh.
3. Một stage ném lỗi thì lui về no-op của nó và request vẫn thành công.
4. Mọi cờ giai đoạn đều tắt và `RETRIEVAL_MAX_ITERATIONS` bằng `1` trong `.env.example`.
5. `hnsw.ef_search` được đặt cho mỗi session và ít nhất gấp đôi `RETRIEVAL_OVER_FETCH`.
6. Không prompt nào chứa UUID; trích dẫn ma bị bỏ qua và ghi log.
7. Mỗi truy vấn ghi đúng một hàng `query_traces`, và một lần ghi trace hỏng không làm hỏng câu trả lời.
8. `eval/dataset.yaml` có 20 mục Set A có nhãn và 8 mục Set B, neo vào `(sha256, page)`; `--check-labels` báo cả 20 nhãn trúng được, và một người đã duyệt bộ câu hỏi.
9. `uv run python -m eval.run` in ra `recall@10` và `MRR`, và **các con số được ghi vào spec**.
10. Toàn bộ suite xanh; `ruff check .` và `ruff format --check .` sạch; CI của PR vào `main` xanh.
11. `worker/` không import gì từ `app/core/`, và `app/core/` không import gì từ `worker/`.
12. `AsyncOpenAIProvider` yêu cầu structured output theo schema, và một câu hỏi thật qua OpenAI trả về một `Draft` hợp lệ (đây là điều mục 1 ngầm đòi).
13. Migration của `query_traces` không xoá gì: index HNSW, cột `tsv` và index của nó còn nguyên sau `alembic upgrade head`.
14. `README.md` mô tả hệ thống đang có: không còn `rag-beginner`, bảng endpoint khớp code, bảng cấu hình khớp `.env.example`. `docs/retrieval-design.md` không còn mô tả hiện tại bằng `docling` hay `text-embedding-3-small`.
