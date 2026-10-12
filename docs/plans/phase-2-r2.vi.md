# Phase 2, R2: tìm kiếm lai (từ khoá + vector, trộn bằng RRF)

> **Cho người thực thi:** làm lần lượt từng task, mỗi task theo TDD (test đỏ → code → test xanh). Không tự commit: mỗi task chụp một snapshot, commit khi chủ repo bảo. Bước nào gọi OpenAI thật phải được đồng ý trước.

**Mục tiêu:** thêm một nguồn tìm theo từ khoá chạy cạnh tìm theo vector, trộn hai danh sách bằng RRF, rồi đo lại `recall@10` trên đúng bộ câu hỏi của R1.

**Kiến trúc:** R0 đã để sẵn hai chỗ nối: danh sách `retrievers` và hàm `fuse` trong `app/core/deps.py`. R2 thêm một retriever và thay `fuse`; `pipeline.py`, `contracts.py`, lược đồ CSDL và đường ghi không đổi. Cờ `RETRIEVAL_BM25_ENABLED` (đã có, mặc định `false`) bật cả hai cùng lúc; tắt cờ thì hệ thống chạy đúng như R1.

**Công nghệ:** Postgres full-text search (`child_chunks.tsv`, chỉ mục GIN `ix_child_chunks_tsv`, có từ Phase 1), SQLAlchemy async, pytest.

**Spec:** `docs/retrieval-design.md` §5.2 (tìm từ khoá), §5.3 (RRF), §8 (đo).

## Ràng buộc chung

- Không thêm cột, không nạp lại tài liệu. (Ban đầu ghi cả "không migration"; xem mục cách cắt mã số ở cuối.)
- Tắt cờ thì kết quả phải y hệt R1 đo trên cùng corpus: đó là điều kiện để so sánh trước/sau có nghĩa.
- Bộ câu hỏi `eval/dataset.yaml` không đổi (spec §8.4).
- Không sửa `web/`. Hợp đồng `POST /query` không đổi, nên phía FE không phải làm gì.
- Hai retriever dùng chung một `AsyncSession`: chạy lần lượt, không `gather`.

## Đã đo trước khi viết plan (SQL trên corpus thật, 2.982 chunk, không gọi OpenAI)

Chỉ dùng tìm từ khoá, chấm bằng nhãn của bộ eval (trang đúng nằm trong 10 kết quả đầu):

| Cách tìm | Trúng trong top 10 | MRR | Câu trả về rỗng |
| --- | --- | --- | --- |
| Câu SQL trong spec: `websearch_to_tsquery` (AND) + `ts_rank_cd` | 0/20 | 0.000 | 19 |
| Ghép từ bằng OR + `ts_rank_cd` | 9/20 | 0.119 | 0 |
| Ghép từ bằng OR + `ts_rank` | 12/20 | 0.334 | 0 |
| **Ghép từ bằng OR + `ts_rank(…, 1)`** (chia cho log độ dài) | **15/20** | **0.397** | 0 |
| Như trên, thêm tên văn bản và tiêu đề điều vào văn bản được tìm | 13/20 | 0.301 | 0 |
| Như trên, bỏ các từ có trong hơn 20% số chunk | 12/20 | 0.302 | 0 |

Ba điều rút ra, cả ba khác với spec:

1. **AND không dùng được cho câu hỏi tự nhiên.** Người dùng hỏi "thì bị xử phạt thế nào", và không chunk nào chứa đủ mọi từ đó. Phải ghép bằng OR.
2. **`ts_rank` tốt hơn `ts_rank_cd`** trên corpus này, ngược với lập luận ở spec §5.2.
3. **Thêm tiêu đề điều vào chỉ mục từ khoá không giúp gì** (13 so với 15, trong mức nhiễu của 20 câu). Vì vậy R2 không đổi lược đồ.

Tìm từ khoá tìm ra a08, a13, a20, là ba trong năm câu mà vector đang trượt (a08, a13, a14, a16, a20). Còn hiệu quả sau khi trộn thì phải chạy eval mới biết (Task 5).

Một câu truy vấn hết 7–13 ms. Nó khớp khoảng 70% số chunk vì ghép bằng OR, nên chi phí tăng tuyến tính theo corpus; ngưỡng cần xem lại là khi `retrieve` vượt 200 ms trong trace.

Bộ eval chỉ có 20 câu, nên bảng trên đủ để loại phương án hỏng (AND), không đủ để tinh chỉnh tham số. Plan không dò thêm tham số nào trên bộ này.

## Ngoài phạm vi

- **Sub-query song ngữ.** Bảng bậc thang ghi nó ở R2, nhưng spec §6.1 đặt nó trong nút rewrite, là một lần gọi LLM, thuộc R4. Corpus hiện là tiếng Việt cộng 6 chunk tiếng Nhật, và bộ eval không có câu hỏi chéo ngôn ngữ nào để đo. Chuyển sang R4.
- **BM25 thật** (`pg_search`): cần một image Postgres riêng. Chỉ xét nếu sau R2 các câu chứa mã số vẫn trượt.
- **Tiếng Nhật, tiếng Trung:** cấu hình `simple` tách từ theo khoảng trắng nên gần như không tìm được theo từ khoá. Các chunk đó vẫn tìm được bằng vector.
- Rerank (R3), rewrite (R4).

## Các file

| File | Việc |
| --- | --- |
| `app/repositories/chunk_repository.py` | thêm `KeywordHit`, `keyword_search()` |
| `app/core/retrievers/bm25.py` | mới: `BM25Retriever` |
| `app/core/stages/fusion.py` | mới: `reciprocal_rank_fusion()` (hàm thuần) và stage `fuse` |
| `app/core/deps.py` | nối retriever và `fuse` theo cờ; chạy các retriever lần lượt |
| `eval/run.py` | in ra các cờ đang bật cùng với kết quả |
| `tests/helpers.py` | thêm `seed_texts()` |
| `tests/test_chunk_repository.py`, `tests/test_bm25_retriever.py`, `tests/test_fusion.py`, `tests/test_hybrid_search.py` | test |
| `docs/retrieval-design.md`, `docs/system-design.md`, `README.md` | Task 6 |

---

### Task 1: tìm theo từ khoá trong repository

**Giao diện tạo ra:**

```python
@dataclass(frozen=True)
class KeywordHit:
    child_id: uuid.UUID
    parent_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    child_content: str
    score: float
    filename: str

class ChunkRepository:
    async def keyword_search(self, query: str, over_fetch: int, filters: Filters) -> list[KeywordHit]: ...
```

- [x] **Bước 1: thêm `seed_texts` vào `tests/helpers.py`**: một tài liệu, một parent, mỗi đoạn văn một child trên một trang riêng, nhận `language`.

- [x] **Bước 2: viết test đỏ** trong `tests/test_chunk_repository.py`:

| Test | Chứng minh |
| --- | --- |
| `test_keyword_search_finds_the_chunk_that_quotes_a_code` | hỏi `04/SS-HĐĐT`, chunk chứa đúng mã đó đứng đầu |
| `test_a_question_matches_without_every_word_being_present` | câu hỏi tự nhiên vẫn ra kết quả dù chunk chỉ chứa một từ của nó (OR, không phải AND) |
| `test_a_question_with_no_searchable_word_finds_nothing` | `"?!..."` trả về `[]`, không lỗi |
| `test_english_chunks_are_matched_through_english_stemming` | chunk `language="en"` chứa "vehicles", hỏi "vehicle examination" vẫn khớp |
| `test_keyword_search_honours_the_document_filter` | lọc `document_id` chỉ trả chunk của tài liệu đó |
| `test_keyword_hits_come_best_first_and_stop_at_over_fetch` | điểm giảm dần, số kết quả không vượt `over_fetch` |

- [x] **Bước 3: chạy, thấy đỏ** (`AttributeError: keyword_search`).

- [x] **Bước 4: cài đặt.** Câu SQL:

```sql
WITH q AS (
  SELECT (SELECT string_agg(quote_literal(lex), ' | ')::tsquery
          FROM   unnest(tsvector_to_array(to_tsvector('simple', :q))) AS lex) AS plain,
         (SELECT string_agg(quote_literal(lex), ' | ')::tsquery
          FROM   unnest(tsvector_to_array(to_tsvector('english', :q))) AS lex) AS english
), candidates AS (
  SELECT c.id, c.parent_id, c.document_id, c.page_number, c.content,
         ts_rank(c.tsv, CASE WHEN c.language = 'en' THEN q.english ELSE q.plain END, 1) AS score
  FROM   child_chunks c, q
  WHERE  ((c.language = 'en' AND c.tsv @@ q.english)
       OR (c.language IS DISTINCT FROM 'en' AND c.tsv @@ q.plain))
    AND  (CAST(:category AS text) IS NULL OR c.category = CAST(:category AS text))
    AND  (CAST(:document_id AS uuid) IS NULL OR c.document_id = CAST(:document_id AS uuid))
  ORDER  BY score DESC, c.id
  LIMIT  :over_fetch
)
SELECT c.id AS child_id, c.parent_id, c.document_id, c.page_number,
       c.content AS child_content, c.score, d.filename
FROM   candidates c
JOIN   documents d ON d.id = c.document_id
ORDER  BY c.score DESC, c.id
```

Vì sao viết như vậy:

- Câu hỏi được tách từ bằng chính `to_tsvector`, cùng cấu hình với cột `tsv`, nên từ trong câu hỏi và từ trong chỉ mục được chuẩn hoá giống nhau. Các từ được nối bằng `|`.
- Cột `tsv` dùng cấu hình `english` cho chunk tiếng Anh và `simple` cho phần còn lại, nên câu hỏi được dựng thành hai tsquery và mỗi dòng so với cái đúng cấu hình của nó. Hai nhánh `OR` viết tách ra để Postgres dùng được chỉ mục GIN cho cả hai.
- Câu hỏi không có từ nào thì `string_agg` trả `NULL`, điều kiện `@@` thành `NULL`, kết quả rỗng.
- `c.id` trong `ORDER BY` để hai chunk bằng điểm luôn ra cùng một thứ tự.

- [x] **Bước 5: chạy, thấy xanh.** `uv run pytest tests/test_chunk_repository.py -q`
- [x] **Bước 6: snapshot `r2-task-1`.**

### Task 2: `BM25Retriever`

**Dùng:** `ChunkRepository.keyword_search`. **Tạo ra:** `BM25Retriever(session)` với `source = "bm25"` và `async def search(plan: Plan) -> list[Candidate]`.

Tên giữ là `bm25` vì `Source`, cờ cấu hình và spec đã gọi như vậy từ R0; docstring nói rõ đây là `ts_rank` của Postgres, không có IDF.

- [x] **Bước 1: test đỏ** trong `tests/test_bm25_retriever.py`:

| Test | Chứng minh |
| --- | --- |
| `test_candidates_are_ranked_from_one_within_the_source` | `rank` chạy 1, 2, 3… không hở; `source == "bm25"` |
| `test_sub_queries_are_merged_without_duplicating_a_child` | hai sub-query cùng khớp một chunk thì chunk đó xuất hiện một lần, giữ điểm cao hơn |
| `test_candidate_carries_a_doc_ref_with_the_child_page` | `ref` là `DocRef`, `page_number` là trang của child |

- [x] **Bước 2: đỏ** (`ModuleNotFoundError`). **Bước 3: cài đặt**, cùng khuôn với `VectorRetriever`: gom theo `child_id`, giữ điểm cao nhất, sắp theo `(-score, child_id)`, đánh `rank` từ 1. **Bước 4: xanh.** **Bước 5: snapshot `r2-task-2`.**

### Task 3: RRF

**Tạo ra** trong `app/core/stages/fusion.py`:

```python
def reciprocal_rank_fusion(
    candidates: list[Candidate], k: int, top_k: int
) -> tuple[list[Candidate], dict[str, int]]:
    """Trả về danh sách đã trộn (đã cắt còn top_k) và bảng đếm nguồn của các
    ứng viên được giữ, ví dụ {"bm25+vector": 4, "vector": 5, "bm25": 1}."""

async def fuse(query: Query, candidates: list[Candidate], trace: Trace) -> list[Candidate]:
    """Stage: đọc k và top_k từ settings, ghi node `fuse_detail` vào trace."""
```

Quy tắc:

- Khoá của một ứng viên là `child_id` (với `WebRef` là `url`).
- Điểm = tổng `1 / (k + rank)` qua các nguồn có nó. Mỗi nguồn đã bảo đảm một chunk chỉ xuất hiện một lần (Task 2 và test sẵn có của `VectorRetriever`).
- Sắp theo điểm giảm dần; bằng điểm thì ứng viên có thứ hạng gốc tốt hơn đứng trước, rồi tới khoá. Thứ tự không phụ thuộc thứ tự đầu vào.
- Ứng viên trả về là bản từ nguồn xếp nó cao nhất, với `rank` mới từ 1 và `score` là điểm RRF.

- [x] **Bước 1: test đỏ** trong `tests/test_fusion.py`:

| Test | Chứng minh |
| --- | --- |
| `test_agreement_between_sources_beats_leading_one_list` | bảng trong spec §5.3: B (vector hạng 5, bm25 hạng 2) đứng trên A và C (mỗi cái đứng đầu một danh sách) |
| `test_fused_ranks_run_from_one_and_scores_are_the_rrf_sums` | `rank` 1…n; điểm của B bằng `1/65 + 1/62` |
| `test_the_list_is_cut_to_top_k` | 15 ứng viên, `top_k=10` → 10 |
| `test_one_source_alone_keeps_its_order` | chỉ có vector thì thứ tự không đổi |
| `test_the_order_does_not_depend_on_the_input_order` | đảo đầu vào, kết quả giống hệt |
| `test_the_summary_counts_where_the_kept_candidates_came_from` | bảng đếm đúng |

- [x] **Bước 2: đỏ. Bước 3: cài đặt. Bước 4: xanh. Bước 5: snapshot `r2-task-3`.**

### Task 4: nối vào `build_stages`

- [x] **Bước 1: test đỏ** trong `tests/test_hybrid_search.py`, qua `POST /query` với `query_client`:

| Test | Chứng minh |
| --- | --- |
| `test_with_the_flag_off_only_vectors_are_searched` | `retrieve_detail.per_source` chỉ có `vector`; không có node `fuse_detail` |
| `test_with_the_flag_on_both_sources_are_searched` | `per_source` có cả `vector` và `bm25`; có `fuse_detail` |
| `test_a_chunk_both_sources_agree_on_leads_the_context` | vector của stub không mang nghĩa; chunk duy nhất khớp từ khoá `04/SS-HĐĐT` được hai nguồn bầu nên đứng đầu `expand_detail` |

- [x] **Bước 2: đỏ.**
- [x] **Bước 3: sửa `app/core/deps.py`:**
  - `if settings.retrieval_bm25_enabled: retrievers.append(BM25Retriever(session))`
  - thay `asyncio.gather` bằng vòng lặp tuần tự, sửa chú thích cho đúng lý do (chung một session);
  - `fuse=fusion.fuse if settings.retrieval_bm25_enabled else noops.fuse`.
- [x] **Bước 4: xanh; chạy cả bộ test** (`uv run pytest --ignore=tests/test_ocr_service.py -q`), `ruff check`, `ruff format --check`.
- [x] **Bước 5: snapshot `r2-task-4`.**

### Task 5: đo

- [x] **Bước 1:** `eval/run.py` in thêm một dòng đầu kết quả, ví dụ `stages           bm25=on`, để con số ghi lại luôn đi kèm bậc thang đã đo. Test trong `tests/test_eval_harness.py`.
- [x] **Bước 2 (không gọi OpenAI):** `uv run python -m eval.run --check-labels` → `labels 20 reachable`.
- [x] **Bước 3 (gọi OpenAI: 20 embedding, cần chủ repo đồng ý):** `RETRIEVAL_BM25_ENABLED=true uv run python -m eval.run`. Số nền để so là R1 đo lại ngay trước đó trên cùng corpus (cờ tắt).
- [x] **Bước 4:** ghi kết quả theo từng loại câu hỏi và danh sách câu trượt. Kết quả nào cũng được ghi, kể cả khi không cải thiện. Nếu `recall@10` không tăng thì cờ giữ mặc định `false` và spec ghi lý do.

### Task 6: tài liệu

- [x] `docs/retrieval-design.md`: §5.2 viết lại (dòng kết quả ở §8.7 chờ Task 5) theo câu SQL thật và bảng đo ở trên (AND → OR, `ts_rank_cd` → `ts_rank`), §3 ghi sub-query song ngữ chuyển sang R4, §8.7 thêm dòng kết quả R2.
- [x] `docs/system-design.md` (mục Phase 2) và `README.md`: trạng thái R2, cách bật cờ.
- [x] Bản tiếng Anh của plan này: `docs/plans/phase-2-r2.en.md`.
- [x] Snapshot `r2-task-6`.

## Điều kiện hoàn thành

1. Cả bộ test xanh, `ruff check` và `ruff format --check` sạch.
2. Tắt cờ: không node nào trong trace khác R1.
3. Bật cờ: trace có `per_source` với hai nguồn và `fuse_detail`.
4. `recall@10` và MRR của R2 được ghi vào spec §8.7, theo từng loại câu hỏi, cạnh số của R1.
5. Spec §5.2 khớp với câu SQL đang chạy.

## Kết quả (2026-10-11)

Đo trên corpus vừa nạp lại (2.986 chunk), hai lần chạy liền nhau:

| | `recall@10` | `MRR` | Câu trượt |
| --- | --- | --- | --- |
| R1, cờ tắt | 0,700 | 0,463 | a06, a08, a10, a13, a14, a16 |
| R2, cờ bật | 0,750 | 0,542 | a08, a13, a14, a16, a18 |

Theo loại: câu chứa mã số từ 0,750 / 0,425 lên 1,000 / 0,750; câu thường từ 0,750 lên 0,875; câu nối tiếp giữ 0,250; câu tóm tắt từ 1,000 xuống 0,750 (mất a18).

Tổng thể hơn đúng một câu, nằm trong mức nhiễu của 20 câu. Chỗ R2 có tác dụng rõ là loại câu chứa mã số, đúng mục đích của nó. a08 và a13 được tìm từ khoá xếp hạng 5 và 6 nhưng vẫn rơi khỏi top 10 sau khi trộn: phần đó thuộc về rerank (R3).

Cờ vẫn mặc định `false`; bật hay không là quyết định của chủ repo.

Còn lại: code R2 chưa được review độc lập.

**Cách cắt mã số (ngoài plan ban đầu, chủ repo duyệt 2026-10-11).** Bộ tách từ của Postgres đọc `168/2024/NĐ-CP` như một đường dẫn file và cắt ở chữ không phải ASCII đầu tiên: chỉ mục chứa `168/2024/n`, `đ-cp`, `đ`, `cp`, nên hỏi "nghị định 168" không khớp. Migration `a0454bfe1fe8` dựng lại cột `tsv` trên `translate(content, '/-', '  ')`, và câu truy vấn áp cùng phép thay đó lên câu hỏi. Ràng buộc "không migration" ở đầu plan không còn đúng với phần này. Test `test_a_code_is_found_from_any_part_of_it` đỏ → xanh; cả bộ 398 xanh. CSDL dev đã chạy migration, không nạp lại.

Đổi lại là nhiễu: `2024` và `cp` có ở rất nhiều chunk; trong 10 kết quả đầu cho "168/2024" chỉ 4 kết quả trích đúng nghị định đó. Tìm từ khoá riêng lẻ trên bộ eval vẫn 15/20 trước và sau. Số R2 sau khi trộn được đo lại sau thay đổi này: vẫn 0,750 / 0,542, cùng danh sách câu trượt.

Cờ `RETRIEVAL_BM25_ENABLED=true` đã bật trong `.env` của máy dev (chủ repo yêu cầu); mặc định trong code và `.env.example` vẫn là `false`.

Nhánh: từ R2 trở đi làm trên `phase-3`, tách từ `phase-2`.
