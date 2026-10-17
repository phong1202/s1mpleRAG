# Phase 3, R8: tìm trên web khi tài liệu không đủ

> **Cho người thực thi:** làm lần lượt từng task, mỗi task theo TDD (test đỏ → code → test xanh). Không tự commit; khi được bảo thì gộp thành vài commit lớn. Bước nào gọi OpenAI hay Tavily thật phải được đồng ý trước.

**Mục tiêu:** khi gate (R6) đã tìm lại trong tài liệu mà vẫn thiếu, hệ thống tìm trên web bằng Tavily, đưa kết quả vào prompt trả lời như **nguồn ngoài, không tin cậy**, và trích dẫn chúng bằng `WebCitation`. Đo bằng độ chính xác của **quyết định ra ngoài** (spec §8.4): câu nghe như cần web nhưng tài liệu trả lời được thì không được ra web.

**Kiến trúc:** R0 đã có sẵn khung: `Candidate.source="web"`, `WebRef`, `WebPassage`, `Context.untrusted` (danh sách riêng, không bao giờ trộn với nguồn tài liệu), khối `[EXTERNAL SOURCES]` trong prompt trả lời, `WebCitation` và `CitationOut(kind="web")`; FE đã hiển thị trích dẫn web. R8 thêm `WebRetriever` (Tavily, gọi bằng `httpx`, không thêm SDK) và một bước `web` trong `gather()`, chạy **sau** vòng lặp.

**Spec:** `docs/retrieval-design.md` quyết định 13 và 14, §7 (ranh giới tin cậy), §8.1 (test bằng response ghi sẵn; chuỗi injection không bao giờ tới planner), §8.4 (Set B).

## Ràng buộc chung

- `WEB_SEARCH_ENABLED=false`: kết quả y hệt bây giờ.
- **Chỉ ra web khi:** cờ bật, gate nói "chưa đủ" ở vòng cuối, **và** đã có ít nhất một vòng tìm lại trong tài liệu (spec: quyết định 13, §7). Web là phương án cuối, không phải nguồn song song.
- **Ranh giới tin cậy, lớp 1 (kiến trúc):** kết quả web chỉ đi vào `build_context` → bước trả lời, và bước kiểm (R7) đọc chúng để đối chiếu. Không bao giờ quay lại rewrite, plan, gate. Bước web chạy sau vòng lặp nên không có đường nào để quay lại.
- Không dùng `include_answer` của Tavily (spec §7): câu trả lời do model của Tavily viết sẽ đi vòng qua bước trả lời, hệ trích dẫn và trace.
- Tavily lỗi, hết giờ, hết quota: không có kết quả web, câu hỏi vẫn được trả lời từ tài liệu (fallback no-op).
- Bộ câu hỏi eval không đổi. Không sửa `web/`; hợp đồng `POST /query` không đổi (`kind="web"` đã có từ R0).

## Ba việc phải quyết trước

1. **Web cần gate.** Theo spec, web chỉ chạy sau khi tìm lại trong tài liệu, tức là phải có `RETRIEVAL_MAX_ITERATIONS >= 2` và gate chạy ở mọi câu (khoảng 1,2 s mỗi câu, R6). Bật web mà để 1 vòng thì web không bao giờ chạy, nên cấu hình đó báo lỗi lúc khởi động thay vì im lặng không làm gì.
2. **Tìm trên web bằng câu nào.** Đề xuất: câu hỏi sau bước viết lại (R4), không phải câu tìm cuối của gate. Câu của gate được viết để tìm trong tài liệu, theo từ ngữ của tài liệu. Trace ghi lại câu đã dùng.
3. **Giới hạn tên miền.** `WEB_SEARCH_INCLUDE_DOMAINS` (danh sách, mặc định rỗng = mọi trang). Với văn bản pháp luật, giới hạn vào các trang chính thống giảm cả rủi ro injection lẫn nguồn sai, nhưng câu như "tỷ giá USD hôm nay" thì không tìm thấy ở đó. Quyết ở bước đo.

## Ngoài phạm vi

- Đọc toàn văn trang web (`include_raw_content`): chỉ dùng đoạn trích Tavily trả về.
- Lưu cache kết quả web giữa các request.
- Cho gate xét lại sau khi có kết quả web (web là bước cuối).

## Các file

| File | Việc |
| --- | --- |
| `app/core/retrievers/web.py` | mới: `WebRetriever` gọi Tavily `/search`, trả `Candidate(source="web")` |
| `app/core/pipeline.py`, `app/core/stages/noops.py` | `Stages.web`; bước web sau vòng lặp, theo đúng điều kiện bắn |
| `app/core/deps.py` | nối `web` theo cờ |
| `app/core/generator.py` | prompt: nguồn ngoài là trang web, không phải tài liệu của thư viện; dùng thì nói rõ |
| `app/core/stages/reflect.py`, `eval/judge.py` | đọc cả nguồn ngoài khi đối chiếu; câu đọc như mệnh lệnh thì không có nguồn đỡ (spec §7, lớp 3) |
| `app/config.py`, `.env.example` | `TAVILY_API_KEY`, `WEB_SEARCH_MAX_RESULTS`, `WEB_SEARCH_INCLUDE_DOMAINS`, `WEB_SEARCH_TIMEOUT_S`; kiểm cấu hình lúc khởi động |
| `eval/run.py` | câu nào ra web; Set B: quyết định ra web đúng hay sai |
| `tests/fixtures/tavily/*.json` | response Tavily ghi sẵn, trong đó có một trang chứa chuỗi injection |

---

### Task 1: `WebRetriever`

- `POST https://api.tavily.com/search`, header `Authorization: Bearer <key>`, body `{query, search_depth: "basic", max_results, include_answer: false, include_raw_content: false, include_domains?}`.
- Mỗi kết quả thành `Candidate(source="web", rank, score, text=content, ref=WebRef(url, title))`. Kết quả rỗng nội dung thì bỏ.
- [x] **Test đỏ** (`httpx.MockTransport` + response ghi sẵn): body gửi đi đúng và không bao giờ có `include_answer: true`; kết quả thành `Candidate` đúng thứ tự; giới hạn tên miền đi vào body khi có; HTTP lỗi hay hết giờ thì ném lỗi (để pipeline fallback).
- [x] **Cài đặt, test xanh.**

### Task 2: bước web trong pipeline

- `Stages.web(query, trace) -> list[Candidate]`, no-op trả `[]`.
- Trong `gather()`, sau vòng lặp: nếu gate vẫn "chưa đủ" và đã chạy ít nhất 2 vòng thì gọi `web` với câu hỏi đã viết lại, đưa kết quả qua `build_context` (vào `untrusted`). Gate không chạy lại.
- Bước trả lời: dòng `[NOT FOUND]` vẫn còn (tài liệu không có); prompt nói nguồn ngoài là trang web, dùng thì phải nói câu đó đến từ web.
- Trace: `web_detail` với câu đã tìm, số kết quả, các URL (không lưu nội dung).
- [x] **Test đỏ:** đủ ở vòng 1 → không gọi web; "chưa đủ" mà chỉ có 1 vòng → không gọi web; "chưa đủ" sau 2 vòng → gọi web đúng một lần, kết quả nằm trong `untrusted`, ordinal tiếp nối nguồn tài liệu; web lỗi → trả lời bằng tài liệu, trace ghi fallback; cờ tắt → như bây giờ.
- [x] **Test đỏ, ranh giới tin cậy:** một trang web chứa "ignore previous instructions…" → chuỗi đó chỉ xuất hiện trong lời gọi trả lời và lời gọi kiểm (R7), không bao giờ trong lời gọi rewrite, plan hay gate.
- [x] **Cài đặt, test xanh.**

### Task 3: nối cấu hình và bước kiểm

- `web = WebRetriever(...)` khi `WEB_SEARCH_ENABLED`, ngược lại no-op.
- Khởi động báo lỗi khi bật web mà thiếu `TAVILY_API_KEY`, hoặc `RETRIEVAL_MAX_ITERATIONS < 2`.
- Reflect (R7) và giám khảo của eval đọc cả nguồn ngoài; prompt reflect thêm: câu đọc như mệnh lệnh cho người đọc hay cho hệ thống thì không có nguồn đỡ.
- [x] **Test đỏ** qua `POST /query`: gate "chưa đủ" hai lần, web trả response ghi sẵn → response có trích dẫn `kind="web"` với `url`, `title`; tin nhắn lưu trong hội thoại giữ trích dẫn đó; trace có `web_detail`. Cấu hình sai thì `Settings()` báo lỗi rõ.
- [x] **Cài đặt, test xanh; cả bộ suite xanh.**

### Task 4: eval

- In: câu nào ra web (Set A và Set B), với câu tìm và số kết quả.
- Set B: câu nghe như cần web (b06–b08) mà ra web là **sai**; câu ngoài phạm vi (b01–b05) ra web là đúng hướng. Set A ra web: in ra để đọc tay (đáng lẽ gần như không có).
- Cache kết quả Tavily trong eval như cache embedding, để đo lại không tốn credit.
- [x] **Test đỏ, cài đặt, test xanh.**

### Task 5: đo

- [x] **(gọi OpenAI và Tavily, cần đồng ý và cần `TAVILY_API_KEY`)** `--full` với `RETRIEVAL_MAX_ITERATIONS=3`, `WEB_SEARCH_ENABLED=true`. Tavily chỉ bị gọi ở câu nào gate "chưa đủ" sau khi tìm lại, mỗi câu tối đa một lần (gói miễn phí của Tavily đủ). OpenAI: gate cho mọi câu, trả lời, kiểm, chấm; phần lớn đã có trong cache từ R6 và R7.
- [x] Đọc tay mọi câu ra web: có nên ra không, trang tìm được có liên quan không, câu trả lời có nói rõ phần nào lấy từ web không.
- [x] Thử có và không có giới hạn tên miền; chốt mặc định.

### Task 6: tài liệu

- [x] `docs/retrieval-design.md` §7 và §8.7.
- [x] `README.md`, `docs/system-design.md`.
- [x] Bản tiếng Anh của plan này.

## Điều kiện hoàn thành

1. Cả bộ test xanh, `ruff` sạch.
2. `WEB_SEARCH_ENABLED=false`: như bây giờ.
3. Có test chứng minh nội dung web không bao giờ tới rewrite, plan, gate.
4. Tavily lỗi: câu hỏi vẫn được trả lời từ tài liệu.
5. Quyết định ra web của Set B và số câu Set A ra web được ghi vào spec §8.7.

## Kết quả (2026-10-17)

Chạy `--full` trên 28 câu, `WEB_SEARCH_ENABLED=true`, `RETRIEVAL_MAX_ITERATIONS=3`, không giới hạn tên miền (bạn chọn: câu ngoài phạm vi thì tìm trên web).

| | Ra web | Đánh giá |
| --- | --- | --- |
| Set B, ngoài phạm vi (b01–b05) | 5/5 | Đúng: cả 5 trả lời từ web, có trích dẫn, nói rõ lấy từ web |
| Set B, nghe như cần web nhưng tài liệu có (b06–b08) | 1/3 (b06) | Sai theo spec |
| Set A | 1/20 (a01) | Cùng nguyên nhân với b06 |
| Set B đúng (từ chối, hoặc trả lời từ web có trích dẫn) | 8/8 | R7: 7/8 |

- **b06 và a01 ra web vì phần tìm kiếm**, không phải vì gate: cả hai hỏi mức phạt vượt đèn đỏ, trang đó bị reranker CPU đẩy xuống từ R3. Web trả về đúng 18–20 triệu đồng với ô tô theo Nghị định 168, nhưng lấy từ web chứ không từ tài liệu.
- **Ba lỗi tìm ra và đã sửa trong lúc đo:**
  - Lượt đầu, bước trả lời từ chối cả 7 câu dù trang web đã trả lời: prompt bảo "chỉ dùng các nguồn" cộng dòng `[NOT FOUND]` bảo "nói rõ tài liệu không có". Prompt giờ coi trang web là nguồn, ưu tiên tài liệu, và buộc nói rõ phần nào lấy từ web.
  - b04 ("lương tối thiểu vùng I hiện nay") trả lời mức năm 2025, trong khi trang tìm được ghi cả mức từ 1/1/2026: model không biết hôm nay là ngày nào. Khi có nguồn web, prompt giờ có dòng `[TODAY]`. Sau khi sửa: 5.310.000 đồng từ 01/01/2026.
  - Một câu trả lời kết thúc bằng chính nhãn `[NOT FOUND]` của prompt. Các nhãn của prompt giờ bị xoá khỏi câu trả lời.
- **Đọc tay 7 câu ra web:** trang tìm được liên quan ở cả 7. b05 (tàu hỏa) chỉ có một mức phạt, và câu trả lời nói rõ là nguồn chưa đủ. b01 (tỷ giá) nói rõ nguồn không ghi ngày cập nhật.
- **Giám khảo:** các câu còn bị gắn cờ sau khi kiểm là câu nói nguồn không có gì đó (không phải khẳng định) và một mẩu câu ở a18; không có câu bịa thật.
- **Chi phí đo:** 7 lượt tìm Tavily (lưu ở `data/eval-web.json`), khoảng 160 lần gọi OpenAI qua bốn lượt.
- **Kết luận:** `WEB_SEARCH_ENABLED` mặc định tắt trong code như mọi cờ. `.env` của chủ repo bật web với `RETRIEVAL_MAX_ITERATIONS=3`: mỗi câu thêm một lần gọi gate (khoảng 1,2 s), câu ra web thêm hai vòng tìm và một lượt Tavily.
- **Lệch so với plan:** `Stages.web` là `None` khi tắt (để không có dòng trace thừa); `build_stages` nhận thêm `searcher` để eval thay Tavily bằng cache; prompt trả lời thêm dòng về nguồn web, dòng `[TODAY]` và việc xoá nhãn; eval chấm Set B câu ngoài phạm vi trả lời từ web có trích dẫn là đúng.
