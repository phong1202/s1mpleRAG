# Phase 3, R5: router và công cụ đọc dàn ý tài liệu

> **Cho người thực thi:** làm lần lượt từng task, mỗi task theo TDD (test đỏ → code → test xanh). Không tự commit; khi được bảo thì gộp thành vài commit lớn. Bước nào gọi OpenAI thật phải được đồng ý trước.

**Mục tiêu:** câu hỏi kiểu "Tóm tắt Luật Đường bộ" được trả lời từ dàn ý của chính văn bản đó (các chương, các điều), không phải từ 10 đoạn rời rạc mà tìm kiếm vector nhặt được. Spec §8.4 ghi đúng chỗ này: "vector search structurally cannot do this".

**Kiến trúc:** R0 đã để sẵn stage `plan` (no-op), `Plan.strategy` / `Plan.document_hint`, và `Source = "browse"`. R5 thay `plan` bằng router, thêm một retriever `BrowseRetriever` đọc `parent_chunks.heading_path`, và cho các stage sau giữ nguyên ứng viên "browse". `pipeline.py` không đổi. Cờ `RETRIEVAL_ROUTER_ENABLED` (đã có, mặc định `false`) bật toàn bộ.

**Spec:** `docs/retrieval-design.md` §6.2 (router), §3 (R5: "Strategy distribution + browse accuracy"), §8.7 (câu tóm tắt chưa đo được).

## Một quyết định lệch spec, cần chủ repo duyệt

Spec §6.2 cho router **một lần gọi model riêng**. Plan này **không thêm lần gọi nào**: router là code, đọc kết quả của bước viết lại ở R4, vốn đã phân loại câu hỏi (`SUMMARIZE_DOC` ở 3/4 câu tóm tắt của bộ eval). Bước viết lại chỉ cần thêm một trường: câu hỏi nhắc tới tài liệu nào (`document_hint`).

- Lý do: mỗi lần gọi model thêm khoảng 1,3 s cho mọi câu hỏi (số đo ở R4). Hai lần gọi chỉ để quyết hai việc liền nhau là trả tiền hai lần cho cùng một lần đọc câu hỏi.
- Đổi lại: router không quyết được những gì bước viết lại không thấy. Với tập chiến lược hiện có (`HYBRID`, `BROWSE_DOC`, `NO_RETRIEVAL`), mọi tín hiệu đều nằm trong intent và tên tài liệu.
- `VECTOR_ONLY` không dùng: không có tín hiệu nào cho thấy câu nào nên bỏ tìm từ khoá.

## Ràng buộc chung

- Tắt cờ: kết quả y hệt R4.
- Router không bao giờ làm hỏng request: không xác định được tài liệu thì quay về `HYBRID`, và trace ghi lý do.
- Trace ghi `strategy` của mọi câu, và cột `query_traces.strategy` (đã có từ R1) được điền. Spec §6.2: router lúc nào cũng trả `HYBRID` là lỗi chỉ thấy được qua phân bố này.
- Không gọi thêm model nào ngoài bước viết lại đã có.
- Bộ câu hỏi eval không đổi; chỉ thêm cách chấm. Không sửa `web/`; hợp đồng `POST /query` không đổi.

## Ngoài phạm vi

- **`METADATA`** ("thư viện có những tài liệu nào?"): FE đã hiện danh sách tài liệu, và bộ eval không có câu nào loại này. Ghi lại để làm sau.
- **Tóm tắt nhiều tài liệu một lúc** và so sánh hai tài liệu.
- **Chấm chất lượng bản tóm tắt** (cần một model làm giám khảo, thuộc R7).
- **R5.5** (agent ReAct cho công cụ đọc, spec §3): chỉ thử sau khi R5 có số đo.

## Các file

| File | Việc |
| --- | --- |
| `app/core/contracts.py` | `Query.document_hint` |
| `app/core/stages/rewrite.py` | `RewriteResult.document_hint`, prompt |
| `app/core/stages/route.py` | mới: router bằng code |
| `app/repositories/document_lookup.py` | mới: từ tên trong câu hỏi ra một tài liệu (chỉ mục `ix_documents_title_fts` có từ Phase 1) |
| `app/core/retrievers/browse.py` | mới: dàn ý tài liệu thành ứng viên |
| `app/core/deps.py`, `app/core/stages/fusion.py`, `app/core/parent_expander.py` | `BROWSE_DOC` chỉ chạy công cụ đọc; ứng viên "browse" đi qua trộn, rerank và mở rộng mà giữ nguyên |
| `app/services/query_service.py` | ghi `strategy` vào trace |
| `eval/run.py` | độ phủ dàn ý và độ chính xác chọn tài liệu |
| test tương ứng; tài liệu ở Task 7 | |

---

### Task 1: bước viết lại nói câu hỏi nhắc tới tài liệu nào

- `RewriteResult.document_hint: str`: tên, số hiệu hay chủ đề của **một** tài liệu mà câu hỏi nhắc tới, viết đúng như người dùng viết; không có thì chuỗi rỗng. (Structured outputs ở chế độ strict đòi đủ trường, nên dùng chuỗi rỗng thay cho `null`.)
- `Query.document_hint` mang nó tới router.
- [x] **Test đỏ:** "Tóm tắt Luật Đường bộ" → `document_hint == "Luật Đường bộ"` đi vào `Query`; câu không nhắc tài liệu → `None`; trace `rewrite_detail` có `document_hint`.
- [x] **Cài đặt, test xanh.**

### Task 2: router

| Intent | `document_hint` | Chiến lược |
| --- | --- | --- |
| `CHITCHAT` | — | `NO_RETRIEVAL` (như R4) |
| `SUMMARIZE_DOC` | có, hoặc client đã gửi `document_id` | `BROWSE_DOC` |
| mọi trường hợp khác | — | `HYBRID` |

- Trace: node `route_detail` với `strategy`, `document_hint`, lý do một câu.
- [x] **Test đỏ:** từng dòng của bảng; tóm tắt mà không biết tài liệu nào → `HYBRID` với lý do "no document named".
- [x] **Cài đặt, test xanh.**

### Task 3: từ tên trong câu hỏi ra một tài liệu

- Client đã gửi `document_id` (người dùng đang mở một tài liệu) thì dùng nó.
- Không thì tìm `document_hint` trên tiêu đề và tên file của các tài liệu đã nạp xong, bằng chỉ mục `ix_documents_title_fts`, ghép từ bằng OR như R2. Chỉ nhận khi tài liệu đứng đầu hơn hẳn tài liệu thứ hai; hai tài liệu sát điểm nhau (ví dụ "Nghị định 151" và "Nghị định 236 sửa đổi Nghị định 151") thì coi là không xác định được.
- [x] **Test đỏ:** "Luật Đường bộ" ra đúng luật, không ra "Luật Trật tự, an toàn giao thông đường bộ"; "Nghị định 168" ra NĐ 168, không ra NĐ 238 (sửa đổi NĐ 168); tên không khớp gì → `None`; hai ứng viên sát điểm → `None`.
- [x] **Đo trên CSDL thật (miễn phí):** với tên của mười tài liệu và các cách viết tắt hay gặp, ngưỡng "hơn hẳn" đặt bao nhiêu thì không nhầm.
- [x] **Cài đặt, test xanh.**

### Task 4: công cụ đọc dàn ý

Với một tài liệu, trả về theo thứ tự:

1. Một ứng viên mở đầu: tiêu đề tài liệu và parent đầu tiên (thường là phần căn cứ và điều về phạm vi điều chỉnh).
2. Mỗi mục cấp cao nhất (chương, hoặc phần tương đương) là một ứng viên: tên mục và danh sách tên các mục con (điều), không kèm nội dung. `ref` trỏ vào parent đầu tiên của mục đó, nên trích dẫn mở đúng trang bắt đầu của chương.

- Dàn ý của văn bản lớn nhất (NĐ 168: 70 đường dẫn tiêu đề) khi bỏ phần lặp của đường dẫn vẫn nằm trong ngân sách 8.000 token của context.
- Ứng viên `source="browse"` không bị RRF xếp lại, không bị reranker cắt, và không bị bước mở rộng parent thay nội dung.
- Tài liệu không có tiêu đề nào (ví dụ file tiếng Nhật 3 trang): chỉ trả ứng viên mở đầu.
- [x] **Test đỏ** trên dữ liệu seed có hai chương: đúng thứ tự, đúng tên mục con, `ref` đúng trang; đi qua `fuse` và `expand` mà không đổi; tài liệu không có tiêu đề.
- [x] **Nối vào `deps.py`:** `BROWSE_DOC` chạy công cụ đọc thay cho các retriever; không tìm được tài liệu thì quay về `HYBRID` và trace ghi lại.
- [x] **Cài đặt, test xanh; cả bộ suite xanh.**

### Task 5: eval chấm được câu tóm tắt

- **Độ phủ dàn ý** cho loại câu `summarize`: trong các mục cấp cao nhất của tài liệu đúng, bao nhiêu phần có mặt trong context. Đây là thước đo spec §8.7 đã nói R5 cần.
- **Chọn đúng tài liệu:** với câu đi theo `BROWSE_DOC`, tài liệu được chọn có phải tài liệu của nhãn không.
- **Phân bố chiến lược** trên Set A và Set B.
- [x] **Test đỏ, cài đặt, test xanh.**
- [x] **Đo mốc R4 (miễn phí, từ cache):** độ phủ dàn ý của 4 câu tóm tắt khi chưa có router.

### Task 6: đo

- [x] **(gọi OpenAI, cần đồng ý)** Prompt viết lại đổi nên cache cũ không dùng được: 28 lần gọi viết lại, khoảng 1 cent.
- [x] So với R4: `recall@10` và MRR theo loại câu (không loại nào được giảm), độ phủ dàn ý, chọn đúng tài liệu, phân bố chiến lược.
- [x] **(gọi OpenAI, cần đồng ý)** `--full` cho 4 câu tóm tắt, để đọc tận mắt câu trả lời: 4 lần gọi trả lời.

### Task 7: tài liệu

- [x] `docs/retrieval-design.md`: §6.2 ghi quyết định router bằng code; §8.7 thêm dòng R5 và thước đo mới.
- [x] `README.md`, `docs/system-design.md`.
- [x] Bản tiếng Anh của plan này.

## Điều kiện hoàn thành

1. Cả bộ test xanh, `ruff` sạch.
2. Tắt cờ: kết quả như R4.
3. Không xác định được tài liệu: câu hỏi vẫn được trả lời theo `HYBRID`, trace ghi lý do.
4. Độ phủ dàn ý của câu tóm tắt, độ chính xác chọn tài liệu và phân bố chiến lược được ghi vào spec §8.7 cạnh R4.

## Kết quả (2026-10-17)

Chủ repo duyệt router bằng code ("tạm thời theo m gợi ý đi").

| | `recall@10` | `MRR` | Độ phủ dàn ý (câu tóm tắt) | Chọn đúng tài liệu |
| --- | --- | --- | --- | --- |
| R4 | 0,900 | 0,580 | 0,440 | — |
| **R5** | **0,900** | **0,624** | **1,000** | **4/4** |

- Cả 4 câu tóm tắt đi `BROWSE_DOC` và tìm đúng văn bản theo tên câu hỏi dùng. MRR của loại tóm tắt từ 0,656 lên 0,875, vì dàn ý mở đầu bằng trang 1 của văn bản, cũng là trang vàng. Các loại câu khác không đổi.
- **Lần đo đầu:** a19 ("Nghị định về vận tải hàng không nói về những nội dung gì?") không được nhận là câu tóm tắt, độ phủ 0,857. Định nghĩa `SUMMARIZE_DOC` trong prompt được mở rộng thành "tóm tắt, dàn ý, hoặc tài liệu nói về những gì": một định nghĩa chung, không viết cho riêng câu này.
- **Đọc tận mắt 4 câu trả lời:** đúng chương mục, có trích dẫn từng chương; 2,2–4,8 s cho cả pipeline. Câu trả lời hay kết bằng một câu nói rằng nguồn chỉ có tên chương và điều. Đó là quy tắc "nói rõ khi nguồn không đủ" của bước trả lời, chưa chỉnh.
- **Set B:** router chọn `HYBRID` cho cả 8 câu. Từ khi prompt có thêm trường `document_hint`, bước viết lại chỉ đánh dấu 1/5 câu ngoài phạm vi là `OUT_OF_SCOPE` (trước đó 4/5). Intent này chưa được dùng trước R7; R7 phải đo lại.
- Thời gian bước viết lại: trung vị 1,4 s.
- Chi phí đo: 56 lần viết lại, 2 embedding, 4 lần trả lời.

Lệch so với plan:

- Không cần đặt ngưỡng tách tài liệu theo điểm số: tên được chấm bằng các từ đầu tiên của tiêu đề (văn bản tự gọi tên mình ở đầu tiêu đề), và từ trong tên mà tiêu đề không có bị phạt nặng (bình phương tỉ lệ khớp). Trên 10 tài liệu thật, mọi cách gọi đã thử đều ra đúng văn bản, và các tên mơ hồ không ra văn bản nào.
- Test giờ mặc định tắt mọi cờ stage, bất kể `.env` của máy dev bật gì. Lý do: khi `.env` bật rerank, test R2 gọi vào container rerank thật.
