# Phase 3, R4: hiểu câu hỏi nối tiếp (rewrite + intent)

> **Cho người thực thi:** làm lần lượt từng task, mỗi task theo TDD (test đỏ → code → test xanh). Không tự commit; khi được bảo thì gộp thành vài commit lớn. Bước nào gọi OpenAI thật phải được đồng ý trước.

**Mục tiêu:** câu hỏi nối tiếp ("Thế nếu người ngồi sau không đội thì sao?") được viết lại thành câu đứng độc lập trước khi tìm kiếm, nhờ lịch sử hội thoại mà server lưu. Đo bằng `recall@10` của loại câu `followup`, hiện 0.500 ở R3.

**Kiến trúc:** R0 đã để sẵn stage `rewrite` (no-op) và `Query.history`. R4 thay `rewrite` bằng một lần gọi `gpt-6-luna` trả về `RewriteResult` (spec §6.1), và thêm hai bảng `conversations` / `messages` để server tự giữ lịch sử (quyết định số 9 của spec). `pipeline.py` không đổi. Cờ `RETRIEVAL_INTENT_ENABLED` (đã có, mặc định `false`) bật toàn bộ.

**Spec:** `docs/retrieval-design.md` §6.1 (rewrite + intent), quyết định số 9 (lưu hội thoại phía server), §8.4 (câu follow-up).

## Ràng buộc chung

- Tắt cờ thì kết quả y hệt R3; hợp đồng `POST /query` chỉ **thêm** trường, không đổi trường cũ, nên FE hiện tại vẫn chạy.
- Rewrite lỗi (model trả rác, quá thời gian) thì dùng câu hỏi gốc: pipeline đã có fallback về no-op cho stage này.
- `standalone` chỉ được dùng từ ngữ của người dùng và của lịch sử; không thêm khái niệm mới (spec §6.1: over-broadening, hallucinated context).
- Mặc định **một** câu tìm kiếm; chỉ tách thành nhiều câu (tối đa 3) khi câu hỏi thật sự hỏi nhiều thứ.
- Bộ câu hỏi eval không đổi. Không sửa `web/`: phần FE đi qua tài liệu bàn giao.

## Ngoài phạm vi

- **Hành động theo intent ngoài `CHITCHAT`.** `SUMMARIZE_DOC` và `METADATA` thuộc R5 (router), `OUT_OF_SCOPE` thuộc R7 (từ chối). R4 chỉ ghi các intent này vào trace để R5, R7 đo trên dữ liệu thật.
- **Sub-query song ngữ.** Spec §6.1 yêu cầu đo recall chéo ngôn ngữ trước, nhưng bộ eval không có câu hỏi chéo ngôn ngữ nào. Chưa làm.
- **API đọc lại lịch sử hội thoại** (`GET /conversations/{id}`): FE đã tự giữ danh sách tin nhắn.
- **Lọc theo tài liệu do model đoán** (`RewriteResult.filters` trong spec): bộ lọc vẫn do giao diện gửi.

## Các file

| File | Việc |
| --- | --- |
| `app/core/contracts.py` | `Query.intent` |
| `app/core/stages/rewrite.py` | mới: `RewriteResult`, prompt, `make_rewrite(provider)` |
| `app/core/stages/noops.py` | `plan`: `CHITCHAT` → `NO_RETRIEVAL`, không tìm kiếm |
| `app/core/generator.py` | câu trả lời cho `CHITCHAT`: ngắn, không trích dẫn, không nêu dữ kiện |
| `app/core/deps.py` | nối `rewrite` theo cờ |
| `app/models/conversation.py`, `app/models/message.py`, `alembic/versions/…` | hai bảng mới |
| `app/repositories/conversation_repository.py` | tạo hội thoại, đọc N lượt gần nhất, ghi lượt hỏi và lượt trả lời |
| `app/schemas/query.py`, `app/services/query_service.py`, `app/controllers/query_controller.py` | `conversation_id` vào và ra; trace ghi `standalone` |
| `eval/run.py` | cache cho lần gọi rewrite; in phân bố intent |
| `docs/fe-query-handoff.md` | phần bàn giao cho session FE |
| test tương ứng | |

---

### Task 1: stage `rewrite`

**Tạo ra:**

```python
Intent = Literal["CHITCHAT", "SEARCH_CORPUS", "SUMMARIZE_DOC", "METADATA", "OUT_OF_SCOPE"]

class RewriteResult(BaseModel):
    intent: Intent
    standalone: str
    sub_queries: list[str] = Field(default_factory=list, max_length=3)

def make_rewrite(provider: AsyncLLMProvider) -> Callable[[Query, Trace], Awaitable[Query]]: ...
```

- Lời nhắn gửi model gồm tối đa 6 lượt gần nhất của lịch sử và câu hỏi hiện tại.
- Trả về `Query` với `text = standalone`, `sub_queries` chỉ khi có từ 2 câu trở lên, `intent`; giữ nguyên `filters`, `history`, `conversation_id`.
- `standalone` rỗng thì dùng câu gốc.
- Trace: node `rewrite_detail` với `intent`, `standalone`, `sub_queries`.
- `Query.refined()` giữ `intent`.

- [x] **Test đỏ** (`AsyncStubProvider.script(RewriteResult, …)`): câu nối tiếp thành câu đứng độc lập; lịch sử nằm trong lời nhắn gửi model, tối đa 6 lượt; một câu tìm kiếm thì `sub_queries` rỗng; `standalone` rỗng thì dùng câu gốc; model lỗi thì pipeline dùng câu gốc (fallback có sẵn) và trace ghi lý do.
- [x] **Cài đặt, test xanh.**

### Task 2: `CHITCHAT`

- `plan`: intent `CHITCHAT` → `Plan(strategy="NO_RETRIEVAL", queries=())`. Các retriever với danh sách câu rỗng trả `[]` mà không gọi embedding hay DB.
- `generate`: intent `CHITCHAT` → prompt riêng: trả lời ngắn, lịch sự, nói mình trả lời dựa trên tài liệu trong thư viện; không trích dẫn, không nêu dữ kiện.
- [x] **Test đỏ:** "cảm ơn" với intent `CHITCHAT` không gọi `embed_query`, không truy vấn DB, có câu trả lời, không có trích dẫn.
- [x] **Cài đặt, test xanh.**

### Task 3: lưu hội thoại phía server

- `conversations(id, created_at, updated_at)`; `messages(id, conversation_id → conversations, role, text, created_at)`, chỉ mục `(conversation_id, created_at)`. Migration chỉ tạo hai bảng.
- `POST /query` nhận `conversation_id` (không bắt buộc). Không có thì tạo hội thoại mới; có mà không tồn tại thì 404. Đọc 6 lượt gần nhất làm `Query.history`. Trả lời xong thì ghi lượt hỏi và lượt trả lời trong cùng transaction với trace.
- Request lỗi thì không ghi lượt nào: lịch sử chỉ có những gì người dùng đã nhận được.
- Response thêm `conversation_id`.
- Cờ tắt vẫn lưu hội thoại (rẻ, và FE được dùng `conversation_id` ngay), chỉ không đọc lịch sử để viết lại câu hỏi.

- [x] **Test đỏ:** câu đầu tạo hội thoại và trả `conversation_id`; câu thứ hai với `conversation_id` đó thì stage `rewrite` nhận được lịch sử; `conversation_id` lạ thì 404; request lỗi không ghi lượt nào; lịch sử dài thì chỉ lấy 6 lượt cuối, đúng thứ tự.
- [x] **Cài đặt, test xanh; cả bộ suite xanh.**

### Task 4: eval

- `--llm-cache PATH`: kết quả `complete()` lưu theo `(model, schema, nội dung lời nhắn)`, cùng cách với `--embed-cache`. Đo lại không tốn tiền khi câu và lịch sử không đổi.
- In phân bố intent trên Set A và Set B. Không câu nào của Set A được là `CHITCHAT`, vì như thế là bỏ qua tìm kiếm cho một câu hỏi thật.
- [x] **Test đỏ, cài đặt, test xanh.**

### Task 5: đo

- [x] **(gọi OpenAI, cần đồng ý)** R4: 28 lần rewrite (Set A và Set B) và embedding cho các câu đã viết lại, lưu cache. Vài cent.
- [x] So với R3 trên cùng corpus: `recall@10` và MRR theo loại câu. Mục tiêu là `followup` tăng mà các loại khác không giảm.
- [x] Thời gian thêm vào mỗi câu hỏi (node `rewrite` trong trace). Nếu quá lâu, phương án là bỏ rewrite ở câu đầu tiên của hội thoại; đo xong mới quyết.
- [x] Phân bố intent; liệt kê câu nào bị xếp sai.

### Task 6: tài liệu

- [x] `docs/retrieval-design.md` §8.7 thêm dòng R4; §6.1 ghi những gì R4 làm và chưa làm.
- [x] `docs/fe-query-handoff.md`: FE gửi lại `conversation_id` ở câu tiếp theo, "cuộc trò chuyện mới" thì bỏ nó đi; prompt mở đầu cho session FE.
- [x] `README.md`, `docs/system-design.md`.
- [x] Bản tiếng Anh của plan này.

## Điều kiện hoàn thành

1. Cả bộ test xanh, `ruff` sạch.
2. Tắt cờ: kết quả eval như R3; API chỉ thêm `conversation_id`.
3. Rewrite lỗi: câu trả lời vẫn có, dùng câu gốc, trace ghi lý do.
4. `recall@10` của `followup` ghi vào spec §8.7 cạnh R3, cùng thời gian thêm vào và phân bố intent.

## Trạng thái (2026-10-17)

Task 1 đến 4 xong: 461 test xanh, `ruff` sạch, CSDL dev đã chạy migration `351b23ad1fd7`. Task 5 chờ đồng ý gọi OpenAI. Task 6 còn dòng R4 ở spec §8.7 và bản tiếng Anh.

Lệch so với plan:

- `messages.id` là số tăng dần, không phải UUID: hai lượt của một lần hỏi được ghi trong cùng transaction, nơi `now()` giống nhau, nên thời gian không xếp được thứ tự.
- `query_traces` thêm cột `conversation_id`, để trace của một câu tìm được cuộc trò chuyện chứa nó.
- `RewriteResult` không có giá trị mặc định hay giới hạn độ dài trong schema (structured outputs ở chế độ strict đòi đủ mọi trường); giới hạn 3 câu tìm kiếm áp trong code.
- Trace của request lỗi chỉ gắn với cuộc trò chuyện mà client gửi lên; cuộc trò chuyện request đó vừa tạo có thể đã bị rollback cùng nó.

## Kết quả đo (2026-10-17)

Cùng corpus, cùng reranker; R4 bật thêm bước viết lại câu hỏi.

| | `recall@10` | `MRR` | Câu trượt |
| --- | --- | --- | --- |
| R3 | 0,850 | 0,583 | a01, a14, a16 |
| **R4** | **0,900** | **0,580** | a01, a15 |

- Cả bốn câu nối tiếp được viết lại đúng; 16 câu còn lại giữ nguyên từng chữ. Recall của câu nối tiếp từ 0,500 lên 0,750; các loại câu khác không đổi. MRR của câu nối tiếp vẫn thấp (0,237): trang đúng có trong top 10 nhưng không đứng đầu.
- a15 vẫn trượt: kết quả là trang 17 thay vì trang 18. Trang 17 cũng ghi phương tiện phải có giấy chứng nhận đăng ký, nên ít nhất một phần lỗi nằm ở nhãn (chủ repo đã chọn giữ nguyên nhãn này).
- Không câu nào của Set A bị coi là câu xã giao. 3/4 câu tóm tắt được xếp `SUMMARIZE_DOC`.
- **Thời gian:** ở `reasoning_effort=low` bước viết lại mất trung vị 1,9 s (tối đa 3,3 s). Ở `none`: 1,3 s (tối đa 1,8 s), cùng câu viết lại, cùng recall, và nhận ra 4/5 câu ngoài phạm vi của Set B thay vì 1/5. Vì vậy bước viết lại dùng `none` (`REWRITE_REASONING_EFFORT`), còn bước trả lời giữ `low`.
- Lượt đo tốn 4 + 2 embedding và 56 lần gọi model; kết quả đã lưu cache (`data/eval-embeddings.json`, `data/eval-llm.json`).

Lệch so với plan: thêm `REWRITE_REASONING_EFFORT`, và `AsyncLLMProvider.complete` nhận `reasoning_effort` cho từng lần gọi.
