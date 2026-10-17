# Bàn giao cho FE: nối khung chat với `POST /query`

Tài liệu này dành cho session làm frontend. Nó mô tả backend đang trả gì, FE đang có gì, và chỗ nào hai bên chưa khớp. Nó **không** phải plan: session FE tự lên plan từ đây, trình duyệt, rồi mới làm.

Viết ngày 2026-10-10, theo code của nhánh `phase-2` sau khi rebase lên `main`.

## 1. Bối cảnh

Phase 1 (nạp tài liệu) đã merge vào `main`. Phase 2 là đường đọc: nhận câu hỏi, tìm trong corpus, trả câu trả lời kèm trích dẫn tới trang. Backend của bậc đầu tiên (R0) đã có `POST /query`. FE đã có khung chat và trình xem PDF, nhưng khung chat chưa gọi backend: `web/hooks/use-chat.ts` chỉ ghi lại câu hỏi rồi báo `chat_unavailable`.

Việc của session FE: thay phần giả đó bằng lời gọi thật, hiển thị câu trả lời cùng nguồn, và cho người dùng bấm vào một trích dẫn để mở PDF tới đúng trang.

Ba điều cần biết trước khi lên plan:

- **`POST /query` mới chỉ có trên nhánh `phase-2`**, chưa có trên `main`. Hỏi người dùng nên cắt nhánh FE từ đâu.
- **Gọi thật hiện chưa chạy được.** Provider của đường đọc còn một lỗi (Task 9c trong `docs/plans/phase-2-r0-r1.vi.md`), và chưa có câu hỏi thật nào được chạy qua (Task 9d). Cho tới khi hai task đó xong, FE phát triển trên dữ liệu giả; xem mục 6.
- **Mỗi câu hỏi thật tốn tiền OpenAI** (một lần embed và một lần chat). Không gọi backend thật khi chưa hỏi người dùng.

## 2. Contract của backend

FE gọi qua rewrite sẵn có `/backend/*` (`web/next.config.mjs`), tức `POST /backend/query`. Cùng origin, nên không cần đổi CSP hay CORS.

**Request**

```json
{
  "question": "Vượt đèn đỏ bị phạt bao nhiêu?",
  "category": null,
  "document_id": null
}
```

| Trường | Kiểu | Ghi chú |
| --- | --- | --- |
| `question` | string, 1 → 2000 ký tự | Bắt buộc. Chuỗi toàn khoảng trắng bị từ chối. |
| `category` | string, tuỳ chọn | Lọc theo nhóm nội dung. FE chưa cần dùng. |
| `document_id` | UUID, tuỳ chọn | Chỉ tìm trong một tài liệu. Dùng được cho "hỏi về tài liệu đang mở". |
| `conversation_id` | UUID, tuỳ chọn | Có từ R4. Lấy từ response của câu trước để hỏi tiếp trong cùng cuộc trò chuyện. Không gửi thì backend mở cuộc trò chuyện mới. Gửi một id không tồn tại thì nhận 404. Xem mục 12. |

FE không gửi lịch sử: backend tự lưu các lượt hỏi đáp theo `conversation_id`.

**Response 200**

Envelope chung của mọi endpoint: `{code, message, data}`. Ví dụ dưới đây minh hoạ hình dạng, không phải kết quả của một lần chạy thật:

```json
{
  "code": 200,
  "message": "Success",
  "data": {
    "answer": "Người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt tiền … [1]. Ngoài ra còn bị trừ điểm giấy phép lái xe [3].",
    "citations": [
      {
        "ordinal": 1,
        "kind": "document",
        "document_id": "5a0c1c4e-0000-0000-0000-000000000000",
        "filename": "nd-168-2024_xu-phat-vi-pham-giao-thong-duong-bo.pdf",
        "page_number": 9,
        "quote": "…tối đa 280 ký tự đầu của đoạn nguồn…",
        "url": null,
        "title": null
      },
      {
        "ordinal": 3,
        "kind": "document",
        "document_id": "5a0c1c4e-0000-0000-0000-000000000000",
        "filename": "nd-168-2024_xu-phat-vi-pham-giao-thong-duong-bo.pdf",
        "page_number": 41,
        "quote": "…",
        "url": null,
        "title": null
      }
    ],
    "latency_ms": 4200
  }
}
```

Ý nghĩa từng trường, theo `app/schemas/query.py` và `app/core/generator.py`:

| Trường | Ý nghĩa |
| --- | --- |
| `answer` | Văn bản câu trả lời, cùng ngôn ngữ với câu hỏi. Chứa các dấu `[n]` trỏ tới nguồn. Model có thể trả Markdown. |
| `citations[].ordinal` | Chính là số `n` trong `[n]` của `answer`. |
| `citations` | Chỉ gồm những nguồn được trích trong `answer`, theo thứ tự xuất hiện lần đầu, mỗi nguồn một lần. Vì thế `ordinal` có thể **không liên tục** (ví dụ 1 rồi 3). Một `[n]` không có nguồn tương ứng đã bị backend bỏ khỏi danh sách, nhưng vẫn có thể còn trong văn bản. |
| `kind` | `"document"` hoặc `"web"`. Hiện tại luôn là `"document"`; `"web"` chỉ xuất hiện từ bậc R8. Với `"web"`, các trường tài liệu là `null` và `url`, `title` có giá trị. |
| `document_id` | UUID của tài liệu, dùng được ngay với trình xem PDF và với `/documents/{id}/file-url`. |
| `filename` | Tên file gốc lúc upload, không phải tiêu đề. Tiêu đề có sẵn ở danh sách tài liệu FE đã tải (`DocumentSummary.title`), tra theo `document_id`. |
| `page_number` | Số trang, tính từ 1, của đoạn con khớp nhất trong nguồn đó. |
| `quote` | Tối đa 280 ký tự đầu của đoạn nguồn. Đây là phần đầu đoạn, **không phải** đúng câu làm căn cứ. Văn bản lấy từ OCR nên có thể lẫn Markdown hoặc thẻ HTML của bảng: hiển thị như văn bản thuần, không chèn làm HTML. |
| `latency_ms` | Thời gian backend xử lý. |

**Lỗi**

| Trường hợp | Trả về |
| --- | --- |
| Câu hỏi rỗng hoặc dài quá 2000 ký tự | 422, `data.errors` là danh sách `{field, message}` |
| Provider lỗi, hết quota, bị giới hạn tốc độ | 500 với envelope chuẩn. Backend chưa phân biệt các trường hợp này. |
| Backend không chạy | Lỗi mạng ở rewrite; FE đã có khoá `backend_unavailable`. |
| Corpus không có câu trả lời | **Không phải lỗi**: 200, `answer` nói rõ là nguồn không trả lời được, `citations` thường rỗng. |

Mọi response mang header `X-Request-ID`. Nên giữ lại để báo lỗi.

**Tốc độ.** Không có streaming: một request, một response. Mỗi câu hỏi là một lần embed và một lần gọi model có suy luận, nên dự kiến mất vài giây. Con số thật chưa có; Task 9d sẽ đo. Cần kiểm xem rewrite của Next có giới hạn thời gian chờ không, và có cần nới không.

## 3. FE đang có gì

| File | Hiện trạng |
| --- | --- |
| `web/hooks/use-chat.ts` | Bản giả. `send()` thêm tin nhắn của người dùng rồi đặt lỗi `chat_unavailable`. Ghi chú trong file nói rõ: nối backend là viết lại `send()` và `stop()` ở đây; `ChatPanel` chỉ phụ thuộc vào hình dạng của hook. |
| `web/components/chat-panel.tsx` | Khung chat, ô nhập, nút dừng, nút "cuộc trò chuyện mới", các gợi ý câu hỏi. Dùng `status` gồm `ready`, `submitted`, `streaming`. |
| `web/components/chat-message.tsx` | Vẽ câu trả lời bằng `Streamdown` (Markdown) và danh sách nguồn `SourceList`. |
| `web/lib/types.ts` | `Message`, `SourceReference`, `SourceLocation`, `ViewerTarget`. `SourceReference` được ghi là **tạm**: "the UI's need, not a wire format". |
| `web/components/viewer/viewer-provider.tsx` | `useViewer().open(docId, target?)` với `target = { page, rects? }`. Mở tới một trang đã chạy được. |
| `web/lib/backend.ts` | `call()` bọc `fetch` tới `/backend/*`, đọc envelope, ném `BackendError` với khoá i18n. |
| `web/lib/i18n.ts` | Hai ngôn ngữ; đã có `errors.chat_unavailable`, `errorGeneric`, `sources`, `chunk`. |

## 4. Chỗ hai bên chưa khớp

`SourceReference` của FE được viết trước khi có backend. So với `citations[]`:

| FE đang cần | Backend đang trả | Nhận xét |
| --- | --- | --- |
| `documentId` | `document_id` | Khớp, khác tên. |
| `documentName` | `filename` | Có thể thay bằng `title` tra từ danh sách tài liệu. |
| `index` (số thứ tự chunk) | Không có | Thứ gần nhất là `ordinal`, và nó có nghĩa hơn: nó khớp với `[n]` trong câu trả lời. |
| `text` | `quote` (280 ký tự đầu) | Không phải toàn văn đoạn nguồn. |
| `score` (FE hiện phần trăm) | Không có | Backend không trả điểm. Điểm tương đồng vector không phải xác suất đúng, nên hiển thị thành phần trăm dễ gây hiểu sai. |
| `locations` (trang + vùng) | Chỉ có `page_number` | Vùng highlight chưa được sinh ra; xem mục 5. Mở tới trang thì làm được ngay. |

Vì kiểu của FE là tạm, hướng mặc định là **sửa kiểu của FE theo backend**, không yêu cầu backend đổi theo FE. Nếu FE thật sự cần thêm dữ liệu, xem mục 7.

Hai điểm khác cần quyết khi lên plan:

- **`[n]` trong câu trả lời.** `Streamdown` sẽ vẽ `[1]` như văn bản thường. Biến nó thành thứ bấm được (mở PDF tới trang của nguồn đó) là phần giá trị nhất của việc này.
- **Các gợi ý câu hỏi** hiện là "Tóm tắt các ý chính", "Kết luận chính là gì?". Tìm kiếm vector của R0 làm rất kém với câu hỏi kiểu tóm tắt cả tài liệu; việc đó thuộc bậc R5. Nên đổi sang gợi ý mà R0 trả lời được, hoặc bỏ.

## 5. Ngoài phạm vi

- ~~**Hội thoại nhiều lượt.**~~ Có từ R4: xem mục 12.
- **Streaming.** Trạng thái `streaming` của hook hiện không có gì để dùng: từ lúc gửi tới lúc có response là `submitted`.
- **Highlight vùng trong PDF.** Format `locations` đã chốt ở `docs/system-design.md` §5.3, nhưng backend chưa sinh ra và sẽ cần nạp lại corpus. Trình xem đã nhận `rects`; cứ để nguyên, chỉ truyền `page`.
- **Xác thực.** Chưa endpoint nào có.
- **Mọi thứ ngoài `web/`.** Session FE không sửa backend.

## 6. Phát triển mà không tốn tiền

Với `LLM_PROVIDER=stub`, `POST /query` trả **500**: provider giả của đường đọc cố ý không có câu trả lời mặc định. Với `LLM_PROVIDER=openai`, mỗi câu hỏi tốn tiền thật, và trước Task 9c thì còn lỗi.

Vì vậy FE cần một nguồn dữ liệu giả ở phía mình trong lúc phát triển, đặt ở ranh giới của `useChat` hoặc của `call()`. Cách làm cụ thể do session FE đề xuất trong plan. Ví dụ JSON ở mục 2 dùng làm fixture được, và nên có thêm ba biến thể: không có trích dẫn nào, `ordinal` không liên tục, và `quote` chứa Markdown hoặc thẻ HTML.

Lần kiểm với backend thật chỉ làm một lần ở cuối, sau khi người dùng đồng ý.

## 7. Nếu FE cần backend đổi

Ghi yêu cầu vào plan của FE và báo người dùng; session backend sẽ làm. Không tự vá ở phía FE cho một thiếu sót của backend.

Những thứ rẻ nếu cần: thêm `title` vào trích dẫn, trả toàn văn đoạn nguồn thay vì 280 ký tự, trả điểm thô. Thứ đáng cân nhắc nhất là một câu trả lời giả cho `/query` khi `LLM_PROVIDER=stub`, để FE và CI chạy được luồng thật mà không tốn tiền.

`locations` không rẻ: nó là một plan riêng.

## 8. Thế nào là xong

1. Gửi một câu hỏi thì hiện câu trả lời và danh sách nguồn lấy từ `POST /backend/query`.
2. Bấm vào một `[n]` trong câu trả lời, hoặc vào một nguồn, thì trình xem PDF mở đúng tài liệu tại đúng `page_number`.
3. Trong lúc chờ có trạng thái đang xử lý; nút dừng huỷ được request; "cuộc trò chuyện mới" xoá sạch.
4. Lỗi 422, 500 và mất kết nối đều hiện thông báo đã dịch ở cả hai ngôn ngữ; `chat_unavailable` không còn được dùng.
5. Câu trả lời không có trích dẫn nào vẫn hiển thị bình thường.
6. `quote` và `answer` không thể chèn HTML vào trang.
7. `pnpm build` xanh: CI chạy lệnh này cho mọi PR vào `main`.
8. Không file nào ngoài `web/` thay đổi.

## 9. Quy ước làm việc trong repo này

- Lên plan trước và chờ duyệt. Không tự commit, push hay merge; mỗi task một commit, và chỉ commit khi được bảo.
- Đọc `web/AGENTS.md` trước khi viết code: bản Next.js này khác với bản thường gặp, tài liệu đúng nằm ở `node_modules/next/dist/docs/`.
- Không gọi OpenAI khi chưa hỏi.
- Không in nội dung `.env`.

## 10. File nên đọc

- `web/hooks/use-chat.ts`, `web/components/chat-panel.tsx`, `web/components/chat-message.tsx`
- `web/lib/types.ts`, `web/lib/backend.ts`, `web/lib/i18n.ts`
- `web/components/viewer/viewer-provider.tsx`
- `app/schemas/query.py`, `app/controllers/query_controller.py` — contract, phía backend
- `app/core/generator.py` — cách `[n]` được ánh xạ thành `citations`
- `docs/system-design.md` §5.3 — format `locations`, để biết thứ sẽ tới sau
- `docs/plans/phase-2-r0-r1.vi.md`, mục "Đồng bộ với `main`" — trạng thái backend

## 11. Prompt mở đầu cho session FE

```
Đọc docs/fe-query-handoff.md rồi lên plan nối khung chat của web/ với POST /query.

Phạm vi: chỉ trong web/. Thay bản giả trong web/hooks/use-chat.ts bằng lời gọi thật
qua /backend/query, hiển thị câu trả lời và nguồn, và cho bấm vào trích dẫn [n] để
mở trình xem PDF tới đúng trang. Những thứ ngoài phạm vi nằm ở mục 5 của tài liệu.

Trước khi viết plan, đọc các file ở mục 10 và web/AGENTS.md. Trong plan, nêu rõ:
kiểu dữ liệu mới thay cho SourceReference, cách có dữ liệu giả để phát triển mà
không gọi OpenAI, cách xử lý [n] trong câu trả lời, và mọi thứ bạn cần backend đổi.

Lưu plan vào docs/plans/, trình mình duyệt trước khi làm. Không tự commit, push
hay merge. Không gọi backend thật khi chưa hỏi mình.
```

## 12. Cập nhật R4: hội thoại nhiều lượt

Backend giờ nhớ cuộc trò chuyện, và viết lại câu hỏi nối tiếp ("Thế nếu người ngồi sau không đội thì sao?") thành câu đứng độc lập trước khi tìm kiếm. Việc của FE chỉ là giữ và gửi lại một id.

**Contract thêm (chỉ thêm, không đổi trường cũ):**

| Ở đâu | Trường | Ghi chú |
| --- | --- | --- |
| request | `conversation_id` (UUID, tuỳ chọn) | Id nhận được ở câu trả lời trước. |
| response `data` | `conversation_id` (UUID, luôn có) | Id của cuộc trò chuyện chứa câu này. Câu đầu tiên không gửi id thì đây là id mới. |

**FE cần làm:**

1. Lưu `conversation_id` của response đầu tiên vào state của khung chat, và gửi nó kèm mọi câu hỏi sau.
2. Nút hoặc thao tác "cuộc trò chuyện mới": bỏ id đi; câu kế tiếp sẽ mở cuộc trò chuyện mới.
3. Response 404 khi gửi id (ví dụ id cũ còn trong state sau khi CSDL bị làm lại): bỏ id, gửi lại câu đó như câu mở đầu, không báo lỗi cho người dùng.
4. Câu hỏi lỗi (500, hết thời gian) không được backend ghi vào lịch sử. FE có thể cho hỏi lại bình thường với cùng id.
5. Có thể bỏ ghi chú "trợ lý không nhớ ngữ cảnh" nếu FE đang hiển thị.

**Hành vi người dùng sẽ thấy:**

- Câu xã giao như "cảm ơn", "xin chào": câu trả lời ngắn, không có nguồn trích dẫn. Đó là đúng: backend không tìm kiếm cho những câu này.
- Mỗi câu hỏi chậm hơn khoảng một lần gọi model (thời gian cụ thể sẽ ghi ở `docs/retrieval-design.md` §8.7 sau khi đo).

**Mock:** route mock (`QUERY_MOCK=1`) nên trả thêm `conversation_id` (một UUID cố định là đủ), để FE thử được luồng gửi lại id mà không tốn tiền.

**Prompt mở đầu cho session FE:**

> Đọc `docs/fe-query-handoff.md`, mục 12. Backend `POST /query` giờ nhận và trả `conversation_id`. Hãy cập nhật `web/` để giữ `conversation_id` trong state của khung chat, gửi lại ở mọi câu sau, bỏ nó khi bắt đầu cuộc trò chuyện mới, và khi nhận 404 thì bỏ id rồi gửi lại câu đó như câu mở đầu. Cập nhật mock để trả `conversation_id`. Chỉ sửa trong `web/`; không commit nếu chưa được bảo.
