# Phase 3: user mặc định và lịch sử hội thoại

> **Cho người thực thi:** làm lần lượt từng task, mỗi task theo TDD (test đỏ → code → test xanh). Không tự commit; khi được bảo thì gộp thành vài commit lớn. Không bước nào gọi OpenAI.

**Mục tiêu:** cuộc trò chuyện được lưu ở server, gắn với một người dùng, mở lại và xoá được, câu trả lời cũ giữ nguyên trích dẫn. Tạm thời mọi request là của một user admin tạo sẵn, cho tới khi có đăng nhập.

**Nguồn:** [`docs/be-conversations-handoff.md`](../be-conversations-handoff.md), do session FE viết theo các quyết định đã chốt với chủ repo. Contract ở mục 3.3 của tài liệu đó là thứ FE đang làm theo, nên plan này **không đổi contract**. Chỗ nào buộc phải đổi thì sửa tài liệu đó trước.

**Kiến trúc:** người dùng hiện tại đến từ một dependency `get_current_user`. Ở chế độ `AUTH_MODE=auto_admin` (mặc định, giá trị duy nhất lúc này), dependency trả về user admin có id cố định. Khi có đăng nhập, chỉ thân của dependency này đổi. Hội thoại và tin nhắn gắn với user; dữ liệu của user khác trả 404 như thể không tồn tại.

## Ràng buộc chung

- Hình dạng request và response của `POST /query` không đổi; mọi test hiện có của nó vẫn xanh.
- Câu hỏi lỗi vẫn không vào lịch sử (như R4).
- Trích dẫn được lưu đúng như `POST /query` đã trả (`CitationOut`, gồm `quote`), không lưu tham chiếu tới chunk: nạp lại corpus làm mọi chunk đổi id, còn tài liệu và số trang thì không.
- Tài liệu vẫn dùng chung cho mọi user (mục 3.4 của tài liệu bàn giao).
- `auto_admin` tương đương không có xác thực: log WARNING khi khởi động, và README ghi rõ không được mở API ra mạng ngoài.
- Không sửa `web/`.

## Ngoài phạm vi

Theo mục 5 của tài liệu bàn giao: đăng nhập, mật khẩu, token, phân quyền ngoài `is_admin`; tách tài liệu theo user; đổi tên, tìm kiếm, chia sẻ hội thoại; huỷ request khi client ngắt.

## Các file

| File | Việc |
| --- | --- |
| `app/models/user.py`, `app/models/conversation.py` | `User`; `Conversation.user_id`, `title`, `updated_at`; `Message.citations` |
| `alembic/versions/…_users_and_conversation_owners.py` | bảng `users`, gieo admin, cột mới, backfill hội thoại cũ về admin, index `(user_id, updated_at desc)` |
| `app/config.py`, `.env.example` | `AUTH_MODE` |
| `app/auth.py` | mới: id cố định của admin, `get_current_user` |
| `app/repositories/conversation_repository.py` | lọc theo user; tiêu đề, `updated_at`, trích dẫn; liệt kê, đọc, xoá |
| `app/schemas/conversation.py` | mới: `UserOut`, `ConversationSummary`, `ConversationDetail`, `MessageOut` |
| `app/controllers/conversation_controller.py` | mới: `GET /me`, `GET /conversations`, `GET /conversations/{id}`, `DELETE /conversations/{id}` |
| `app/services/query_service.py`, `app/controllers/query_controller.py` | `POST /query` nhận user, kiểm chủ sở hữu, ghi tiêu đề và trích dẫn |
| `app/app.py` | WARNING lúc khởi động khi `auto_admin` |
| `README.md` | bảng API, mục giới hạn |
| test tương ứng | |

---

### Task 1: user và tự xác thực

- Bảng `users` theo mục 3.1 của tài liệu bàn giao. Migration gieo admin với id cố định `00000000-0000-4000-8000-000000000001`, username `admin`, display name `Admin`, `is_admin = true`.
- `AUTH_MODE: Literal["auto_admin"] = "auto_admin"`.
- `get_current_user` trả về admin; admin không có trong DB (chưa migrate) thì 500 với thông báo rõ, không tự tạo.
- `GET /me`.
- [x] **Test đỏ:** `GET /me` trả đúng admin; xoá admin khỏi DB thì 500 có thông báo; khởi động app với `auto_admin` thì log một dòng WARNING.
- [x] **Cài đặt, test xanh.**

### Task 2: hội thoại gắn với user, có tiêu đề và trích dẫn

- `conversations.user_id` (not null, backfill hội thoại cũ về admin trong cùng migration), `title` (câu hỏi đầu tiên, cắt còn 120 ký tự), `updated_at`; `messages.citations` (jsonb, null).
- `POST /query`: hội thoại mới mang `user_id` của user hiện tại; `conversation_id` của user khác → 404 `Conversation not found`; lượt đầu đặt `title`; mỗi lượt cập nhật `updated_at` và lưu `citations` của câu trả lời (`[]` khi không có trích dẫn).
- [x] **Test đỏ:** hội thoại mới thuộc admin; lượt đầu đặt tiêu đề, lượt sau không đổi tiêu đề; `updated_at` tăng sau mỗi lượt; `citations` của tin nhắn trợ lý bằng đúng `citations` mà response đã trả; hội thoại của user khác → 404; test hiện có của `POST /query` vẫn xanh.
- [x] **Cài đặt, test xanh; migration chạy được trên CSDL dev (có sẵn hội thoại) và trên CSDL trống.**

### Task 3: đọc và xoá hội thoại

- `GET /conversations?limit&offset`: của user hiện tại, `updated_at` giảm dần, kèm `message_count`; `data` là `PaginatedData` như `GET /documents`.
- `GET /conversations/{id}`: toàn bộ tin nhắn, cũ trước.
- `DELETE /conversations/{id}`: xoá hội thoại và tin nhắn; trace còn lại với `conversation_id` null.
- Hội thoại của user khác: 404 ở cả ba.
- [x] **Test đỏ:** theo mục 7 của tài liệu bàn giao, ý 3 đến 6.
- [x] **Cài đặt, test xanh.**

### Task 4: tài liệu

- [x] `README.md`: bảng endpoint; mục giới hạn ghi rủi ro của `auto_admin`.
- [x] `docs/be-conversations-handoff.md`: ghi trạng thái "đã làm" và chỗ nào khác đề xuất (nếu có).
- [x] `docs/system-design.md`.
- [x] Bản tiếng Anh của plan này.

## Điều kiện hoàn thành

Mục 7 của tài liệu bàn giao, cả tám ý.

## Kết quả (2026-10-17)

Làm đúng contract của tài liệu bàn giao, không đổi chỗ nào. Cả 8 ý ở mục 7 có test. 525 test xanh, `ruff` sạch.

- **Migration `a2c10a08e2a2`** đã chạy trên CSDL dev. CSDL dev chưa có hội thoại nào, nên phần gán hội thoại cũ về admin được thử riêng: lùi migration, chèn một hội thoại mẫu có hai tin nhắn, chạy lại migration. Kết quả: hội thoại thuộc admin, tiêu đề là câu hỏi đầu, trích dẫn của tin nhắn cũ để trống. Dữ liệu mẫu đã xoá.
- **`updated_at` dùng `clock_timestamp()` chứ không dùng `now()`:** `now()` là thời điểm bắt đầu transaction, giống nhau cho mọi lượt ghi trong cùng một transaction.
- **Lệch so với plan:** một migration chung cho Task 1 và 2, để CSDL chỉ chạy một lần.
