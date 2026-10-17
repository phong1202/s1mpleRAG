# Phase 3, R6: kiểm tra đủ hay chưa, thiếu thì tìm lại

> **Cho người thực thi:** làm lần lượt từng task, mỗi task theo TDD (test đỏ → code → test xanh). Không tự commit; khi được bảo thì gộp thành vài commit lớn. Bước nào gọi OpenAI thật phải được đồng ý trước.

**Mục tiêu:** sau khi tìm, một lần gọi model xét xem các đoạn đã có có trả lời được phần chính của câu hỏi không. Chưa đủ thì model nói thiếu gì và viết câu tìm mới; pipeline tìm thêm, cộng dồn vào context, tối đa 3 vòng. Spec gọi đây là chỗ hệ thống bắt đầu thành agentic.

**Kiến trúc:** vòng lặp đã có trong `pipeline.py` từ R0 (`max_iterations`, context cộng dồn, có test giữ nó sống). R6 thay stage `gate` (đang là no-op "luôn đủ") bằng một lần gọi `gpt-6-luna`, và nâng `RETRIEVAL_MAX_ITERATIONS` từ 1 lên 3. Gate chỉ chạy khi `RETRIEVAL_MAX_ITERATIONS > 1`; không thêm cờ mới.

**Spec:** `docs/retrieval-design.md` §6.3 (gate), §4.3 (vòng lặp), quyết định số 12 (context cộng dồn), §8.2 (mean iterations, mục tiêu 1,2–1,6).

## Ràng buộc chung

- `RETRIEVAL_MAX_ITERATIONS=1`: kết quả y hệt R5.
- Gate lỗi thì coi là "đủ" và trả lời với những gì đã có (fallback có sẵn trong pipeline). Hết vòng mà vẫn thiếu thì **vẫn trả lời**, nói rõ tài liệu có thể chưa đủ; không bao giờ trả lỗi vì tìm không đủ.
- Prompt của gate có câu cân chỉnh của spec §6.3: chỉ "chưa đủ" khi phần thiếu làm câu trả lời **sai hoặc gây hiểu lầm**. Thiếu câu này, gate gần như lúc nào cũng nói "chưa đủ" và mọi câu hỏi tốn 3 vòng.
- Bộ câu hỏi eval không đổi. Không sửa `web/`; hợp đồng `POST /query` không đổi.

## Ba việc phải sửa hoặc quyết trước

1. **Pipeline đang trả lời câu tìm kiếm thay cho câu hỏi.** Khi gate nói "chưa đủ", `query` bị thay bằng câu tìm mới, và bước trả lời nhận đúng câu đó. Người dùng hỏi "phạt bao nhiêu" thì model lại trả lời một câu tìm kiếm do gate viết. R6 sửa: câu tìm mới chỉ dùng để tìm; bước trả lời luôn nhận câu hỏi sau bước viết lại (R4). Đây là thay đổi duy nhất trong `pipeline.py`.
2. **Câu nào không cần gate.** Câu xã giao (không tìm gì) và câu tóm tắt (đã có cả dàn ý) thì tìm lại không giúp được gì: gate trả "đủ" ngay, không gọi model.
3. **Đo ở đâu.** Từ R1 tới R5, eval chấm ở đầu ra của `fuse` (10 ứng viên của một lượt tìm). R6 thêm lượt tìm, nên chấm ở đó không thấy gì. R6 chấm ở **context cuối cùng** (các đoạn thật sự đưa cho model, sau mọi vòng). Mốc R5 được đo lại theo cách này (miễn phí, từ cache) để hai con số so được với nhau.

## Ngoài phạm vi

- Gate dùng nguồn web (R8).
- Gate kiểm câu trả lời (đó là R7: gate kiểm context **trước** khi trả lời).
- Đổi ngân sách 8.000 token của context. Context cộng dồn vẫn nằm trong ngân sách đó, nên vòng sau chỉ thêm được khi vòng trước chưa dùng hết.

## Các file

| File | Việc |
| --- | --- |
| `app/core/stages/gate.py` | mới: `GateResult`, prompt có câu cân chỉnh, `make_gate(provider)` |
| `app/core/pipeline.py` | bước trả lời nhận câu hỏi, không nhận câu tìm của gate |
| `app/core/generator.py`, `app/core/deps.py` | hết vòng mà vẫn thiếu: báo model biết thiếu gì để nó nói ra |
| `app/config.py`, `.env.example` | `GATE_REASONING_EFFORT` |
| `eval/run.py` | chấm ở context cuối; số vòng trung bình; các quyết định của gate |
| test tương ứng; tài liệu ở Task 6 | |

---

### Task 1: stage `gate`

```python
class GateResult(BaseModel):
    sufficient: bool
    missing: str          # rỗng khi đủ
    refined_query: str    # rỗng khi đủ
```

- Model nhận câu hỏi và các đoạn trong context (tên tài liệu, tiêu đề, trang, nội dung), trả `GateResult`. Không trường nào có giá trị mặc định (structured outputs ở chế độ strict đòi đủ trường), nên dùng chuỗi rỗng thay cho `null`.
- `refined_query` phải dùng từ của câu hỏi và của context, không thêm khái niệm mới (cùng quy tắc với bước viết lại ở R4).
- Context rỗng: "chưa đủ" mà không gọi model, câu tìm mới là chính câu hỏi.
- Intent `CHITCHAT` hoặc `SUMMARIZE_DOC`: "đủ" mà không gọi model.
- Trace: `gate_detail` với `sufficient`, `missing`, `refined_query`, số đoạn đã xét.
- [x] **Test đỏ** (`AsyncStubProvider`): "đủ" và "chưa đủ" đi đúng vào `GateVerdict`; prompt có câu cân chỉnh và có mọi đoạn trong context; context rỗng và câu xã giao/tóm tắt không gọi model; model lỗi thì pipeline coi là "đủ" (fallback có sẵn).
- [x] **Cài đặt, test xanh.**

### Task 2: pipeline trả lời đúng câu hỏi

- `answer()` giữ câu hỏi sau bước viết lại; vòng lặp tìm bằng câu tìm của gate; bước trả lời nhận câu hỏi.
- Hết vòng mà gate vẫn "chưa đủ": bước trả lời được báo phần thiếu (`missing`) để nói rõ tài liệu có thể chưa trả lời đủ phần đó.
- [x] **Test đỏ:** gate "chưa đủ" một lần rồi "đủ" → bước trả lời nhận câu hỏi gốc chứ không phải câu tìm mới; hết vòng vẫn "chưa đủ" → prompt trả lời có phần thiếu; test vòng lặp sẵn có vẫn xanh.
- [x] **Cài đặt, test xanh; cả bộ suite xanh.**

### Task 3: nối vào `deps.py`

- `gate = make_gate(provider, GATE_REASONING_EFFORT)` khi `RETRIEVAL_MAX_ITERATIONS > 1`, ngược lại là no-op.
- `GATE_REASONING_EFFORT` mặc định `none`, giống bước viết lại; Task 5 đo cả `none` lẫn `low`.
- [x] **Test đỏ** qua `POST /query`: gate "chưa đủ" một lần → trace có 2 vòng, `query_traces.iterations == 2`, context có đoạn của cả hai vòng.
- [x] **Cài đặt, test xanh.**

### Task 4: eval

- Chạy vòng lặp tìm kiếm của pipeline tới context cuối (không gọi bước trả lời), rồi chấm `recall` và MRR trên các đoạn trong context theo thứ tự.
- In số vòng trung bình, số câu gate nói "chưa đủ", và với từng câu đó thì thiếu gì, tìm lại bằng câu nào.
- [x] **Test đỏ, cài đặt, test xanh.**
- [x] **Đo lại mốc R5 ở context cuối** (miễn phí, từ cache).

### Task 5: đo

- [x] **(gọi OpenAI, cần đồng ý)** R6 với `GATE_REASONING_EFFORT=none` rồi `low`. Mỗi câu một lần gọi gate (đọc khoảng 4–8 nghìn token context), thêm lần nữa cho mỗi vòng lặp; cộng embedding cho các câu tìm mới. Ước khoảng 3–6 cent mỗi lượt; lưu cache.
- [x] So với R5 ở context cuối: `recall`, MRR theo loại câu, số vòng trung bình (spec: khỏe là 1,2–1,6), thời gian thêm vào mỗi câu.
- [x] Đọc từng câu gate nói "chưa đủ": thiếu có thật không, câu tìm mới có hợp lý không.

### Task 6: tài liệu

- [x] `docs/retrieval-design.md` §6.3 và §8.7.
- [x] `README.md`, `docs/system-design.md`.
- [x] Bản tiếng Anh của plan này.

## Điều kiện hoàn thành

1. Cả bộ test xanh, `ruff` sạch.
2. `RETRIEVAL_MAX_ITERATIONS=1`: như R5.
3. Gate lỗi hoặc hết vòng: câu hỏi vẫn được trả lời; bước trả lời nhận câu hỏi chứ không phải câu tìm.
4. Số vòng trung bình, `recall` và MRR ở context cuối, và thời gian thêm vào được ghi vào spec §8.7 cạnh R5.

## Kết quả (2026-10-17)

Chấm ở context cuối cùng, so với R5 chấm cùng cách:

| | Context recall | Context MRR | Số vòng trung bình | Thời gian gate |
| --- | --- | --- | --- | --- |
| R5 (một vòng) | 0,900 | 0,635 | 1,00 | — |
| R6, gate `none` | 0,900 | 0,635 | 1,20 | trung vị 1,2 s, tối đa 3,4 s |
| R6, gate `low` | 0,900 | 0,635 | 1,15 | trung vị 1,3 s, tối đa 5,2 s |
| R5, tắt rerank | 0,850 | 0,598 | 1,00 | — |
| R6, tắt rerank, gate `low` | 0,850 | 0,598 | 1,20 | trung vị 1,3 s |

- **Gate hoạt động đúng:** trung bình 1,15–1,20 vòng, trong hoặc sát vùng khỏe 1,2–1,6 của spec. Gate chỉ nói "chưa đủ" ở những câu thật sự thiếu: a01 (mức phạt ô tô vượt đèn đỏ), a14 (thời hạn tước bằng), a13, a11.
- **Nhưng không câu nào được cứu.** Câu tìm của gate không đưa được trang thiếu lên. a01 bị reranker CPU đẩy xuống từ R3, và mọi lượt tìm lại đều đi qua chính reranker đó. Khi tắt rerank, câu tìm lại của a14 ("Thời hạn tước quyền sử dụng giấy phép lái xe…") vẫn không ra trang 15, vì cụm từ đó có ở gần như mọi điều của nghị định.
- **Ở mức `none`**, câu tìm mới thường lặp lại nguyên câu hỏi. **Ở mức `low`** thì cụ thể hơn, nhưng kết quả như nhau.
- **Hai lỗi tìm ra và đã sửa trong lúc làm:**
  - Bước trả lời từng nhận câu tìm cuối cùng của gate thay cho câu hỏi của người dùng.
  - Vòng đầu đã dùng 7.712/8.000 token, các vòng sau không còn chỗ. Giờ mỗi vòng sau được cộng thêm `CONTEXT_ROUND_EXTRA` (3.000 token), và chỉ câu nào gate cho là thiếu mới dùng tới.
- **Kết luận:** thứ giúp được các câu này là phần tìm kiếm (một reranker mạnh hơn), không phải thêm vòng với cùng phần tìm kiếm đó. `RETRIEVAL_MAX_ITERATIONS` mặc định giữ 1: gate tốn mỗi câu một lần gọi model (khoảng 1,2 s) mà không có lợi đo được trên bộ này. Khi bật mà hết vòng, câu trả lời nói rõ phần tài liệu có thể chưa có.
- **Chi phí đo:** khoảng 70 lần gọi gate và 5 embedding, tất cả đã lưu cache.

Lệch so với plan: thêm `Context.rounds` và `CONTEXT_ROUND_EXTRA`; tách `gather()` ra khỏi `answer()` trong `pipeline.py`, để eval chạy đúng vòng lặp của pipeline mà không gọi bước trả lời.
