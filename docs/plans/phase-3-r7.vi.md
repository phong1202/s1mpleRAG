# Phase 3, R7: kiểm câu trả lời có bám nguồn không

> **Cho người thực thi:** làm lần lượt từng task, mỗi task theo TDD (test đỏ → code → test xanh). Không tự commit; khi được bảo thì gộp thành vài commit lớn. Bước nào gọi OpenAI thật phải được đồng ý trước.

**Mục tiêu:** sau khi trả lời, một lần gọi model riêng đọc câu trả lời cạnh các nguồn và chỉ ra câu nào không có nguồn đỡ. Câu đó bị bỏ, thay bằng một dòng nói tài liệu không đề cập. Đo bằng tỷ lệ bịa (spec §8.2) trên Set A và Set B, Set B lần đầu được chấm.

**Kiến trúc:** stage `reflect` có sẵn từ R0 (no-op trả nguyên câu trả lời, fallback có sẵn trong `pipeline.py`). R7 thay nó bằng `make_reflect(provider)` khi `RETRIEVAL_REFLECT_ENABLED=true` (cờ đã có, mặc định `false`). `pipeline.py` chỉ đổi hai chỗ: `reflect` nhận thêm câu hỏi (để bỏ qua câu xã giao), và bước trả lời + kiểm tách thành `respond()` để eval chấm được cả bản nháp lẫn bản cuối.

**Spec:** `docs/retrieval-design.md` §6.5 (reflection), §6 (chính sách lỗi: reflect lỗi thì giữ nguyên câu trả lời), §8.2 (fabrication rate, citation accuracy), §8.4 (Set B).

## Ràng buộc chung

- `RETRIEVAL_REFLECT_ENABLED=false`: kết quả y hệt bây giờ.
- Reflect lỗi, trả sai schema, hay chỉ ra câu không có trong câu trả lời: câu trả lời giữ nguyên ở phần đó. Không bao giờ trả lỗi vì bước kiểm.
- Bỏ câu chứ không viết lại (spec §6.5): viết lại tốn thêm một lần gọi mà không chắc đúng hơn; bỏ cả câu trả lời thì thường là quá tay.
- Trích dẫn tính lại từ văn bản sau khi bỏ câu: số `[n]` chỉ còn trong câu đã bỏ thì trích dẫn đó cũng mất.
- Bộ câu hỏi eval không đổi. Không sửa `web/`; hợp đồng `POST /query` không đổi.

## Ba việc phải quyết trước

1. **Dòng thay cho câu bị bỏ viết bằng ngôn ngữ nào.** Câu trả lời theo ngôn ngữ của câu hỏi, pipeline thì không biết đó là tiếng gì. Đề xuất: thêm trường `note` vào `Reflection` (spec chỉ có `grounded`, `unsupported`): một câu, cùng ngôn ngữ với câu trả lời, nói tài liệu không đề cập phần đã bỏ; rỗng khi không bỏ gì. Không cần lần gọi thứ hai.
2. **Ai chấm tỷ lệ bịa.** Không thể để chính `reflect` tự chấm mình. Đề xuất: eval có một prompt giám khảo riêng (cùng `gpt-6-luna`, vì provider chỉ có một model, nhưng reasoning `high`), và mọi câu giám khảo hoặc reflect gắn cờ đều được in ra để đọc tay. 28 câu trả lời đọc tay được, và đọc tay mới là phép kiểm thật; giám khảo chỉ để biết đọc chỗ nào.
3. **Đo trước/sau trên cùng một bản nháp.** Bước trả lời không cho cùng một câu hai lần, nên chạy eval hai lần (tắt rồi bật reflect) thì khác biệt lẫn với nhiễu. Eval sinh bản nháp một lần, chấm nó, cho qua `reflect`, rồi chấm bản cuối. Hai con số đi theo cặp ngay trong một lần chạy, và cache làm lần chạy sau miễn phí.

## Ngoài phạm vi

- Hành động theo intent `OUT_OF_SCOPE` (bỏ tìm kiếm, từ chối ngay). Từ R5 rewriter chỉ nhận ra 1/5 câu ngoài phạm vi, và nhận nhầm một câu thật thì mất cả câu đó. R7 đo xem câu trả lời có từ chối đúng không; có cần hành động theo intent hay không thì quyết từ con số đó.
- Kiểm trích dẫn có đúng trang không (`[n]` trỏ đúng nguồn nhưng nguồn đó không nói điều câu viết). Reflect thấy được một phần việc này vì nó đọc câu cạnh nguồn, nhưng không có bước sửa riêng.
- Stream câu trả lời. Khi làm stream thì phần đã gửi không bỏ được nữa; lúc đó phải chọn giữa kiểm xong mới gửi và gửi xong mới kiểm. Ghi lại ở đây để không quên.

## Các file

| File | Việc |
| --- | --- |
| `app/core/stages/reflect.py` | mới: `Reflection`, prompt, `make_reflect(provider, reasoning_effort)`, bỏ câu |
| `app/core/generator.py` | tách phần dựng danh sách nguồn để `reflect` dùng lại (prompt của gate cũng đang chép đoạn này) |
| `app/core/pipeline.py`, `app/core/stages/noops.py` | `reflect(query, answer, context, trace)`; `respond()` trả cả bản nháp và bản cuối |
| `app/core/deps.py` | nối `reflect` theo cờ |
| `app/config.py`, `.env.example` | `REFLECT_REASONING_EFFORT` |
| `eval/run.py`, `eval/judge.py` | chế độ `--full`: nháp → chấm → reflect → chấm, cả Set A và Set B |
| `eval/inject.py`, `eval/injections.yaml` | cài một câu bịa vào mỗi bản nháp, đo reflect bắt được bao nhiêu |
| test tương ứng; tài liệu ở Task 5 | |

---

### Task 1: stage `reflect`

```python
class Reflection(BaseModel):
    grounded: bool
    unsupported: list[str]   # nguyên văn các câu không có nguồn đỡ; rỗng khi grounded
    note: str                # một câu, cùng ngôn ngữ câu trả lời; rỗng khi grounded
```

- Model nhận câu hỏi, các nguồn (cùng nhãn tài liệu, tiêu đề, trang như bước trả lời) và câu trả lời có `[n]`. Prompt nói rõ: một câu được nguồn đỡ khi nguồn nói điều đó, kể cả diễn đạt khác; câu nói tài liệu không đề cập gì đó không phải là khẳng định; chép nguyên văn câu, không sửa.
- Bỏ câu: so khớp sau khi gộp khoảng trắng; bỏ luôn các `[n]` ngay sau câu; dòng gạch đầu dòng chỉ còn ký hiệu thì bỏ cả dòng. Câu không tìm thấy trong câu trả lời thì bỏ qua và ghi vào trace.
- Bỏ hết mọi câu: câu trả lời chỉ còn `note`.
- Intent `CHITCHAT`: không gọi model (không có nguồn và không có khẳng định nào).
- Trace: `reflect_detail` với `grounded`, `unsupported`, số câu đã bỏ, các câu không khớp, `asked_model`.
- [x] **Test đỏ** (`AsyncStubProvider`): grounded thì câu trả lời và trích dẫn y nguyên; một câu không có nguồn bị bỏ cùng `[n]` của nó và `note` được thêm, trích dẫn chỉ có ở câu đó biến mất, trích dẫn còn dùng ở câu khác thì giữ; câu không khớp bị bỏ qua và ghi trace; bỏ hết thì chỉ còn `note`; gạch đầu dòng rỗng bị bỏ; câu xã giao không gọi model; prompt có mọi nguồn và câu trả lời; model lỗi thì pipeline trả bản nháp (fallback có sẵn).
- [x] **Cài đặt, test xanh.**

### Task 2: nối vào `deps.py`

- `reflect = make_reflect(provider, REFLECT_REASONING_EFFORT)` khi `RETRIEVAL_REFLECT_ENABLED`, ngược lại là no-op.
- `REFLECT_REASONING_EFFORT` mặc định `low`; Task 4 đo cả `none` lẫn `low`, rồi chốt.
- [x] **Test đỏ** qua `POST /query`: reflect bỏ một câu → response và tin nhắn lưu trong hội thoại đều là bản đã bỏ, `citations` khớp với văn bản đó; trace có `reflect_detail`; cờ tắt thì không gọi model lần nào thêm.
- [x] **Cài đặt, test xanh; cả bộ suite xanh.**

### Task 3: eval

- `eval/judge.py`: `Judgement{declined: bool, unsupported: list[str]}`, prompt giám khảo (khác prompt của reflect), reasoning `high`.
- `--full`: với mỗi câu của Set A và Set B, dùng context mà vòng tìm đã có (không tìm lại), sinh nháp, chấm nháp, cho qua `reflect` (qua fallback của pipeline), chấm bản cuối.
- In ra:
  - tỷ lệ bịa (câu trả lời có ít nhất một câu giám khảo cho là không có nguồn) trước và sau reflect, riêng Set A và Set B;
  - Set B: câu ngoài phạm vi có từ chối không (5), câu nghe như cần web có trả lời không (3);
  - độ chính xác trích dẫn trước và sau (bỏ câu có thể bỏ mất trích dẫn đúng);
  - từng câu reflect đã bỏ, kèm việc giám khảo có gắn cờ câu đó ở bản nháp không: reflect bỏ mà giám khảo cho là có nguồn là dấu hiệu bỏ nhầm;
  - thời gian reflect.
- Bỏ lần chạy pipeline thứ hai trong `--full` hiện nay (nó tìm lại từ đầu cho mỗi câu).
- [x] **Test đỏ** (không gọi mạng): tỷ lệ bịa và quyết định từ chối tính đúng từ `Judgement` giả; trước/sau dùng cùng một bản nháp (bước trả lời chỉ được gọi một lần mỗi câu).
- [x] **Cài đặt, test xanh.**

### Task 4: đo

- [x] **(gọi OpenAI, cần đồng ý)** `--full` với reflect `none` rồi `low`. Mỗi câu: một lần trả lời, một lần reflect, hai lần chấm, mỗi lần đọc khoảng 8 nghìn token context; 28 câu. Ước 0,3–0,6 USD cho lượt đầu; lượt sau chỉ trả cho reflect và lần chấm bản cuối. Lưu cache.
- [x] Đọc tay mọi câu bị gắn cờ (bởi reflect hoặc giám khảo): bịa thật, bỏ nhầm, hay giám khảo sai.
- [x] Chốt `REFLECT_REASONING_EFFORT` và bật hay không `RETRIEVAL_REFLECT_ENABLED` mặc định, dựa trên: tỷ lệ bịa giảm bao nhiêu, bỏ nhầm bao nhiêu câu, thêm bao nhiêu giây.

### Task 5: tài liệu

- [x] `docs/retrieval-design.md` §6.5 và §8.7.
- [x] `README.md`, `docs/system-design.md`.
- [x] Bản tiếng Anh của plan này.

## Điều kiện hoàn thành

1. Cả bộ test xanh, `ruff` sạch.
2. `RETRIEVAL_REFLECT_ENABLED=false`: như bây giờ.
3. Reflect lỗi: câu hỏi vẫn được trả lời, bằng bản nháp.
4. Tỷ lệ bịa trước/sau, quyết định từ chối của Set B, số câu bỏ nhầm và thời gian thêm vào được ghi vào spec §8.7.

## Kết quả (2026-10-17)

Chạy `--full` trên 28 câu (Set A 20, Set B 8). Mỗi câu một bản nháp, giám khảo chấm bản nháp rồi chấm lại bản sau khi kiểm.

| | Bịa, Set A | Bịa, Set B | Set B đúng | Trích dẫn đúng trang | Bỏ nhầm | Thời gian kiểm |
| --- | --- | --- | --- | --- | --- | --- |
| Bản nháp | 0,000 | 0,000 | 7/8 | 0,900 | | |
| Prompt đầu, `none` | 0,000 | 0,000 | 7/8 | 0,850 | 1 (a13) | trung vị 1,4 s |
| Prompt đầu, `low` | 0,100 | 0,000 | 7/8 | 0,800 | 2 (a08, a13) | trung vị 1,5 s, tối đa 6,6 s |
| Prompt sửa, `none` | 0,050 | 0,000 | 7/8 | 0,900 | 1 (a05) | trung vị 1,4 s, tối đa 3,1 s |
| **Prompt sửa, `low`** | **0,000** | **0,000** | **7/8** | **0,900** | **0** | trung vị 1,5 s, tối đa 5,4 s |

- **Không có gì để bắt.** Bản nháp không có câu nào giám khảo cho là không có nguồn. Cả 5 câu ngoài phạm vi của Set B đều được bước trả lời từ chối, dù bước viết lại chỉ nhận ra 1 trong 5 là ngoài phạm vi. Câu sai duy nhất của Set B là b06: corpus trả lời được nhưng câu trả lời lại từ chối. Đó là việc của phần tìm kiếm và R8.
- **Prompt đầu chỉ gây hại**, đọc tay từng câu:
  - a13: trang 17 ghi mức phạt ở đầu khoản ("Phạt tiền từ 400.000 đồng đến 600.000 đồng…"), còn hành vi ở điểm i bên dưới. Bước kiểm không nối được hai chỗ, bỏ mất câu trả lời đúng duy nhất.
  - a08: bỏ câu trả lời đúng vì thiếu một ngoại lệ, tức là thiếu ý chứ không phải bịa. Dòng thay thế còn nói với người dùng rằng "câu trả lời không nêu ngoại lệ này".
  - Tỷ lệ bịa "sau" ở mức `low` (0,100) là do chính các dòng thay thế đó.
- **Đã sửa prompt:** thiếu ý không phải là không có nguồn; mức phạt ở đầu khoản áp cho mọi điểm bên dưới; dòng thay thế không nhắc tới "câu trả lời". Sau khi sửa, `low` không bỏ câu nào. `none` vẫn bỏ nhầm một câu ở a05. Có lúc đã thêm luật "mọi câu bị gắn cờ thì giữ bản nháp", nhưng đã bỏ: luật đó rút từ chính hai ca trên bộ eval, và nó tắt reflect đúng ở câu trả lời bịa từ đầu đến cuối.
- **Câu bịa cài vào: bắt được 20/20.** Bản nháp không có gì để bắt, nên mỗi bản nháp của Set A được cài một câu bịa (`eval/injections.yaml`, chạy `python -m eval.inject`), từng câu đã kiểm là không có trong nguồn: 4 câu gán hành vi vào khung phạt của hành vi khác, 5 câu sai số liệu, 11 câu thêm quy định không có. Ở mức `low`, reflect bỏ đủ 20 câu và không bỏ câu nào khác.
- **Câu về đầu khoản có overfit không:** câu này viết sau ca a13, nên lo nó làm reflect dễ dãi với câu sai khung phạt. Đo riêng bằng cách bỏ đúng câu đó khỏi prompt: còn 19/20, và ở a13 reflect bỏ câu phạt đúng mà giữ câu bịa. Ba câu sai khung còn lại bị bắt cả khi có lẫn khi không có câu đó. Vậy nó không làm reflect dễ dãi; nó giúp đọc đúng cấu trúc khoản, điểm. Giới hạn: 20 câu bịa do một model ngôn ngữ (Claude) viết tay, nên có thể dễ nhận ra hơn câu bịa thật của bước trả lời; chỉ có 4 câu sai khung phạt; mỗi biến thể chạy một lần.
- **Kết luận:** `REFLECT_REASONING_EFFORT=low`. `RETRIEVAL_REFLECT_ENABLED` mặc định tắt như mọi cờ khác. Bật hay không là đổi khoảng 1,5 s mỗi câu lấy việc bắt được câu bịa: trên bộ này bản nháp thật chưa bịa câu nào, còn câu bịa cài vào thì bị bắt đủ.
- **Chi phí đo:** khoảng 210 lần gọi model và 8 embedding, tất cả đã lưu cache.
- **Lệch so với plan:** `reflect` nhận thêm câu hỏi và `respond()` tách khỏi `answer()` trong `pipeline.py`; thêm phép đo câu bịa cài vào (`eval/inject.py`); eval không còn in thời gian của các lần gọi lấy từ cache.
