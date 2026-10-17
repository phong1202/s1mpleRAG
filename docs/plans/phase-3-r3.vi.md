# Phase 3, R3: rerank

> **Cho người thực thi:** làm lần lượt từng task, mỗi task theo TDD (test đỏ → code → test xanh). Không tự commit; khi được bảo thì gộp thành vài commit lớn. Bước nào gọi API trả tiền (OpenAI, OpenRouter) phải được đồng ý trước.

**Mục tiêu:** sau khi trộn bằng RRF, chấm lại các ứng viên bằng một cross-encoder rồi giữ 10 ứng viên tốt nhất, để trang đúng lên đầu danh sách. Đo bằng `MRR`, vì rerank không đưa thêm ứng viên mới nên `recall` không tăng được.

**Kiến trúc:** rerank nằm trong stage `fuse`, ngay sau RRF (spec §4.1: `R2: RRF  R3: +rerank`). Phía API chỉ có **một** client HTTP, nói theo dạng `/rerank` của Cohere: `{model, query, documents, top_n}` → `results[{index, relevance_score}]`. Cùng một dạng đó được phục vụ bởi:

- `reranker` — container mới của repo, chạy model nhỏ trên CPU. Mặc định, không cần GPU, không cần key;
- OpenRouter (`https://openrouter.ai/api/v1/rerank`), Cohere và vLLM — đổi bằng `.env`, không đổi code.

**Spec:** `docs/retrieval-design.md` §5.4 (rerank), §8.2 (MRR), §10 (cấu hình). Plan này sửa quyết định số 7 của spec (xem dưới).

## Đã đo trước khi viết plan

Mỗi câu hỏi của bộ eval: 50 ứng viên đầu của tìm từ khoá, chấm lại, xem trang đúng đứng thứ mấy. Không gọi API trả tiền.

| Model | Chạy trên | Trúng top 10 | Đúng hạng 1 | MRR | Thời gian cho 50 cặp |
| --- | --- | --- | --- | --- | --- |
| không rerank | — | 15/20 | — | 0.399 | — |
| `Qwen3-Reranker-4B` | GPU, vLLM | 17/20 | 15 | **0.800** | 1,35 s (9 GB VRAM) |
| `Qwen3-Reranker-0.6B` | GPU | 17/20 | 10 | 0.625 | 0,66 s |
| `bge-reranker-v2-m3` | GPU | 17/20 | 9 | 0.607 | 0,19 s |
| `bge-reranker-v2-m3` | CPU | 17/20 | 9 | 0.607 | 10,5 s |
| `mmarco-mMiniLMv2-L12` (118M), ONNX int8 | CPU | 17/20 | 8 | 0.557 | **0,8 s** |
| `ViRanker` | GPU | 16/20 | 8 | 0.544 | 0,15 s |
| `AITeamVN/Vietnamese_Reranker` | GPU | 13/20 | 10 | 0.540 | 0,19 s |

Rút ra:

- Quyết định số 7 (`bge-reranker-v2-m3` tự host, chạy như Chandra) không còn đứng được: trên CPU nó mất 10 giây, còn GPU thì Chandra chiếm gần hết khi chạy.
- Model tốt nhất đo được (`Qwen3-Reranker-4B`) tự host thì quá nặng và quá chậm. Qua API thì có: OpenRouter, DeepInfra.
- Model nhỏ ONNX int8 là phương án duy nhất chạy được ở mọi lúc trên máy này. Nó kém Qwen-4B nhiều, nhưng vẫn hơn hẳn không rerank.

Vì vậy: container CPU là mặc định để R3 luôn chạy được; API là một dòng cấu hình khi chủ repo có key.

Giới hạn của bảng: ứng viên lấy từ tìm từ khoá chứ không phải danh sách hybrid; chỉ 20 câu.

## Ràng buộc chung

- Tắt cờ `RETRIEVAL_RERANK_ENABLED` thì kết quả y hệt R2.
- **Reranker lỗi hoặc quá thời gian thì giữ thứ tự RRF**, ghi vào trace và log, không bao giờ làm hỏng request (spec §5.4).
- Không giữ kết nối DB trong lúc chờ reranker: kết thúc transaction trước khi gọi ra ngoài (cùng lý do với `expand`).
- Bộ câu hỏi eval không đổi. Không sửa `web/`; hợp đồng `POST /query` không đổi.
- Test của bộ suite chính không cần `onnxruntime` và không tải model.

## Ngoài phạm vi

- Chọn API rerank cuối cùng: cần key và một lượt đo trên danh sách hybrid (Task 5).
- Rerank trên GPU, và chạy Qwen-4B tự host.
- Đưa tiêu đề điều vào văn bản gửi reranker: chưa đo; R3 gửi nội dung chunk con như lúc đo ở trên.

## Các file

| File | Việc |
| --- | --- |
| `app/config.py`, `.env.example` | thêm `RETRIEVAL_RERANK_CANDIDATES`, `RERANKER_URL`, `RERANKER_MODEL`, `RERANKER_API_KEY`, `RERANKER_TIMEOUT_S` |
| `app/core/rerank_client.py` | mới: client HTTP dạng Cohere |
| `app/core/stages/fusion.py` | RRF giữ `RETRIEVAL_RERANK_CANDIDATES`, rerank, cắt còn `RETRIEVAL_RERANK_KEEP` |
| `app/core/deps.py` | nối client vào stage theo cờ; kết thúc transaction trước khi rerank |
| `reranker_service/` | mới: FastAPI + onnxruntime, `POST /rerank`, `GET /health` |
| `docker-compose.yml` | service `reranker` |
| `eval/run.py` | cache embedding câu hỏi để đo lại không tốn tiền |
| `tests/test_rerank_client.py`, `tests/test_fusion.py`, `tests/test_reranker_service.py`, `tests/test_hybrid_search.py` | test |
| `docs/retrieval-design.md`, `docs/system-design.md`, `README.md` | Task 6 |

---

### Task 1: client rerank

**Tạo ra:**

```python
class RerankError(Exception): ...

class RerankClient:
    def __init__(self, url: str, model: str, api_key: str | None, timeout_s: float,
                 transport: httpx.AsyncBaseTransport | None = None) -> None: ...
    async def scores(self, query: str, documents: list[str]) -> list[float]:
        """Điểm theo đúng thứ tự `documents`. Lỗi mạng, HTTP lỗi, quá thời gian
        hoặc trả thiếu chỉ số đều thành RerankError."""
```

- `POST {url}/rerank` với `{"model", "query", "documents", "top_n": len(documents)}`; có key thì thêm `Authorization: Bearer …`.
- Kết quả về theo thứ tự điểm; client đặt lại theo `index`, nên chỉ số nào thiếu hoặc trùng là lỗi.
- `documents` rỗng thì trả `[]`, không gọi mạng.

- [x] **Test đỏ** (`httpx.MockTransport`, không có mạng): đúng body và header; điểm theo đúng thứ tự gốc dù server trả theo thứ tự khác; không có key thì không gửi header; 500, quá thời gian, JSON thiếu chỉ số → `RerankError`; danh sách rỗng không gọi mạng.
- [x] **Cài đặt, test xanh.**

### Task 2: stage `fuse` có rerank

- `reciprocal_rank_fusion` giữ nguyên. Khi bật rerank, RRF cắt ở `RETRIEVAL_RERANK_CANDIDATES` (mặc định 50) thay vì `RETRIEVAL_TOP_K`; reranker chấm số đó; giữ `RETRIEVAL_RERANK_KEEP` (10).
- Ứng viên giữ `source` gốc; `score` thành điểm rerank; `rank` đánh lại từ 1.
- `RerankError` → thứ tự RRF cắt ở `RETRIEVAL_TOP_K`; trace ghi `rerank_detail` với `fallback` và lý do.
- Trace luôn có `rerank_detail`: số ứng viên gửi đi, thời gian, và thứ hạng RRF của 10 ứng viên được giữ (để thấy rerank đã kéo ai lên).

- [x] **Test đỏ** với một reranker giả: rerank đảo thứ tự; giữ đúng 10; lỗi thì về RRF và trace ghi lý do; tắt cờ thì giống R2.
- [x] **Nối vào `deps.py`**: client tạo khi `RETRIEVAL_RERANK_ENABLED`; `await session.commit()` trước khi gọi reranker; test kiểm không có transaction mở trong lúc reranker chạy (cùng cách với `test_no_transaction_is_open_while_the_model_writes`).
- [x] **Test xanh; cả bộ suite xanh.**

### Task 3: container `reranker` (CPU)

- `reranker_service/app.py`: FastAPI, `POST /rerank` theo dạng Cohere, `GET /health`. Phần chấm điểm là một hàm truyền vào, để test chạy với hàm giả.
- `reranker_service/onnx_scorer.py`: `onnxruntime` + `tokenizers`; model `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; file ONNX chọn bằng biến môi trường. Mặc định `model_quint8_avx2.onnx` vì chạy được trên mọi CPU có AVX2; `model_qint8_avx512_vnni.onnx` là bản đã đo 0,8 s trên máy này.
- `reranker_service/Dockerfile` riêng; model tải lúc build, nên container khởi động không cần mạng.
- `docker-compose.yml`: service `reranker`, giới hạn CPU và RAM, `restart: unless-stopped`; `api` có `RERANKER_URL=http://reranker:8200`.

- [x] **Test đỏ** (`tests/test_reranker_service.py`, hàm chấm giả): đúng dạng response, `index` khớp đầu vào, `top_n`, danh sách rỗng, `/health`.
- [x] **Cài đặt, test xanh.**
- [x] **Đo hai file ONNX trên máy này** (không cần GPU, không gọi API): thời gian cho 50 và 100 cặp, MRR trên ứng viên từ khoá. Chọn file mặc định theo số đo.
- [x] **Build và chạy container**, gọi `/rerank` một lần từ `api`.

### Task 4: eval đo lại không tốn tiền

- `eval/run.py --embed-cache data/eval-embeddings.json`: embedding của câu hỏi lưu theo `(model, dimensions, text)`; lần sau đọc từ file. `data/` đã gitignore.
- [x] **Test đỏ:** lần hai không gọi provider; đổi model hoặc số chiều thì không dùng cache cũ.
- [x] **Cài đặt, test xanh.**

### Task 5: đo

- [x] **(gọi OpenAI lần cuối: 20 embedding, cần đồng ý)** R2 với cache bật, để nạp cache.
- [x] **(miễn phí)** R3 với container CPU.
- [ ] **(cần key OpenRouter và đồng ý; dưới 0,10 USD)** R3 qua OpenRouter: `qwen/qwen3-reranker-4b` nếu tài khoản dùng được, rồi Cohere `rerank-4-fast` và Voyage `rerank-2.5-lite`.
- [x] Ghi `recall@10`, MRR theo từng loại câu, thời gian của node `rerank` trong trace. Nếu MRR không tăng so với R2 thì cờ để tắt và spec ghi lý do (spec §8.2: "If MRR does not move, the reranker is not worth a container").

### Task 6: tài liệu

- [x] `docs/retrieval-design.md`: quyết định số 7 viết lại theo bảng đo; §5.4 và §10 theo cấu hình mới. (§8.7 chờ Task 5.)
- [x] `docs/system-design.md`, `README.md`: service `reranker`, cách chuyển sang API.
- [x] Bản tiếng Anh của plan này.

## Điều kiện hoàn thành

1. Cả bộ test xanh, `ruff` sạch; test không tải model, không gọi mạng.
2. Tắt cờ: trace và kết quả như R2.
3. Reranker tắt hoặc chết: request vẫn trả lời, trace ghi fallback.
4. Đổi giữa container CPU và OpenRouter chỉ bằng `.env`.
5. Số R3 (ít nhất bản CPU) ghi vào spec §8.7 cạnh R2.

## Trạng thái (2026-10-16)

Task 1 đến 4 xong; cả bộ test xanh, `ruff` sạch. Task 5 dừng ở bước đầu, chờ đồng ý gọi OpenAI. Task 6 còn dòng R3 ở spec §8.7 và bản tiếng Anh của plan.

Lệch so với plan, có lý do:

- **Container dùng sentence-transformers thay vì onnxruntime + tokenizers.** Model chưa chốt, và `gte-multilingual-reranker-base` cần code riêng của nó; sentence-transformers nạp được mọi cross-encoder, chạy bằng ONNX hoặc torch. Đổi lại image nặng hơn (2,57 GB).
- **Model được tải vào `/opt/model` lúc build và nạp từ đó.** Nạp theo tên repo thì thư viện vẫn hỏi Hugging Face danh sách file mỗi lần khởi động, kể cả khi đã bật chế độ offline. Lỗi này lộ ra nhờ bước build nạp lại model khi tắt mạng.
- **File ONNX mặc định là bản `avx512_vnni`** (đã đo 0,8 s trên máy này), không phải bản `avx2` như plan ghi, vì bản `avx2` chưa đo. Máy không có AVX-512 VNNI thì đổi `RERANKER_LOCAL_ONNX_FILE`.
- **Thêm các model có tiếng quốc tế vào bảng đo** (theo yêu cầu của chủ repo): `gte-multilingual-reranker-base`, `bge-reranker-v2-m3` bản int8, `Qwen3-Reranker-0.6B` trên CPU.

Container đo qua HTTP: 50 đoạn 0,62 s, 100 đoạn 1,2 s, 759 MiB RAM; MRR 0.539 trên ứng viên từ khoá (đo trực tiếp bằng onnxruntime là 0.557).

## Kết quả đo (2026-10-17)

Cùng corpus, cùng vector câu hỏi (lần R2 trả tiền 20 embedding và lưu cache, lần R3 đọc từ cache):

| | `recall@10` | `MRR` | Câu trượt |
| --- | --- | --- | --- |
| R2 | 0,750 | 0,542 | a08, a13, a14, a16, a18 |
| R3, reranker CPU `mmarco-mMiniLMv2` int8 | **0,850** | **0,583** | a01, a14, a16 |

Rerank kéo được a08, a13, a18 từ ngoài top 10 của RRF vào, nên recall cũng tăng, không chỉ MRR. Mất a01; MRR của câu chứa mã số giảm từ 0,750 xuống 0,613. Bước đo với API (OpenRouter) chưa làm: chưa có key.

## Sau review độc lập (2026-10-17)

Reviewer không tìm thấy lỗi nghiêm trọng, nhưng có 8 lỗi quan trọng. Đã sửa, mỗi lỗi có test đỏ → xanh:

- **Không phải mọi lỗi của reranker đều giữ thứ tự RRF.** URL sai (`httpx.InvalidURL`), số quá lớn hay JSON lồng quá sâu thoát khỏi client; stage khi đó rơi về no-op của pipeline, tức 10 kết quả vector đầu, mất hết kết quả từ khoá. Giờ client biến mọi lỗi thành `RerankError`, và stage bắt mọi lỗi để giữ thứ tự RRF của cả hai nguồn.
- **NaN và vô cực được nhận**, làm sai thứ tự và làm trace không ghi được vào Postgres. Giờ bị từ chối, cùng với chỉ số không phải số nguyên.
- **Eval không phân biệt được rerank thật với lần rơi về dự phòng.** Giờ eval đọc model từ trace của từng câu, và từ chối in số nếu có câu nào không được rerank.
- **Một loạt yêu cầu cùng lúc làm mọi bên gọi quá thời gian**, và yêu cầu đã bị bỏ vẫn chạy tiếp. Giờ container chấm từng yêu cầu một, cho tối đa 2 yêu cầu chờ, quá thì trả 503 ngay; yêu cầu mà bên gọi đã bỏ thì không chấm; `/health` không cần luồng phụ nên luôn trả lời.
- **`RERANKER_TIMEOUT_S` không phải hạn chót của cả lần gọi** (httpx tính thời gian cho từng lần đọc). Giờ bọc bằng `asyncio.timeout`.
- **Cấu hình URL và model dễ sai.** `RERANKER_MODEL` không còn bắt buộc: để trống thì container dùng model nó được build. `.env.example` và README ghi rõ để trống ba biến khi dùng container.
- **`cpus: 8`** làm container không khởi động trên máy ít hơn 8 CPU: đã bỏ. Cổng 8200 giờ chỉ mở trên loopback.
- **Đổi model dễ hỏng lúc build:** khi repo không có file ONNX đã chọn thì báo lỗi rõ ràng; ghi chú cho CPU không có AVX-512 VNNI.
- Việc nhỏ: thời gian rerank không bị cộng hai lần vào trace; giới hạn kích thước đầu vào; ghim phiên bản thư viện của container; cache embedding ghi an toàn và vẫn lưu khi một lượt chạy hỏng giữa chừng; kiểm `RETRIEVAL_RERANK_CANDIDATES ≥ RETRIEVAL_RERANK_KEEP, RETRIEVAL_TOP_K`.

Chưa sửa: `httpx.AsyncClient` tạo mới mỗi lần gọi (tốn thêm một lần bắt tay TLS với API ngoài); bộ lọc `.gitignore` `data/` không neo vào gốc repo (file này chủ repo giữ ngoài commit).

Sau khi sửa: 444 test xanh. Đo lại R3 từ cache: vẫn 0,850 / 0,583, lần này có trace xác nhận cả 20 câu đều được rerank thật. Thử 30 yêu cầu cùng lúc vào container: 3 được trả lời trong 2,4 s, 27 bị từ chối trong 0,24 s, yêu cầu kế tiếp 0,66 s.

## Chốt model (2026-10-17)

Chủ repo chốt model CPU hiện tại: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, bản int8, trong container `reranker`. Spec quyết định số 7 và §5.4 đã ghi lại. Bước đo qua OpenRouter không làm.
