# Phase 2 · Ngữ cảnh điều khoản — sửa khâu nạp để chunk biết nó thuộc điều nào · Implementation Plan

> **Cho agentic worker:** REQUIRED SUB-SKILL: dùng superpowers:executing-plans để làm plan này theo từng task. Các bước dùng cú pháp checkbox (`- [ ]`) để theo dõi.

**Mục tiêu:** Mỗi chunk mang theo tên điều, mục, chương và tên văn bản của nó, cả trong `heading_path` lẫn trong văn bản được embed, để các điều khoản giống chữ nhau không còn bị lấy nhầm.

**Kiến trúc:** S2 thôi tin vào cấp heading của markdown: nó coi mọi dòng heading hoặc in đậm là tiêu đề, và suy ra thứ bậc từ cách các tiêu đề đánh số lặp lại trong chính tài liệu, không dùng danh sách từ khoá nào. S3 được đưa tiêu đề của chunk khi viết câu ngữ cảnh. S4 ghép tên văn bản và tiêu đề vào đầu văn bản được embed. Một lệnh nạp lại chạy S2 → S5 từ checkpoint của S1, nên không phải OCR lại.

**Tech Stack:** Celery · PyMuPDF/Chandra (không đổi) · OpenAI qua `shared/llm.py` · pytest · uv · ruff

**Spec:** [`docs/system-design.md`](../system-design.md) §6 (S2, S3, S4). Plan này sửa hành vi của ba stage đó; Task 6 cập nhật spec.

**Bản tiếng Anh:** viết sau khi plan này được duyệt, để không phải dịch hai lần.

---

## Vì sao có plan này

Ngày 2026-10-11, câu hỏi thật đầu tiên ("ô tô vượt đèn đỏ bị phạt bao nhiêu") được trả lời bằng mức phạt của xe máy chuyên dùng. Điều tra cho thấy lỗi nằm ở dữ liệu đã nạp, không ở đường đọc:

| Quan sát | Số liệu |
| --- | --- |
| Mệnh đề "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông" trong NĐ 168/2024 | Xuất hiện nguyên văn ở 4 trang (14, 20, 25, 28), mỗi trang một loại xe |
| Chunk ở trang 14 (ô tô) | Nội dung, câu ngữ cảnh của S3 và `heading_path` đều không có chữ "ô tô" |
| Cách OCR đánh dấu dòng "Điều N." trong cùng một nghị định | h2: 3, h3: 14, h4: 7, in đậm: 29, chữ thường: 1 |
| S2 hiện tại | Chỉ nhận dòng bắt đầu bằng `#`, và tin vào số dấu `#` làm độ sâu |

Hệ quả: 29 trên 54 điều của NĐ 168 không được nhận là tiêu đề, và các điều được nhận thì nằm sai chỗ trong cây (cả Chương II từng nằm dưới "Điều 2"). Tìm kiếm từ khoá (R2) và rerank (R3) không cứu được, vì cả hai chỉ đọc nội dung chunk.

Thuật toán ở Task 1 đã được chạy thử trên checkpoint của cả 10 tài liệu: trang 14 nhận tiêu đề "Điều 6 … xe ô tô …", trang 20 "Điều 7 … xe mô tô, xe gắn máy …", trang 25 "Điều 8 … xe máy chuyên dùng".

Giới hạn đã biết của thuật toán: nó cần tiêu đề có số. Một tài liệu chỉ có tiêu đề không đánh số và cấp heading lộn xộn vẫn được chia như hiện nay, tức vẫn có thể sai cây.

## Global Constraints

- **Git:** KHÔNG chạy `git commit`, `git push`, `git merge`. Mỗi task kết thúc bằng dừng và báo cáo.
- **Suite xanh ở cuối mọi task:** `uv run pytest -q --ignore=tests/test_ocr_service.py`, `uv run ruff check .`, `uv run ruff format --check .`.
- **Gọi OpenAI là việc có chủ đích.** Chỉ Task 5 gọi: nạp lại 10 tài liệu là khoảng 3.000 chunk qua S3 và S4, ước chừng 0,25 USD theo bảng giá ở Task 14 của plan R0 + R1. Hỏi trước khi chạy.
- **Không OCR lại.** Nạp lại bắt đầu từ `parsed.json`. Checkpoint trong MinIO tự hết hạn sau 7 ngày (các tài liệu nạp ngày 06 và 07/10); bản sao của cả 10 file nằm ở `data/parsed-backup/` (không commit).
- **Tài liệu không có tiêu đề đánh số không được đổi hành vi:** S2 chia theo heading markdown như hiện nay.
- **Không viết cứng từ khoá của một loại tài liệu nào vào code.** Cấu trúc được học từ từng tài liệu.
- **`web/` không thuộc plan này.**
- **Sau khi nạp lại, mọi chunk đổi.** Nhãn của `eval/dataset.yaml` neo vào `(sha256, page)` nên vẫn còn nghĩa, nhưng phải chạy lại `--check-labels`. Con số nền của R1 chỉ được đo sau Task 5.

## Cấu trúc file

```
worker/steps/chunking.py      SỬA — dàn ý học từ tài liệu; child mang heading_path (Task 1, 2)
shared/llm.py                 SỬA — enrich() gửi kèm tiêu đề (Task 2)
worker/steps/embedding.py     SỬA — contextualize() ghép tên văn bản và tiêu đề (Task 3)
worker/stages.py              SỬA — embed/persist truyền tên văn bản; relaunch_from_structure (Task 3, 4)
worker/reingest.py            ★ MỚI — lệnh nạp lại từ checkpoint của S1 (Task 4)
tests/test_chunking.py        Task 1, 2
tests/test_llm_provider.py    Task 2
tests/test_embedding.py       Task 3
tests/test_stage_chain.py     Task 4
docs/system-design.md         SỬA — §6 S2, S3, S4 (Task 6)
```

---

## Task 1: S2 dựng dàn ý từ chính tài liệu, không cần danh sách từ khoá

Bản đầu của task này nhận diện bốn từ "Phần, Chương, Mục, Điều" viết cứng trong code. Bản này bỏ danh sách đó. Ba quy tắc, không quy tắc nào nhắc tới một từ cụ thể:

1. **Tiêu đề là gì.** Một dòng heading markdown, hoặc một dòng in đậm từ đầu tới cuối. Số dấu `#` không được dùng làm độ sâu.
2. **Loại của tiêu đề.** Tiêu đề có dạng "chữ + số" ("Điều 6.", "Chương II", "Article 3", "Phụ lục I") thuộc một loại, là phần chữ. Loại nào xuất hiện từ hai lần trở lên trong tài liệu thì là cấu trúc. Một dòng thường cũng được nhận nếu nó là số kế tiếp của một loại đã biết ("Điều 7." ngay sau "Điều 6.").
3. **Thứ bậc.** Một tiêu đề đóng tiêu đề cùng loại đang mở, cùng mọi thứ nằm trong nó. Loại A chỉ là phần chia nhỏ của loại B nếu có chỗ hai tiêu đề A trở lên nằm giữa hai tiêu đề B liên tiếp; không thì A không bao giờ được xếp dưới B.

Tài liệu không có tiêu đề đánh số nào thì chia theo heading markdown như hiện nay.

Đã chạy thử trên checkpoint của cả 10 tài liệu (2026-10-11): thuật toán tự tìm ra "chương, mục, điều", và cả "phụ lục", "mẫu số" ở hai nghị định; số điều tìm được là 87, 109, 89, 38, 53, 65, 19, 21, 55, so với 85, 111, 89, 39, 54, 65, 19, 21, 55 của bản dùng từ khoá. Trang 14 của NĐ 168 nhận đường dẫn "Chương II … > Mục 1 … > Điều 6 … xe ô tô …".

**Files:**
- Modify: `worker/steps/chunking.py` (thêm `split_document`; `chunk_document` gọi nó; `split_sections` giữ nguyên làm đường lui)
- Modify: `tests/test_chunking.py`

**Interfaces:**
- Produces: `split_document(pages: list[str]) -> list[list[tuple[str, str]]]`: với mỗi trang, danh sách `(heading_path, body)`. Cần cả tài liệu một lúc, vì loại và thứ bậc được học từ toàn bộ các tiêu đề.

- [x] **Step 1: Viết test đỏ**

```python
# tests/test_chunking.py — thêm vào cuối; thêm `split_document` vào dòng import từ
# worker.steps.chunking ở đầu file.


def _paths(*pages: str) -> list[str]:
    return [path for page in split_document(list(pages)) for path, _ in page]


def test_a_numbered_heading_counts_whatever_markup_the_parser_gave_it():
    """One decree, parsed in one run, held the same kind of line as h2, h3, h4,
    h5, bold and plain text. Only '#' lines were read as headings, so 29 of its
    54 articles were missed and their chunks inherited the wrong heading."""
    page = (
        "##### Điều 5. Mot\nnoi dung 5\n\n"
        "**Điều 6. Xử phạt người điều khiển xe ô tô**\nnoi dung 6\n\n"
        "Điều 7. Xử phạt người điều khiển xe mô tô\nnoi dung 7\n"
    )

    assert _paths(page) == [
        "Điều 5. Mot",
        "Điều 6. Xử phạt người điều khiển xe ô tô",
        "Điều 7. Xử phạt người điều khiển xe mô tô",
    ]


def test_depth_comes_from_how_headings_recur_not_from_the_markdown_level():
    """`# Điều 2` followed by `## Chương II` used to file the whole chapter
    under Article 2. A chapter never falls twice between two articles, so it
    cannot be a subdivision of one."""
    page = (
        "# Điều 1. A\nx\n\n# Điều 2. B\nx\n\n"
        "## Chương II\n\n## HÀNH VI VI PHẠM\n\n"
        "#### Mục 1. QUY TẮC\ny\n\n**Điều 6. Xe ô tô**\nz\n\n"
        "#### Mục 2. KHÁC\n\n**Điều 7. C**\nz\n\n"
        "## Chương III\n\n## THI HÀNH\n\n**Điều 8. D**\nz\n"
    )

    paths = _paths(page)

    assert "Chương II. HÀNH VI VI PHẠM > Mục 1. QUY TẮC > Điều 6. Xe ô tô" in paths
    assert paths[-1] == "Chương III. THI HÀNH > Điều 8. D"


def test_no_list_of_words_is_involved():
    """The kinds are whatever the document numbers. Nothing here names them."""
    page = (
        "# Part 1\n\n## Article 1. Scope\nx\n\n## Article 2. Terms\ny\n\n"
        "# Part 2\n\n## Article 3. Fees\nz\n"
    )

    assert _paths(page) == [
        "Part 1",
        "Part 1 > Article 1. Scope",
        "Part 1 > Article 2. Terms",
        "Part 2",
        "Part 2 > Article 3. Fees",
    ]


def test_a_reference_or_a_quoted_article_is_not_a_heading():
    """"Điều 24 của Luật này" is a reference, and an amending decree quotes
    whole articles of the decree it amends. Neither opens a section."""
    page = (
        "**Điều 1. Sửa đổi, bổ sung Điều 9**\n"
        "Điều 24 của Luật này được áp dụng.\n"
        "“Điều 9. Cơ sở dữ liệu\n"
        "a) theo quy định tại Điều 6. của Nghị định này\n\n"
        "**Điều 2. Hiệu lực**\nx\n"
    )

    assert _paths(page) == ["Điều 1. Sửa đổi, bổ sung Điều 9", "Điều 2. Hiệu lực"]


def test_the_open_article_carries_over_a_page_break():
    first = "**Điều 5. A**\nx\n\n**Điều 6. Xe ô tô**\nkhoan 1\n"

    assert _paths(first, "khoan 9, tiep theo trang truoc\n")[-1] == "Điều 6. Xe ô tô"


def test_a_document_with_no_numbered_headings_still_follows_its_markdown():
    assert _paths("# A\nx\n\n## B\ny\n\n# C\nz\n") == ["A", "A > B", "C"]
```

- [x] **Step 2: Chạy để xác nhận đỏ** — `ImportError: cannot import name 'split_document'`.

- [x] **Step 3: Thêm `split_document` và cho `chunk_document` dùng nó**

Đặt khối dưới đây ngay sau hàm `split_sections` hiện có. `_HEADING` đã có sẵn trong file.

```python
# worker/steps/chunking.py
_MARKUP = re.compile(r"^[#\s*_]+|[\s*_]+$")
_NUMBERED = re.compile(r"^((?:[^\W\d_]+\s+){1,2})(\d+[a-z]?|[IVXLCDM]+)(?=[\s.:)]|$)")
_TITLE_MAX = 300


def _marked(line: str) -> bool:
    text = line.strip()
    return bool(_HEADING.match(text)) or (text.startswith("**") and text.endswith("**") and len(text) > 4)


def _kind(line: str) -> tuple[str, str, str] | None:
    """(kind, number, bare title) of a numbered heading line."""
    bare = _MARKUP.sub("", line)
    found = _NUMBERED.match(bare)
    if not found:
        return None
    return found.group(1).strip().casefold(), found.group(2), bare[:_TITLE_MAX]


def learn(lines: list[str]) -> tuple[set[str], set[tuple[str, str]]]:
    """The kinds that recur among marked lines, and the pairs (a, b) meaning
    'a is not a subdivision of b'.

    A kind subdivides another only if somewhere two or more of it sit between
    one heading of the other and the next: four sections inside one chapter.
    A chapter never does that to an article -- at most one chapter falls
    between two consecutive articles -- so a chapter is never filed under one,
    whatever '#' level the parser gave either."""
    seen = [k[0] for line in lines if _marked(line) and (k := _kind(line))]
    kinds = {k for k in seen if seen.count(k) >= 2}
    seen = [k for k in seen if k in kinds]
    outside: set[tuple[str, str]] = set()
    for a in kinds:
        for b in kinds - {a}:
            most = run = 0
            started = False
            for k in seen:
                if k == b:
                    started, run = True, 0
                elif k == a and started:
                    run += 1
                    most = max(most, run)
            if most < 2:
                outside.add((a, b))
    return kinds, outside


def split_document(pages: list[str]) -> list[list[tuple[str, str]]]:
    """Per page, the (heading_path, body) sections."""
    lines = [line for page in pages for line in page.splitlines()]
    kinds, outside = learn(lines)
    if not kinds:
        # No numbered headings to go by: the markdown levels are all there is.
        open_headings: list[str] = []
        return [split_sections(page, open_headings) for page in pages]
    stack: list[tuple[str, str]] = []  # (kind, title)
    last_number: dict[str, str] = {}
    out = []
    untitled = None
    for page in pages:
        sections, buffer = [], []
        path = " > ".join(t for _, t in stack)
        for line in page.splitlines():
            k = _kind(line)
            marked = _marked(line)
            opens = bool(k) and k[0] in kinds and (marked or _follows(last_number.get(k[0]), k[1]))
            if opens:
                if buffer:
                    sections.append((path, "\n".join(buffer).strip()))
                kind, number, title = k
                if any(kd == kind for kd, _ in stack):
                    del stack[[kd for kd, _ in stack].index(kind) :]
                while stack and (kind, stack[-1][0]) in outside:
                    stack.pop()
                stack.append((kind, title))
                last_number[kind] = number
                untitled = len(stack) - 1 if not title[_NUMBERED.match(title).end() :].strip(" .:") else None
                path, buffer = " > ".join(t for _, t in stack), [line]
                continue
            if untitled is not None and line.strip():
                if marked or line.strip().isupper():
                    kd, t = stack[untitled]
                    stack[untitled] = (kd, f"{t}. {_MARKUP.sub('', line)}"[:_TITLE_MAX])
                    path = " > ".join(t for _, t in stack)
                untitled = None
            buffer.append(line)
        if buffer:
            sections.append((path, "\n".join(buffer).strip()))
        out.append([(p, b) for p, b in sections if b])
    return out


def _follows(previous: str | None, number: str) -> bool:
    """A plain line continues a numbered series only as its very next item."""
    return bool(previous) and previous.isdigit() and number.isdigit() and int(number) == int(previous) + 1
```

Trong `chunk_document`, thay vòng lặp theo trang:

```python
    pages = [(page["page"], sanitize(page["markdown"])) for page in parsed["pages"]]
    outline = split_document([text for _, text in pages])
    for (page_number, _), sections in zip(pages, outline, strict=True):
        for heading_path, section in sections:
            ...  # thân vòng lặp giữ nguyên; page["page"] thành page_number
```

Biến `open_headings` của `chunk_document` không còn cần: `split_document` tự giữ trạng thái qua các trang.

- [x] **Step 4: Chạy test** — PASS: 6 test mới và 11 test có sẵn.

- [x] **Step 5: Đối chiếu với dữ liệu thật**

```bash
uv run python - <<'PY'
import json, pathlib
from worker.steps.chunking import sanitize, split_document
backup = pathlib.Path("data/parsed-backup")
names = dict(line.split("|", 1) for line in (backup / "documents.txt").read_text().splitlines())
for doc, name in sorted(names.items(), key=lambda item: item[1]):
    pages = json.loads((backup / f"{doc}.parsed.json").read_text())["pages"]
    paths = [p for page in split_document([sanitize(x["markdown"]) for x in pages]) for p, _ in page]
    last = {p.split(" > ")[-1].split(".")[0] for p in paths if p}
    print(f"{sum(1 for x in last if x.casefold().startswith('điều')):4} điều  {name}")
PY
```

Expected: 87, 109, 89, 38, 53, 65, 19, 21, 55 cho chín văn bản luật, 0 cho file tiếng Nhật. So từng con số với số điều thật của văn bản; văn bản nào lệch từ 2 điều trở lên thì tìm dòng bị sót và báo, đừng nới quy tắc cho khớp.

- [x] **Step 6: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `fix(chunking): build the outline from how headings recur, not from markdown levels`

---

## Task 2: S3 biết chunk thuộc điều nào

Câu ngữ cảnh hiện được viết từ nội dung của riêng chunk, nên với một mệnh đề nằm giữa Điều 6 nó chỉ viết được "đoạn này mở đầu nhóm vi phạm bị phạt 18 đến 20 triệu". Đưa thêm tiêu đề thì nó viết được cả đối tượng áp dụng.

**Files:**
- Modify: `worker/steps/chunking.py` (`chunk_document`: thêm `heading_path` vào mỗi child)
- Modify: `shared/llm.py` (`OpenAIProvider.enrich`)
- Modify: `tests/test_chunking.py`, `tests/test_llm_provider.py`

**Interfaces:**
- Consumes: `heading_path` của parent (Task 1)
- Produces: mỗi child trong `chunks.json` có `"heading_path": str | None`; request của `enrich()` gửi `{"id", "heading", "content"}` cho mỗi chunk

- [x] **Step 1: Viết test đỏ**

```python
# tests/test_chunking.py — thêm vào cuối. `chunk_document` đã được import ở đầu file.


def test_a_child_carries_the_heading_of_its_parent():
    """S3 and S4 see children only. Without the heading on the child, the
    stage that writes its context cannot know which article it sits in."""
    body = " ".join(["noi dung"] * 400)
    parsed = {"pages": [{"page": 1, "markdown": f"**Điều 6. Xe ô tô**
{body}"}]}

    chunks = chunk_document(parsed)

    assert chunks["children"]
    assert {c["heading_path"] for c in chunks["children"]} == {"Điều 6. Xe ô tô"}
```

```python
# tests/test_llm_provider.py — thêm vào cuối


def test_enrich_sends_each_chunks_heading_and_asks_for_whom_it_applies(monkeypatch, chunks):
    """A clause in the middle of an article does not say whose fine it is. The
    heading does, and the context sentence is the only place that can carry it
    into the embedding."""
    requests: list[dict] = []
    provider = _openai_returning(monkeypatch, chat_content='{"chunks": []}', requests=requests)

    provider.enrich([{**c, "heading_path": "Điều 6. Xe ô tô"} for c in chunks])

    (request,) = requests
    sent = json.loads(request["messages"][-1]["content"])
    assert {chunk["heading"] for chunk in sent} == {"Điều 6. Xe ô tô"}
    assert {key for chunk in sent for key in chunk} == {"id", "heading", "content"}
    assert "heading" in request["messages"][0]["content"]
```

Test có sẵn `test_enrich_asks_for_schema_bound_output_on_the_configured_model` khẳng định chỉ có `id` và `content` được gửi đi; sửa dòng cuối của nó thành `{"id", "heading", "content"}`.

- [x] **Step 2: Chạy để xác nhận đỏ** — `KeyError: 'heading_path'` ở test chunking; thiếu khoá `heading` ở test provider.

- [x] **Step 3: Sửa code**

```python
# worker/steps/chunking.py — trong chunk_document, dict của child thêm một dòng
                            "heading_path": heading_path or None,
```

```python
# shared/llm.py — trong OpenAIProvider.enrich
        prompt = (
            "For each chunk, write ONE sentence of context situating it, in the "
            "chunk's own language, and choose exactly one category. "
            "Each chunk comes with its heading: the chapter, section and article "
            "it sits under. When the chunk itself does not say what or whom it "
            "applies to and the heading does, the sentence must say so."
        )
        # Only what the task reads: every other field of a chunk would be
        # tokens paid for and ignored.
        sent = [
            {"id": c["id"], "heading": c.get("heading_path") or "", "content": c["content"]}
            for c in chunks
        ]
```

`StubProvider` không đổi: nó không đọc tiêu đề.

- [x] **Step 4: Chạy test, suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(enrich): give the context sentence its chunk's heading`

---

## Task 3: Văn bản được embed mang tên văn bản và tiêu đề

Câu ngữ cảnh do model viết có thể vẫn bỏ sót. Ghép thẳng tên văn bản và tiêu đề vào đầu văn bản được embed thì không phụ thuộc vào model, và không tốn thêm lời gọi nào.

**Files:**
- Modify: `worker/steps/embedding.py` (`contextualize`)
- Modify: `worker/stages.py`, `worker/steps/persistence.py` (mọi nơi gọi `contextualize`)
- Modify: `tests/test_embedding.py`

**Interfaces:**
- Produces: `contextualize(context: str, content: str, title: str | None = None, heading: str | None = None) -> str`

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_embedding.py — thêm vào cuối. `contextualize` đã được import ở đầu file;
# nếu chưa thì thêm `from worker.steps.embedding import contextualize`.


def test_the_embedded_text_opens_with_the_document_and_the_heading():
    """Three articles of one decree share a clause word for word. What tells
    their vectors apart has to be in the text that is embedded."""
    text = contextualize(
        "Mức phạt 18 đến 20 triệu.",
        "b) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;",
        title="Nghị định 168/2024/NĐ-CP",
        heading="Điều 6. Xử phạt người điều khiển xe ô tô",
    )

    assert text.splitlines()[0] == "Nghị định 168/2024/NĐ-CP > Điều 6. Xử phạt người điều khiển xe ô tô"
    assert text.endswith("b) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;")


def test_a_chunk_with_no_title_and_no_heading_is_embedded_as_before():
    assert contextualize("ctx", "noi dung") == "ctx

noi dung"
```

- [ ] **Step 2: Chạy để xác nhận đỏ** — `TypeError: contextualize() got an unexpected keyword argument 'title'`.

- [ ] **Step 3: Sửa `contextualize` và nơi gọi nó**

```python
# worker/steps/embedding.py
def contextualize(
    context: str, content: str, title: str | None = None, heading: str | None = None
) -> str:
    """What gets embedded -- and, in S5, stored as `contextualized`. One
    definition for both, or the stored text could drift away from the text
    its vector was actually computed from.

    The document title and the heading path come first. They are what tells
    apart two provisions that read the same word for word."""
    where = " > ".join(part for part in (title, heading) if part)
    return "

".join(part for part in (where, context, content) if part).strip()
```

Tìm mọi nơi gọi:

```bash
grep -rn "contextualize(" worker/ tests/
```

Ở mỗi nơi gọi trong `worker/`, truyền thêm `title=` (cột `documents.title`, đọc qua `DocumentStateRepository.get`) và `heading=child.get("heading_path")`. S4 và S5 phải truyền **cùng** hai giá trị đó: văn bản lưu ở `contextualized` và văn bản được embed là một.

- [ ] **Step 4: Chạy test, suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(embed): put the document title and heading into the embedded text`

---

## Task 4: Lệnh nạp lại từ checkpoint của S1

**Files:**
- Modify: `worker/stages.py` (thêm `relaunch_from_structure` cạnh `launch`)
- Create: `worker/reingest.py`
- Modify: `tests/test_stage_chain.py`

**Interfaces:**
- Produces: `relaunch_from_structure(document_id: str) -> None`; `python -m worker.reingest [--all | <document_id> ...] [--restore-from data/parsed-backup]`

Hai điều code hiện tại quyết định cách làm (đã đọc ngày 2026-10-11):
- `structure` bỏ qua nếu `staging/<id>/chunks.json` đã tồn tại, nên lệnh phải xoá file đó trước.
- Khi một stage tính lại artifact của nó, `invalidate_downstream` xoá artifact của các stage sau; và S5 upsert rồi `delete_leftovers`, nên chunk cũ thừa ra sẽ bị xoá.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_stage_chain.py — thêm vào cuối. Dùng fixture `seeded_document` và `store`
# có sẵn trong tests/conftest.py; chain chạy eager trong test như các test khác của file.


def test_a_relaunch_rebuilds_chunks_from_the_parse_checkpoint(seeded_document, store):
    """Re-ingesting after a chunking fix must not go back through OCR, and
    must not be skipped by the stale chunks.json of the first run."""
    from worker.stages import launch, relaunch_from_structure

    document_id = str(seeded_document)
    launch(document_id)
    store.put_json(f"staging/{document_id}/chunks.json", {"parents": [], "children": []})
    parsed_before = store.sha256(f"staging/{document_id}/parsed.json")

    relaunch_from_structure(document_id)

    assert store.get_json(f"staging/{document_id}/chunks.json")["children"]
    assert store.sha256(f"staging/{document_id}/parsed.json") == parsed_before
```

Trước khi viết test, đọc ba test đầu của `tests/test_stage_chain.py` để dùng đúng cách file đó khởi chạy chain và đúng kiểu giá trị của `seeded_document`; sửa hai dòng đầu của test cho khớp nếu khác.

- [ ] **Step 2: Chạy để xác nhận đỏ** — `ImportError: cannot import name 'relaunch_from_structure'`.

- [ ] **Step 3: Viết code**

```python
# worker/stages.py — ngay dưới launch()
def relaunch_from_structure(document_id: str) -> None:
    """Runs S2 to S5 again from the parse checkpoint: for a change to
    chunking, enrichment or embedding, which OCR has no part in.

    chunks.json goes first, or S2 would take its checkpoint skip and nothing
    downstream would change."""
    from shared.storage import get_store

    store = get_store()
    if not store.exists(f"staging/{document_id}/parsed.json"):
        raise FileNotFoundError(f"no parse checkpoint for {document_id}; it has to be parsed again")
    store.delete(f"staging/{document_id}/chunks.json")
    chain(structure.s(document_id), enrich.s(), embed.s(), persist.s()).apply_async()
```

```python
# worker/reingest.py
"""Re-runs S2 to S5 for documents that are already ingested.

    python -m worker.reingest --all
    python -m worker.reingest <document_id> [<document_id> ...]
    python -m worker.reingest --all --restore-from data/parsed-backup

--restore-from puts back parse checkpoints that storage has expired, from
files named <document_id>.parsed.json.
"""

import argparse
import pathlib

from sqlalchemy import select

from app.models import Document
from shared.storage import get_store
from worker.db import session_scope
from worker.stages import relaunch_from_structure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_ids", nargs="*")
    parser.add_argument("--all", action="store_true", help="every completed document")
    parser.add_argument("--restore-from", type=pathlib.Path)
    args = parser.parse_args()

    ids = list(args.document_ids)
    if args.all:
        with session_scope() as session:
            done = select(Document.id).where(Document.completed_at.is_not(None))
            ids = [str(row) for row in session.scalars(done)]

    store = get_store()
    for document_id in ids:
        key = f"staging/{document_id}/parsed.json"
        backup = args.restore_from / f"{document_id}.parsed.json" if args.restore_from else None
        if not store.exists(key) and backup and backup.exists():
            store.put(key, backup.read_bytes())
            print(f"restored  {document_id}")
        relaunch_from_structure(document_id)
        print(f"relaunched {document_id}")


if __name__ == "__main__":
    main()
```

`session_scope` nằm ở đâu thì import từ đó (`grep -rn "def session_scope" worker/`).

- [ ] **Step 4: Chạy test, suite đầy đủ, rồi dừng**

Commit message đề xuất: `feat(worker): re-ingest from the parse checkpoint`

---

## Task 5: Nạp lại corpus và kiểm chứng

**Files:** không có code.

- [ ] **Step 1: Hỏi trước khi chạy.** Bước này gọi OpenAI cho khoảng 3.000 chunk (S3 và S4 của 10 tài liệu), ước chừng 0,25 USD.

- [ ] **Step 2: Bật stack với code mới và nạp lại**

```bash
docker compose up -d --wait db rabbitmq redis minio
docker compose up -d --wait --force-recreate worker-cpu worker-llm api
docker compose exec worker-cpu python -m worker.reingest --all --restore-from data/parsed-backup
```

`worker-ocr` không cần chạy: không stage nào của lần này dùng hàng đợi `ocr`. Theo dõi qua `GET /documents` cho tới khi cả 10 tài liệu trở lại `COMPLETED`.

- [ ] **Step 3: Kiểm dữ liệu**

```bash
docker compose exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
select left(d.filename, 28) as doc, count(*) as parents,
       count(*) filter (where p.heading_path like '%Điều %') as under_an_article
from parent_chunks p join documents d on d.id = p.document_id group by 1 order by 1;
SQL
```

Expected: với chín văn bản luật, gần như mọi parent nằm dưới một điều (trừ phần mở đầu và phụ lục). Rồi kiểm riêng ba trang giống chữ nhau: `heading_path` của parent ở trang 14, 20 và 25 của NĐ 168 phải lần lượt chứa "xe ô tô", "xe mô tô" và "xe máy chuyên dùng".

- [ ] **Step 4: Chạy lại kiểm nhãn**

```bash
uv run python -m eval.run --check-labels
```

Expected: `labels           20 reachable`. Chia chunk đã đổi, nên một đoạn trích có thể đã rơi sang chunk khác; nhãn nào bị báo thì sửa `quote`, không sửa `gold_page` trừ khi trang thật sự sai.

- [ ] **Step 5: Hỏi lại câu đã sai** (1 embed + 1 chat)

"Người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu tiền?" Đạt khi câu trả lời là 18 đến 20 triệu đồng **và** trích dẫn chỉ vào trang 14. Dù đạt hay không, đọc `context_detail` trong `query_traces` để biết trang 14 có vào ngữ cảnh không, và ghi lại.

- [ ] **Step 6: Đo con số nền của R1** — đây là Task 12 Step 9 của plan R0 + R1: `uv run python -m eval.run`, ghi `recall@10` và `MRR` vào `docs/retrieval-design.md` §8.

- [ ] **Step 7: Dừng và báo cáo**

---

## Task 6: Cập nhật spec

**Files:**
- Modify: `docs/system-design.md` §6 (S2, S3, S4)

- [ ] **Step 1:** S2: thay "split on markdown headings first" bằng ba quy tắc của Task 1, kèm lý do (bảng số liệu ở đầu plan này) và phần lui về heading markdown cho tài liệu không có tiêu đề đánh số.
- [ ] **Step 2:** S3: ghi rằng mỗi chunk được gửi kèm tiêu đề. S4: ghi dạng của văn bản được embed (`tên văn bản > tiêu đề`, câu ngữ cảnh, nội dung).
- [ ] **Step 3:** Ghi một rủi ro còn lại: cột `child_chunks.tsv` (nền của BM25 ở R2) sinh từ `content`, không gồm tiêu đề, nên tìm theo từ khoá vẫn chưa thấy tên điều. Việc đó thuộc plan của R2.

Commit message đề xuất: `docs(design): describe keyword-based structure and heading-aware enrichment`

---

## Definition of Done

1. Sáu test mới của `split_document` xanh; không từ khoá của loại tài liệu nào nằm trong code; tài liệu không có tiêu đề đánh số chia như cũ.
2. Mỗi child trong `chunks.json` mang `heading_path`; request của `enrich()` gửi kèm tiêu đề.
3. Văn bản được embed và văn bản lưu ở `contextualized` giống hệt nhau, và mở đầu bằng tên văn bản cùng tiêu đề.
4. `python -m worker.reingest --all` nạp lại cả corpus mà không gọi OCR.
5. Parent ở trang 14, 20 và 25 của NĐ 168 mang đúng tiêu đề Điều 6, 7 và 8.
6. `--check-labels` báo 20 nhãn trúng được trên corpus mới.
7. Câu hỏi về ô tô vượt đèn đỏ được trả lời đúng, với trích dẫn chỉ vào trang 14; nếu không, kết quả và trace được ghi lại và báo cáo.
8. Con số nền của R1 được đo trên corpus mới và ghi vào spec.
9. Suite xanh; `ruff check .` và `ruff format --check .` sạch.
