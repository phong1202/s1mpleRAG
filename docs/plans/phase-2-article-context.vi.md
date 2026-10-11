# Phase 2 · Ngữ cảnh điều khoản — sửa khâu nạp để chunk biết nó thuộc điều nào · Implementation Plan

> **Cho agentic worker:** REQUIRED SUB-SKILL: dùng superpowers:executing-plans để làm plan này theo từng task. Các bước dùng cú pháp checkbox (`- [ ]`) để theo dõi.

**Mục tiêu:** Mỗi chunk mang theo tên điều, mục, chương và tên văn bản của nó, cả trong `heading_path` lẫn trong văn bản được embed, để các điều khoản giống chữ nhau không còn bị lấy nhầm.

**Kiến trúc:** S2 thôi tin vào cấp heading của markdown và nhận diện cấu trúc văn bản luật bằng từ khoá (Phần, Chương, Mục, Điều). S3 được đưa tiêu đề của chunk khi viết câu ngữ cảnh. S4 ghép tên văn bản và tiêu đề vào đầu văn bản được embed. Một lệnh nạp lại chạy S2 → S5 từ checkpoint của S1, nên không phải OCR lại.

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

## Global Constraints

- **Git:** KHÔNG chạy `git commit`, `git push`, `git merge`. Mỗi task kết thúc bằng dừng và báo cáo.
- **Suite xanh ở cuối mọi task:** `uv run pytest -q --ignore=tests/test_ocr_service.py`, `uv run ruff check .`, `uv run ruff format --check .`.
- **Gọi OpenAI là việc có chủ đích.** Chỉ Task 5 gọi: nạp lại 10 tài liệu là khoảng 3.000 chunk qua S3 và S4, ước chừng 0,25 USD theo bảng giá ở Task 14 của plan R0 + R1. Hỏi trước khi chạy.
- **Không OCR lại.** Nạp lại bắt đầu từ `parsed.json`. Checkpoint trong MinIO tự hết hạn sau 7 ngày (các tài liệu nạp ngày 06 và 07/10); bản sao của cả 10 file nằm ở `data/parsed-backup/` (không commit).
- **Tài liệu không phải văn bản luật không được đổi hành vi:** không có dòng Phần, Chương, Mục, Điều nào thì S2 chia theo heading markdown như hiện nay.
- **`web/` không thuộc plan này.**
- **Sau khi nạp lại, mọi chunk đổi.** Nhãn của `eval/dataset.yaml` neo vào `(sha256, page)` nên vẫn còn nghĩa, nhưng phải chạy lại `--check-labels`. Con số nền của R1 chỉ được đo sau Task 5.

## Cấu trúc file

```
worker/steps/chunking.py      SỬA — cấu trúc theo từ khoá; child mang heading_path (Task 1, 2)
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

## Task 1: S2 nhận diện cấu trúc bằng từ khoá

**Files:**
- Modify: `worker/steps/chunking.py` (`split_sections` và các hằng số ngay trên nó)
- Modify: `tests/test_chunking.py`

**Interfaces:**
- Produces: `split_sections(markdown, stack=None) -> list[tuple[str, str]]`, chữ ký không đổi. Với văn bản luật, `heading_path` có dạng `Chương II. <tên> > Mục 1. <tên> > Điều 6. <tên>`.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/test_chunking.py — thêm vào cuối. `split_sections` đã được import ở đầu file.
def _paths(markdown: str, stack=None) -> list[str]:
    return [path for path, _ in split_sections(markdown, stack)]


def test_an_article_is_a_heading_whatever_markup_the_parser_gave_it():
    """One decree, parsed in one run, held the same kind of line as h2, h3, h4,
    h5, bold and plain text. Only the bold ones were being missed -- 29 of 54
    articles -- and every chunk under them inherited the wrong heading."""
    markdown = (
        "##### Điều 5. Mot\nnoi dung 5\n\n"
        "**Điều 6. Xử phạt người điều khiển xe ô tô**\nnoi dung 6\n\n"
        "Điều 7. Xử phạt người điều khiển xe mô tô\nnoi dung 7\n"
    )

    assert _paths(markdown) == [
        "Điều 5. Mot",
        "Điều 6. Xử phạt người điều khiển xe ô tô",
        "Điều 7. Xử phạt người điều khiển xe mô tô",
    ]


def test_depth_comes_from_the_keyword_not_the_markdown_level():
    """`# Điều 2` followed by `## Chương II` used to file the whole chapter
    under Article 2."""
    markdown = "# Điều 2. Đối tượng\nx\n\n## Chương II\n\n## HÀNH VI VI PHẠM\n\n#### Mục 1. QUY TẮC\ny\n\n**Điều 6. Xe ô tô**\nz\n"

    assert _paths(markdown)[-1] == "Chương II. HÀNH VI VI PHẠM > Mục 1. QUY TẮC > Điều 6. Xe ô tô"


def test_a_new_article_closes_the_one_before_it():
    assert _paths("**Điều 6. A**\nx\n\n**Điều 7. B**\ny\n") == ["Điều 6. A", "Điều 7. B"]


def test_a_reference_or_a_quoted_article_is_not_a_heading():
    """"Điều 24 của Luật này" is a reference, and an amending decree quotes
    whole articles of the decree it amends. Neither opens a section here."""
    markdown = (
        "**Điều 1. Sửa đổi, bổ sung Điều 9**\n"
        "Điều 24 của Luật này được áp dụng.\n"
        "“Điều 9. Cơ sở dữ liệu\n"
        "a) theo quy định tại Điều 6. của Nghị định này\n"
    )

    assert _paths(markdown) == ["Điều 1. Sửa đổi, bổ sung Điều 9"]


def test_the_open_article_carries_over_a_page_break():
    stack: list[str] = []
    split_sections("## Chương I\n\n## QUY ĐỊNH CHUNG\n\n**Điều 6. Xe ô tô**\nkhoan 1\n", stack)

    assert _paths("khoan 9, tiep theo trang truoc\n", stack) == [
        "Chương I. QUY ĐỊNH CHUNG > Điều 6. Xe ô tô"
    ]


def test_a_document_with_no_legal_structure_still_follows_its_markdown():
    assert _paths("# A\nx\n\n## B\ny\n\n# C\nz\n") == ["A", "A > B", "C"]
```

- [ ] **Step 2: Chạy để xác nhận đỏ**

```bash
uv run pytest -q tests/test_chunking.py
```

Expected: 5 FAIL trong 6 test mới (test cuối xanh sẵn, nó giữ hành vi cũ); 11 test có sẵn vẫn PASS.

- [ ] **Step 3: Thay `split_sections`**

Giữ dòng `_HEADING = re.compile(...)`. Thay toàn bộ hàm `split_sections` hiện tại bằng khối dưới đây, và giữ lại docstring cũ của hàm (ba đoạn về heading, về việc dòng tiêu đề ở lại trong thân, và về `stack`).

```python
# worker/steps/chunking.py
# Vietnamese legal structure, outermost first. The markup a parser put on such
# a line is ignored: within one corpus the same "Điều" arrives as h2, h3, h4,
# h5, bold or plain text, so a markdown level says nothing about depth.
_LEGAL_LEVELS = ("Phần", "Chương", "Mục", "Điều")
_LEGAL = re.compile(
    r"^(?:(Phần)\s+(?:thứ\s+\S+|[IVXLC\d]+)|(Chương)\s+[IVXLC\d]+|(Mục)\s+\d+|(Điều)\s+\d+[a-z]?\.)"
    r"(?=[\s.:]|$)"
)
_MARKUP = re.compile(r"^[#\s*_]+|[\s*_]+$")
_TITLE_MAX = 300


def _legal_level(line: str) -> tuple[int, str] | None:
    """(depth, title) when the line opens a part, chapter, section or article.
    A quoted article inside an amending decree starts with a quote mark and a
    reference in running text ("Điều 24 của Luật này") has no full stop after
    the number: neither matches."""
    bare = _MARKUP.sub("", line)
    found = _LEGAL.match(bare)
    if not found:
        return None
    keyword = next(group for group in found.groups() if group)
    return _LEGAL_LEVELS.index(keyword) + 1, bare[:_TITLE_MAX]


def _is_bare_marker(title: str) -> bool:
    """"Chương II" with its name on the following line."""
    return not title[_LEGAL.match(title).end() :].strip(" .:")


def _names_the_marker(line: str) -> bool:
    text = line.strip()
    return bool(_HEADING.match(text)) or text.startswith("**") or text.isupper()


def split_sections(markdown: str, stack: list[str] | None = None) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    stack = [] if stack is None else stack
    path, buffer = " > ".join(stack), []
    untitled: int | None = None

    for line in markdown.splitlines():
        legal = _legal_level(line)
        if legal:
            if buffer:
                sections.append((path, "\n".join(buffer).strip()))
            level, title = legal
            # Everything at this depth or deeper closes, and so does any
            # markdown heading: its level was never comparable.
            stack[:] = [h for h in stack if (seen := _legal_level(h)) and seen[0] < level]
            stack.append(title)
            untitled = len(stack) - 1 if _is_bare_marker(title) else None
            path, buffer = " > ".join(stack), [line]
            continue
        if untitled is not None and line.strip():
            if _names_the_marker(line):
                stack[untitled] = f"{stack[untitled]}. {_MARKUP.sub('', line)}"[:_TITLE_MAX]
                path = " > ".join(stack)
            untitled = None
            buffer.append(line)
            continue
        found = _HEADING.match(line)
        if found and not any(_legal_level(h) for h in stack):
            if buffer:
                sections.append((path, "\n".join(buffer).strip()))
            level = len(found.group(1))
            del stack[level - 1 :]
            stack.append(found.group(2).strip())
            path, buffer = " > ".join(stack), [line]
            continue
        buffer.append(line)

    if buffer:
        sections.append((path, "\n".join(buffer).strip()))
    return [(path, body) for path, body in sections if body]
```

- [ ] **Step 4: Chạy test** — PASS, 17 test trong file này.

- [ ] **Step 5: Đối chiếu với dữ liệu thật**

```bash
uv run python - <<'PY'
import json, pathlib
from worker.steps.chunking import sanitize, split_sections
backup = pathlib.Path("data/parsed-backup")
names = dict(line.split("|", 1) for line in (backup / "documents.txt").read_text().splitlines())
for doc, name in sorted(names.items(), key=lambda item: item[1]):
    stack, articles = [], set()
    for page in json.loads((backup / f"{doc}.parsed.json").read_text())["pages"]:
        for path, _ in split_sections(sanitize(page["markdown"]), stack):
            last = path.split(" > ")[-1] if path else ""
            if last.startswith("Điều"):
                articles.add(last.split(".")[0])
    print(f"{len(articles):4} điều  {name}")
PY
```

Expected, theo lần chạy thử ngày 2026-10-11: 85, 111, 89, 39, 54, 65, 19, 21, 55 điều cho chín văn bản luật và 0 cho file tiếng Nhật. So từng con số với số điều thật của văn bản; lệch từ 2 điều trở lên thì dừng và báo, đừng nới regex cho khớp.

- [ ] **Step 6: Suite đầy đủ, rồi dừng**

Commit message đề xuất: `fix(chunking): read legal structure from keywords, not markdown levels`

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

- [ ] **Step 1: Viết test đỏ**

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

- [ ] **Step 2: Chạy để xác nhận đỏ** — `KeyError: 'heading_path'` ở test chunking; thiếu khoá `heading` ở test provider.

- [ ] **Step 3: Sửa code**

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

- [ ] **Step 4: Chạy test, suite đầy đủ, rồi dừng**

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

- [ ] **Step 1:** S2: thay "split on markdown headings first" bằng mô tả nhận diện theo từ khoá cho văn bản luật, kèm lý do (bảng số liệu ở đầu plan này) và phần lui về heading markdown cho tài liệu khác.
- [ ] **Step 2:** S3: ghi rằng mỗi chunk được gửi kèm tiêu đề. S4: ghi dạng của văn bản được embed (`tên văn bản > tiêu đề`, câu ngữ cảnh, nội dung).
- [ ] **Step 3:** Ghi một rủi ro còn lại: cột `child_chunks.tsv` (nền của BM25 ở R2) sinh từ `content`, không gồm tiêu đề, nên tìm theo từ khoá vẫn chưa thấy tên điều. Việc đó thuộc plan của R2.

Commit message đề xuất: `docs(design): describe keyword-based structure and heading-aware enrichment`

---

## Definition of Done

1. Sáu test mới của `split_sections` xanh, và các tài liệu không phải văn bản luật chia như cũ.
2. Mỗi child trong `chunks.json` mang `heading_path`; request của `enrich()` gửi kèm tiêu đề.
3. Văn bản được embed và văn bản lưu ở `contextualized` giống hệt nhau, và mở đầu bằng tên văn bản cùng tiêu đề.
4. `python -m worker.reingest --all` nạp lại cả corpus mà không gọi OCR.
5. Parent ở trang 14, 20 và 25 của NĐ 168 mang đúng tiêu đề Điều 6, 7 và 8.
6. `--check-labels` báo 20 nhãn trúng được trên corpus mới.
7. Câu hỏi về ô tô vượt đèn đỏ được trả lời đúng, với trích dẫn chỉ vào trang 14; nếu không, kết quả và trace được ghi lại và báo cáo.
8. Con số nền của R1 được đo trên corpus mới và ghi vào spec.
9. Suite xanh; `ruff check .` và `ruff format --check .` sạch.
