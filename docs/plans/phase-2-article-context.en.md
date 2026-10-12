# Phase 2 · Article context — fixing ingestion so a chunk knows which article it belongs to · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every chunk carries the name of its article, section, chapter and document, both in `heading_path` and in the text that is embedded, so that provisions which read alike are no longer mistaken for one another.

**Architecture:** S2 stops trusting markdown heading levels: it treats every heading or bold line as a heading and infers the hierarchy from how numbered headings recur within the document itself, with no list of keywords. S3 is given a chunk's heading when it writes the context sentence. S4 puts the document title and the heading at the top of the embedded text. A re-ingest command runs S2 → S5 from S1's checkpoint, so nothing is OCR'd again.

**Tech Stack:** Celery · PyMuPDF/Chandra (unchanged) · OpenAI through `shared/llm.py` · pytest · uv · ruff

**Spec:** [`docs/system-design.md`](../system-design.md) §6 (S2, S3, S4). This plan changes what those three stages do; Task 6 updates the spec.

**Status (2026-10-11):** all six tasks are done; the outcome is recorded at the end. The Vietnamese file, `phase-2-article-context.vi.md`, is the one that was executed and carries the ticked steps.

---

## Why this plan exists

On 2026-10-11 the first real question ("what is the fine for a car that runs a red light") was answered with the fine for special-purpose machinery. The fault was in the ingested data, not in the read path:

| Observation | Figure |
| --- | --- |
| The clause "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông" in decree 168/2024 | Appears word for word on 4 pages (14, 20, 25, 28), one per class of vehicle |
| The chunk on page 14 (cars) | Its content, its S3 context sentence and its `heading_path` all lacked the word for "car" |
| How OCR marked up "Điều N." lines within that one decree | h2: 3, h3: 14, h4: 7, bold: 29, plain: 1 |
| S2 at the time | Read only lines starting with `#`, and took the number of `#` as depth |

So 29 of the decree's 54 articles were not recognised as headings, and the ones that were sat in the wrong place in the tree (all of Chapter II once hung under "Article 2"). Keyword search (R2) and reranking (R3) could not have rescued it: both read only the chunk's content.

The algorithm of Task 1 was tried on the parse checkpoints of all 10 documents before the plan was written.

Known limit: it needs numbered headings. A document with only unnumbered headings and noisy levels is still split as before, and can still get the wrong tree.

## Global Constraints

- **Git:** do NOT run `git commit`, `git push`, `git merge`. Every task ends by stopping and reporting.
- **Suite green at the end of every task:** `uv run pytest -q --ignore=tests/test_ocr_service.py`, `uv run ruff check .`, `uv run ruff format --check .`.
- **Calling OpenAI is a deliberate act.** Only Task 5 does: re-ingesting 10 documents is about 3,000 chunks through S3 and S4, roughly USD 0.25. Ask first.
- **No OCR again.** Re-ingest starts from `parsed.json`. Checkpoints in MinIO expire after 7 days; copies of all 10 are in `data/parsed-backup/` (not committed).
- **A document with no numbered headings must not change behaviour:** S2 splits it on its markdown headings as before.
- **No keyword of any document type is written into the code.** Structure is learned from each document.
- **`web/` is not part of this plan.**
- **After the re-ingest every chunk is new.** The labels in `eval/dataset.yaml` are anchored to `(sha256, page)` and still mean something, but `--check-labels` has to be run again. The R1 baseline is measured only after Task 5.

## File structure

```
worker/steps/chunking.py      MODIFY — outline learned from the document; children carry heading_path (Tasks 1, 2)
shared/llm.py                 MODIFY — enrich() sends the heading (Task 2)
worker/steps/embedding.py     MODIFY — contextualize() adds the title and heading (Task 3)
worker/stages.py              MODIFY — title travels in chunks.json; relaunch_from_structure (Tasks 3, 4)
worker/reingest.py            ★ NEW — re-ingest from S1's checkpoint (Task 4)
tests/test_chunking.py        Tasks 1, 2
tests/test_llm_provider.py    Task 2
tests/test_embedding.py       Task 3
tests/test_stage_chain.py     Tasks 3, 4
docs/system-design.md         MODIFY — §6 S2, S3, S4 (Task 6)
```

---

## Task 1: S2 builds the outline from the document itself, with no keyword list

The first draft of this task recognised four words, "Phần, Chương, Mục, Điều", written into the code. This one drops the list. Three rules, none of which names a word:

1. **What a heading is.** A markdown heading, or a line that is bold from end to end. The number of `#` is not used as depth.
2. **A heading's kind.** A heading of the form "words + number" ("Điều 6.", "Chương II", "Article 3", "Phụ lục I") has a kind: the words. A kind that occurs twice or more in the document is structure. A plain line is accepted too when it is the next number of a known kind ("Điều 7." right after "Điều 6.").
3. **Hierarchy.** A heading closes the open heading of its own kind, and everything inside it. Kind A is a subdivision of kind B only if somewhere two or more A sit between two consecutive B; otherwise A is never filed under B.

A document with no numbered headings is split on its markdown headings as before.

Trial run on all 10 checkpoints (2026-10-11): the algorithm found "chương, mục, điều" by itself, and "phụ lục" and "mẫu số" in two decrees; it found 87, 109, 89, 38, 53, 65, 19, 21, 55 articles, against 85, 111, 89, 39, 54, 65, 19, 21, 55 for the keyword version. Page 14 of decree 168 got the path "Chương II … > Mục 1 … > Điều 6 … xe ô tô …".

**Files:**
- Modify: `worker/steps/chunking.py` (add `split_document`; `chunk_document` calls it; `split_sections` stays as the fallback)
- Modify: `tests/test_chunking.py`

**Interfaces:**
- Produces: `split_document(pages: list[str]) -> list[list[tuple[str, str]]]`: for each page, its `(heading_path, body)` sections. It needs the whole document at once, because kinds and nesting are learned from all the headings.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunking.py — append; add `split_document` to the import from worker.steps.chunking


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

- [ ] **Step 2: Run them to verify they fail** — `ImportError: cannot import name 'split_document'`.

- [ ] **Step 3: Add `split_document` and make `chunk_document` use it**

Put the block below right after the existing `split_sections`. `_HEADING` is already in the file.

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

In `chunk_document`, replace the per-page loop:

```python
    pages = [(page["page"], sanitize(page["markdown"])) for page in parsed["pages"]]
    outline = split_document([text for _, text in pages])
    for (page_number, _), sections in zip(pages, outline, strict=True):
        for heading_path, section in sections:
            ...  # the loop body is unchanged; page["page"] becomes page_number
```

`chunk_document`'s `open_headings` variable is no longer needed: `split_document` keeps its own state across pages.

- [ ] **Step 4: Run the tests** — PASS: 6 new tests and the 11 existing ones.

- [ ] **Step 5: Check against real data**

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

Expected: 87, 109, 89, 38, 53, 65, 19, 21, 55 for the nine legal documents, 0 for the Japanese file. Compare each figure with the document's real article count; where one is off by 2 or more, find the missed line and report it rather than loosening a rule to fit.

- [ ] **Step 6: Full suite, then stop**

Proposed commit message: `fix(chunking): build the outline from how headings recur, not from markdown levels`

---

## Task 2: S3 knows which article a chunk belongs to

The context sentence was written from the chunk alone, so for a clause in the middle of Article 6 it could only say "this passage opens the group of violations fined 18 to 20 million". Given the heading, it can also say whom the clause applies to.

**Files:**
- Modify: `worker/steps/chunking.py` (`chunk_document`: add `heading_path` to each child)
- Modify: `shared/llm.py` (`OpenAIProvider.enrich`)
- Modify: `tests/test_chunking.py`, `tests/test_llm_provider.py`

**Interfaces:**
- Consumes: the parent's `heading_path` (Task 1)
- Produces: every child in `chunks.json` has `"heading_path": str | None`; the `enrich()` request sends `{"id", "heading", "content"}` per chunk

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunking.py — append. `chunk_document` is already imported at the top of the file.


def test_a_child_carries_the_heading_of_its_parent():
    """S3 and S4 see children only. Without the heading on the child, the
    stage that writes its context cannot know which article it sits in."""
    body = " ".join(["noi dung"] * 400)
    page = f"**Điều 5. A**\nx\n\n**Điều 6. Xe ô tô**\n{body}"

    chunks = chunk_document({"pages": [{"page": 1, "markdown": page}]})

    assert chunks["children"]
    assert {c["heading_path"] for c in chunks["children"]} == {"Điều 6. Xe ô tô"}
```

```python
# tests/test_llm_provider.py — append


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

The existing `test_enrich_asks_for_schema_bound_output_on_the_configured_model` asserts that only `id` and `content` are sent; change its last line to `{"id", "heading", "content"}`.

- [ ] **Step 2: Run them to verify they fail** — `KeyError: 'heading_path'` in the chunking test; the `heading` key missing in the provider test.

- [ ] **Step 3: Change the code**

```python
# worker/steps/chunking.py — in chunk_document, one more line in the child's dict
                            "heading_path": heading_path or None,
```

```python
# shared/llm.py — in OpenAIProvider.enrich
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

`StubProvider` is unchanged: it does not read the heading.

- [ ] **Step 4: Run the tests, the full suite, then stop**

Proposed commit message: `feat(enrich): give the context sentence its chunk's heading`

---

## Task 3: The embedded text carries the document title and the heading

The model-written context sentence can still leave it out. Putting the title and heading straight into the embedded text does not depend on the model and costs no extra call.

**Files:**
- Modify: `worker/steps/embedding.py` (`contextualize`)
- Modify: `worker/stages.py`, `worker/steps/persistence.py` (every caller of `contextualize`)
- Modify: `tests/test_embedding.py`, `tests/test_stage_chain.py`

**Interfaces:**
- Produces: `contextualize(context: str, content: str, title: str | None = None, heading: str | None = None) -> str`; `chunks.json` gains a `title` key

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embedding.py — append. `contextualize` is already imported at the top of the file.


def test_the_embedded_text_opens_with_the_document_and_the_heading():
    """Three articles of one decree share a clause word for word. What tells
    their vectors apart has to be in the text that is embedded."""
    text = contextualize(
        "Mức phạt 18 đến 20 triệu.",
        "b) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;",
        title="Nghị định 168/2024/NĐ-CP",
        heading="Điều 6. Xử phạt người điều khiển xe ô tô",
    )

    assert text.splitlines()[0] == (
        "Nghị định 168/2024/NĐ-CP > Điều 6. Xử phạt người điều khiển xe ô tô"
    )
    assert text.endswith("b) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;")


def test_a_chunk_with_no_title_and_no_heading_is_embedded_as_before():
    assert contextualize("ctx", "noi dung") == "ctx\n\nnoi dung"
```

- [ ] **Step 2: Run them to verify they fail** — `TypeError: contextualize() got an unexpected keyword argument 'title'`.

- [ ] **Step 3: Change `contextualize` and its callers**

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
    return "\n\n".join(part for part in (where, context, content) if part).strip()
```

S2 writes `documents.title` into `chunks.json` once (key `title`, in the `structure` stage), and both S4 and S5 read `chunks.get("title")` and `child.get("heading_path")`. One source, so the text stored as `contextualized` cannot drift from the text that was embedded. The two callers of `contextualize` are `worker/stages.py` (embed) and `worker/steps/persistence.py`.

- [ ] **Step 4: Run the tests, the full suite, then stop**

Proposed commit message: `feat(embed): put the document title and heading into the embedded text`

---

## Task 4: A command that re-ingests from S1's checkpoint

**Files:**
- Modify: `worker/stages.py` (add `relaunch_from_structure` beside `launch`)
- Create: `worker/reingest.py`
- Modify: `tests/test_stage_chain.py`

**Interfaces:**
- Produces: `relaunch_from_structure(document_id: str) -> None`; `python -m worker.reingest [--all | <document_id> ...] [--restore-from data/parsed-backup]`

Two things in the existing code decide how (read on 2026-10-11):
- `structure` skips when `staging/<id>/chunks.json` exists, so the command has to delete that file first.
- When a stage recomputes its artifact, `invalidate_downstream` deletes the later stages' artifacts; and S5 upserts and then runs `delete_leftovers`, so surplus old chunks are removed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage_chain.py — append. Uses the `seeded_document` fixture from tests/conftest.py;
# the chain runs eagerly in tests, as in the rest of that file.


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

- [ ] **Step 2: Run it to verify it fails** — `ImportError: cannot import name 'relaunch_from_structure'`.

- [ ] **Step 3: Write the code**

```python
# worker/stages.py — right below launch()
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

- [ ] **Step 4: Run the tests, the full suite, then stop**

Proposed commit message: `feat(worker): re-ingest from the parse checkpoint`

---

## Task 5: Re-ingest the corpus and verify

**Files:** no code.

- [ ] **Step 1: Ask before running.** This step calls OpenAI for about 3,000 chunks (S3 and S4 of 10 documents), roughly USD 0.25.

- [ ] **Step 2: Bring the stack up on the new code and re-ingest**

```bash
docker compose up -d --wait db rabbitmq redis minio
docker compose up -d --wait --force-recreate worker-cpu worker-llm api
docker compose exec worker-cpu python -m worker.reingest --all --restore-from data/parsed-backup
```

`worker-ocr` need not run: no stage of this run uses the `ocr` queue. Watch `GET /documents` until all 10 documents are back at `COMPLETED`.

- [ ] **Step 3: Check the data**

```bash
docker compose exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
select left(d.filename, 28) as doc, count(*) as parents,
       count(*) filter (where p.heading_path like '%Điều %') as under_an_article
from parent_chunks p join documents d on d.id = p.document_id group by 1 order by 1;
SQL
```

Expected: for the nine legal documents nearly every parent sits under an article (all but the preamble and appendices). Then check the three look-alike pages: the `heading_path` of the parents on pages 14, 20 and 25 of decree 168 must name cars, motorbikes and special-purpose machinery respectively.

- [ ] **Step 4: Run the label check again**

```bash
uv run python -m eval.run --check-labels
```

Expected: `labels           20 reachable`. Chunking has changed, so a quote may have moved into another chunk; for a label that is reported, fix the `quote`, not the `gold_page`, unless the page really is wrong.

- [ ] **Step 5: Re-ask the question that went wrong** (1 embedding + 1 chat call)

"Người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu tiền?" It passes when the answer is 18 to 20 million VND **and** the citation points at page 14. Pass or not, read `context_detail` in `query_traces` to see whether page 14 reached the context, and record it.

- [ ] **Step 6: Measure the R1 baseline** — this is Task 12 Step 9 of the R0 + R1 plan: `uv run python -m eval.run`, with `recall@10` and `MRR` written into `docs/retrieval-design.md` §8.

- [ ] **Step 7: Stop and report**

---

## Task 6: Update the spec

**Files:**
- Modify: `docs/system-design.md` §6 (S2, S3, S4)

- [ ] **Step 1:** S2: replace "split on markdown headings first" with the three rules of Task 1, the reason for them (the table at the top of this plan), and the fallback to markdown headings for documents with no numbered headings.
- [ ] **Step 2:** S3: record that each chunk is sent with its heading. S4: record the shape of the embedded text (`title > heading`, the context sentence, the content).
- [ ] **Step 3:** Record one remaining risk: `child_chunks.tsv` (what R2's BM25 stands on) is generated from `content`, without the heading, so a keyword search still cannot see an article's title. That belongs to the R2 plan.

Proposed commit message: `docs(design): describe the learned outline and heading-aware enrichment`

---

## Definition of Done

1. The six new `split_document` tests pass; no keyword of any document type is in the code; documents with no numbered headings split as before.
2. Every child in `chunks.json` carries `heading_path`; the `enrich()` request sends the heading.
3. The embedded text and the text stored as `contextualized` are identical, and open with the document title and the heading.
4. `python -m worker.reingest --all` re-ingests the corpus without calling OCR.
5. The parents on pages 14, 20 and 25 of decree 168 carry the headings of Articles 6, 7 and 8.
6. `--check-labels` reports 20 reachable labels on the new corpus.
7. The question about a car running a red light is answered correctly, citing page 14; if not, the result and the trace are recorded and reported.
8. The R1 baseline is measured on the new corpus and written into the spec.
9. Suite green; `ruff check .` and `ruff format --check .` clean.

## Outcome (2026-10-11)

- All 10 documents re-ingested from their parse checkpoints in about 4.5 minutes; 2,946 child chunks; one OpenAI timeout, retried by the pipeline.
- Pages 14, 20 and 25 of decree 168 carry the headings for cars, motorbikes and special-purpose machinery; the embedded text of the page-14 clause opens with the decree's title, then "Chương II … > Mục 1 … > Điều 6 …".
- `--check-labels`: 20 reachable.
- The car question: "18.000.000 đồng đến 20.000.000 đồng", citing page 14, with page 14 first in the context. Before the re-ingest that page was not among the first ten.
- Baseline: `recall@10` 0.850, `MRR` 0.573 over 20 questions; the per-kind figures are in `docs/retrieval-design.md` §8.7.
- Two departures from the plan as first written: the title travels in `chunks.json` instead of being read twice from the database (Task 3), and one older test's "starts with" became "contains", because the stored text now opens with the title and heading.

## After the second review (2026-10-11)

The reviewer ran `split_document` over the real data and found mislabelled chunks: Article 18 of decree 168 sat under Article 17 (OCR had glued three headings onto one line), Chapter III of decree 208 was missed, articles with a letter suffix (10a, 98h) were missed, a sentence "Điều 24 của Luật này." was taken for a heading, and a non-legal document with two bold "Table N" captions lost its whole outline. The code in the repository was fixed and **differs from the block in Task 1 above**:

- Lines glued with `****` are split; a heading whose name is bolded separately is still read; a line opening with a bold numbered marker is a heading too.
- A kind is structure only when it recurs **and** appears at least once as a markdown heading (or the document has no markdown headings at all).
- A plain line is accepted only as the next item of a series (10 → 11, 10 → 10a, or the first item) and written the way that kind is written (a full stop after the number).
- The nesting rule became: A is never filed under B when B is a subdivision of A, nor when A only appears after the last B and nothing that belongs in B appears again.
- An unnumbered markdown heading is a sub-level of the numbered heading that is open.

On the nine legal documents the article series is now unbroken, no number missing (86, 113, 89, 39, 55, 65, 19, 21, 55).

Left unfixed: forms and appendices that are only ever bold ("Mẫu số", "Phụ lục" in decrees 151 and 236) are not recognised as structure and still sit under the final article.

The re-ingest command now locks the document row, refuses a document that is still running, and clears the failure record of the previous run.

After the appendix rule was added (a bold kind that only begins after the last heading of every other kind stands on its own) the corpus was re-ingested once more: 2,982 child chunks; the forms of decree 151 and the appendices of decree 236 stand on their own. Two unnumbered appendices (decree 208 pp. 49–62, decree 151 pp. 34–38) still sit under the final article. Baseline re-measured: `recall@10` 0.750, `MRR` 0.517.

## After the third review (2026-10-11)

This review ran on the fixes of the second and found that they had put a wrong heading on about 260 real chunks. Fixed, each with a failing test first:

- **Sections lost their chapter.** In decree 208 and the Waterway Law two chapters sit between the last section of one chapter and the first section of the next chapter that has any, so chapters looked like subdivisions of sections. 233 chunks carried "Mục 1 > Điều 7" instead of "Chương II > Mục 1 > Điều 7". When two kinds each look like a subdivision of the other, the one the document opened with is now the outer.
- **One clause labelled the clauses after it.** OCR tagged "1. Lịch bay căn cứ:" as a heading and left clauses 2 to 9 as text, so all eight carried the label of clause 1 (30 chunks in 5 documents). A heading written as a list item now ends at the next item of that list.
- **Ordinary markdown documents were mis-nested.** "## Configuration" sat under "### Step 2 Install". An unnumbered heading no deeper than the open numbered heading now closes it. A heading that opens with a quotation mark (an article quoted by an amending decree) stays inside the article that quotes it.
- **Form titles were lost.** "**Mẫu số 04b.** Văn bản chấp thuận…" kept only "Mẫu số 04b.", and the "BỘ CÔNG AN" line after it was taken for the name. The text after a bold marker is now the title unless it ends like a sentence.
- **Numbers with a letter.** "04đ" and "VIIa" are numbers; a series of suffixes need not follow the Latin alphabet (Vietnamese goes d, đ, e, g).
- **Re-ingestion.** A chain that could not be published (broker down) left the document QUEUED for good; its state is now put back. `--force` relaunches a document stuck because its task was lost. One failing document no longer ends the run.
- Minor: `****` inside text is no longer rewritten; a paragraph over 300 characters tagged as a heading does not become one; a table row in capitals is not taken for a heading's name.

Checked on the 10 real documents from their parse checkpoints: 2,986 child chunks, the article series unbroken in all nine legal documents, no clause carrying another clause's label.

Not fixed, recorded in `docs/system-design.md` §6: unnumbered appendices; a numbered appendix marked as a heading only once (7 chunks of decree 151); an article quoted in a footnote as a heading (2 chunks); a numbered heading is never filed under an unnumbered one; two bold "Table N" captions in the last numbered section are taken for appendices; lines inside a code block.

The corpus has been re-ingested with this code (10/10 COMPLETED, 2,986 child chunks). The baseline measured again on it: `recall@10` 0.700, `MRR` 0.463 (spec §8.7). The earlier 0.750 / 0.517 belongs to the previous corpus and is not comparable with the new figure.
