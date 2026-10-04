"""S1 -- Parse. PyMuPDF for a fast first pass, Docling for the hard pages.

The task runs on worker-cpu but the REAL work happens inside the docling
container: the worker only holds an HTTP connection, not model weights.
That is why S1 is safe at concurrency 8.
"""

import logging
import re
import unicodedata
from collections import Counter

import httpx
import pymupdf

from app.config import get_settings
from app.exceptions import AppException, ErrorCode
from shared.storage import ObjectStore

logger = logging.getLogger(__name__)

_settings = get_settings()
MAX_PAGE_COUNT = _settings.max_page_count
SCANNED_CHARS_PER_PAGE = 100

_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_HEADING_MARKUP = re.compile(r"^#{1,6}\s+")

_BOLD_FLAG = 16
# A line this much larger than the document's body text reads as a heading.
_HEADING_SIZE_RATIO = 1.15
_HEADING_MAX_CHARS = 200
# Vietnamese legal structure, with or without diacritics, outermost first.
# Decrees set "Điều 5." in bold at body size, so size alone misses it; a
# line opening this way AND set in bold is taken as a heading of that rank.
_STRUCTURE = [
    re.compile(r"^(phần|phan)\s+\S+", re.IGNORECASE),
    re.compile(r"^(chương|chuong)\s+[ivxlcdm\d]+\b", re.IGNORECASE),
    re.compile(r"^(mục|muc)\s+\d+\b", re.IGNORECASE),
    re.compile(r"^(điều|dieu)\s+\d+", re.IGNORECASE),
]

# ReportLab's own hardcoded default (pdfdoc.py) when no title was set on the
# document -- not just a quirk of our own test fixtures. Any real-world PDF
# a ReportLab-based tool produced without an explicit title carries this
# exact string, and trusting it as an author's real title would put a fake
# one on every such document.
_REPORTLAB_PLACEHOLDER_TITLE = "(anonymous)"


def is_scanned(total_text_chars: int, page_count: int) -> bool:
    """No text layer means an image. page_count == 0 counts as scanned
    rather than a division by zero."""
    if page_count == 0:
        return True
    return total_text_chars / page_count < SCANNED_CHARS_PER_PAGE


def extract_title(metadata: dict, pages: list[dict]) -> str | None:
    """PDF metadata, then the first h1, then the first non-empty line of
    page one. Returns None when the PDF has none of the three -- the caller
    falls back to the filename, the one thing guaranteed to exist."""
    title = (metadata or {}).get("title", "").strip()
    if title and title != _REPORTLAB_PLACEHOLDER_TITLE:
        return title

    for page in pages:
        found = _H1.search(page["markdown"])
        if found:
            return found.group(1).strip()

    for line in (pages[0]["markdown"] if pages else "").splitlines():
        if line.strip():
            # A lower-level heading opening the page is still a heading:
            # its text is the title, its markup is not.
            return _HEADING_MARKUP.sub("", line.strip())[:200]

    return None


def _styled_lines(page: pymupdf.Page) -> list[tuple[int, str, float, bool]]:
    """(block number, text, font size, bold) for every non-empty text line."""
    lines = []
    for number, block in enumerate(page.get_text("dict")["blocks"]):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = unicodedata.normalize("NFC", "".join(s["text"] for s in line["spans"]).strip())
            size = round(max(s["size"] for s in spans), 1)
            bold = all(s["flags"] & _BOLD_FLAG or "bold" in s["font"].lower() for s in spans)
            lines.append((number, text, size, bold))
    return lines


def _style_groups(lines: list[tuple[int, str, float, bool]]) -> list[list[tuple]]:
    """Consecutive lines of one block set in the same type: a heading that
    wraps onto a second line is still one heading."""
    groups: list[list[tuple]] = []
    for line in lines:
        if groups and (line[0], line[2], line[3]) == (groups[-1][-1][0], *groups[-1][-1][2:]):
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def _heading_key(text: str, size: float, bold: bool, body_size: float) -> tuple | None:
    """Where a heading sorts among the document's headings -- larger type
    first, then bold, then structural rank, so a bold "Chương" outranks a
    bold "Điều" set in the same type -- or None for body text."""
    if len(text) > _HEADING_MAX_CHARS:
        return None
    rank = next((i + 1 for i, p in enumerate(_STRUCTURE) if p.match(text)), 0)
    if size >= body_size * _HEADING_SIZE_RATIO or (bold and rank):
        return (-size, -int(bold), rank)
    return None


def _markdown_pages(document: pymupdf.Document) -> list[str]:
    """PyMuPDF's text with the headings marked up as markdown. Plain
    get_text() carries no structure at all, so S2's heading split -- and
    with it heading_path -- came out empty for every digital PDF, docling
    pages being the only exception. Typography is the signal: the size most
    characters are set in is body text; headings are set larger, or in bold
    with a structural opening. Levels go by how prominent each style is
    across the whole document, so they agree from one page to the next."""
    pages = [_style_groups(_styled_lines(page)) for page in document]

    weight: Counter[float] = Counter()
    for groups in pages:
        for group in groups:
            for _, text, size, _ in group:
                weight[size] += len(text)
    body_size = weight.most_common(1)[0][0] if weight else 0.0

    keyed = [
        [
            (group, _heading_key(" ".join(line[1] for line in group), *group[0][2:], body_size))
            for group in groups
        ]
        for groups in pages
    ]
    styles = sorted({key for groups in keyed for _, key in groups if key})
    level = {key: min(rank, 6) for rank, key in enumerate(styles, start=1)}

    markdown = []
    for groups in keyed:
        out = []
        for group, key in groups:
            if key:
                out.append("#" * level[key] + " " + " ".join(line[1] for line in group))
            else:
                out.extend(line[1] for line in group)
        markdown.append("\n".join(out))
    return markdown


def parse_document(object_key: str, store: ObjectStore, docling_url: str | None) -> dict:
    data = store.get(object_key)

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise AppException(ErrorCode.PDF_MALFORMED, f"Could not open PDF: {exc}") from exc

    if document.needs_pass:
        raise AppException(ErrorCode.PDF_ENCRYPTED)
    if document.page_count > MAX_PAGE_COUNT:
        raise AppException(
            ErrorCode.PDF_TOO_LARGE, f"{document.page_count} pages, cap is {MAX_PAGE_COUNT}"
        )

    texts = [page.get_text() for page in document]
    total_chars = sum(len(t) for t in texts)

    if is_scanned(total_chars, document.page_count) and docling_url:
        needs_docling = list(range(1, document.page_count + 1))
    elif docling_url:
        needs_docling = [
            i + 1
            for i, page in enumerate(document)
            if page.find_tables().tables or page.get_images()
        ]
    else:
        needs_docling = []

    pages = [
        {"page": i + 1, "markdown": markdown, "source": "pymupdf", "confidence": 1.0}
        for i, markdown in enumerate(_markdown_pages(document))
    ]

    for page_number in needs_docling:
        try:
            response = httpx.post(
                f"{docling_url}/parse",
                json={"object_key": object_key, "pages": [page_number]},
                timeout=_settings.docling_page_timeout_s,
            )
            response.raise_for_status()
            parsed = response.json()["pages"][0]
            pages[page_number - 1] = {
                "page": page_number,
                "markdown": parsed["markdown"],
                "source": "docling",
                "confidence": parsed["confidence"],
            }
        except Exception:
            # Timeout or error: keep PyMuPDF's raw text, mark confidence 0
            # and log it -- a degraded page beats a dead document, but the
            # degradation has to be visible or nobody knows to check it.
            logger.warning(
                "Docling failed for %s page %d; keeping PyMuPDF text at confidence 0",
                object_key,
                page_number,
                exc_info=True,
            )
            pages[page_number - 1]["confidence"] = 0.0

    return {
        # Extracted AFTER merging in Docling: an h1 usually only exists on
        # pages Docling rendered into markdown.
        "title": extract_title(document.metadata, pages),
        "page_count": document.page_count,
        "pages": pages,
    }
