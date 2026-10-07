"""S1 -- Parse. PyMuPDF for a fast first pass, Chandra OCR for the hard pages.

The task runs on worker-cpu but the model runs inside the chandra container
(vLLM): the worker renders pages and holds HTTP connections, not model
weights. That is why S1 is safe at concurrency 8.
"""

import logging
import re
import time
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

# Pages per OCR request batch: the chandra server's --max-num-seqs, so one
# batch keeps it full without queueing pages the GPU cannot start yet.
_OCR_BATCH_PAGES = 8
# How long and how often to ask /health before calling the server down.
# vLLM's event loop has been seen to stall 17 s preprocessing a burst of
# page images, answering nothing, /health included.
_PROBE_TIMEOUT_S = 30
_PROBE_ATTEMPTS = 3
_PROBE_PAUSE_S = 5
# Chandra's own default is 6. Its repetition check reads a page whose text
# genuinely repeats -- identical table rows, dot leaders, a form -- as the
# model looping, and retries; on a 16 GB card each attempt is about a
# minute, and one such page measured 7 requests and 8 minutes, to return
# the text it had right the first time. 2 still rescues a real loop.
_OCR_MAX_RETRIES = 2

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


class OcrUnavailable(Exception):
    """The OCR server, not a page, is what failed. Raised instead of saving
    the batch's pages degraded: degraded, a scan's pages fall back to an
    empty text layer, and the document reads as blank."""


def _ocr_server_down(ocr_url: str) -> str | None:
    """Why the server is down, or None when /health says it is up.

    Asked only after a batch has come back with errors, because Chandra's
    library returns the same error=True for a page the model could not read
    as for a refused connection, a 500 or a timeout -- the answer here is
    the only way to tell them apart."""
    reason = None
    for attempt in range(_PROBE_ATTEMPTS):
        if attempt:
            time.sleep(_PROBE_PAUSE_S)
        try:
            response = httpx.get(f"{ocr_url}/health", timeout=_PROBE_TIMEOUT_S)
        except httpx.TimeoutException:
            reason = f"not answering /health within {_PROBE_TIMEOUT_S}s"
            continue
        except httpx.TransportError as exc:
            reason = f"unreachable ({type(exc).__name__})"
            continue
        if response.status_code == 200:
            return None
        # vLLM answers 503 once its engine process is gone; the API server
        # itself outlives it, so the port alone proves nothing.
        reason = f"engine not serving (/health {response.status_code})"
    return reason


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
    with it heading_path -- came out empty for every digital PDF, OCR'd
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


def _ocr(data: bytes, page_numbers: list[int], ocr_url: str) -> dict[int, str | None]:
    """Chandra's markdown for each page, or None for a page it still failed
    on after its own retries.

    Raises if the OCR server itself is down -- checked up front, and again
    after any batch with errors, since a server can die mid-document. An
    outage is a stage failure to retry later, not a page to degrade:
    degraded, every page of a scan falls back to its empty text layer, and
    the document would be dead-lettered as blank -- permanently, for what
    was a restart."""
    if not page_numbers:
        return {}

    from chandra.input import load_pdf_images
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem

    httpx.get(f"{ocr_url}/health", timeout=10).raise_for_status()

    manager = InferenceManager(method="vllm")
    out: dict[int, str | None] = {}
    # A batch at a time: a page rendered for the model is ~12 MB, so a
    # 500-page scan rendered all at once would not fit in the worker.
    for start in range(0, len(page_numbers), _OCR_BATCH_PAGES):
        batch = page_numbers[start : start + _OCR_BATCH_PAGES]
        # Chandra's own renderer, so pages reach the model exactly as it was
        # trained to see them: at least 192 DPI, form fields flattened.
        images = load_pdf_images(data, page_range=[n - 1 for n in batch])
        results = manager.generate(
            [BatchInputItem(image=image, prompt_type="ocr_layout") for image in images],
            vllm_api_base=f"{ocr_url}/v1",
            include_images=False,
            max_retries=_OCR_MAX_RETRIES,
        )
        if any(result.error for result in results):
            reason = _ocr_server_down(ocr_url)
            if reason:
                # The whole batch goes, not just its failed pages: a page
                # that did come back is cheap to read again on the retry.
                raise OcrUnavailable(f"OCR server {reason} at {ocr_url}")
        for number, result in zip(batch, results, strict=True):
            out[number] = None if result.error else result.markdown
    return out


def parse_document(
    object_key: str, store: ObjectStore, ocr_url: str | None, ocr_all_pages: bool = False
) -> dict:
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

    if not ocr_url:
        needs_ocr = []
    elif ocr_all_pages or is_scanned(total_chars, document.page_count):
        needs_ocr = list(range(1, document.page_count + 1))
    else:
        needs_ocr = [
            i + 1
            for i, page in enumerate(document)
            if page.find_tables().tables or page.get_images()
        ]

    pages = [
        {"page": i + 1, "markdown": markdown, "source": "pymupdf", "confidence": 1.0}
        for i, markdown in enumerate(_markdown_pages(document))
    ]

    for page_number, markdown in _ocr(data, needs_ocr, ocr_url).items():
        if markdown is None:
            # Keep PyMuPDF's text, mark confidence 0, and log it -- a
            # degraded page beats a dead document, but the degradation has
            # to be visible or nobody knows to check it.
            logger.warning(
                "OCR failed for %s page %d; keeping PyMuPDF text at confidence 0",
                object_key,
                page_number,
            )
            pages[page_number - 1]["confidence"] = 0.0
        else:
            pages[page_number - 1] = {
                "page": page_number,
                "markdown": markdown,
                "source": "chandra",
                "confidence": 1.0,
            }

    return {
        # Extracted AFTER merging in OCR output: on a scan, an h1 only
        # exists in what Chandra rendered into markdown.
        "title": extract_title(document.metadata, pages),
        "page_count": document.page_count,
        "pages": pages,
    }
