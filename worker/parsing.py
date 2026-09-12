"""S1 -- Parse. PyMuPDF for a fast first pass, Docling for the hard pages.

The task runs on worker-cpu but the REAL work happens inside the docling
container: the worker only holds an HTTP connection, not model weights.
That is why S1 is safe at concurrency 8.
"""

import logging
import re

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
            return line.strip()[:200]

    return None


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
        {"page": i + 1, "markdown": text, "source": "pymupdf", "confidence": 1.0}
        for i, text in enumerate(texts)
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
