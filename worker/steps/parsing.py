"""S1 -- Parse. PyMuPDF for a fast first pass, Chandra OCR for the hard pages.

The task runs on worker-ocr but the model runs inside the chandra container
(vLLM): the worker renders pages and holds HTTP connections, not model
weights. Its concurrency is sized to the GPU, not the CPU -- see the
worker-ocr service in docker-compose.yml.
"""

import logging
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

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
# How long one page may take. A page measured ~60 s alone and ~2.5 min in
# a full batch; this is past both, and bounds what a wedged server can hold
# a worker for -- Chandra's own client waits 600 s, retried twice, per
# attempt, which could keep one batch past the time slice and RabbitMQ's
# 30-minute ack timeout alike.
_PAGE_TIMEOUT_S = 300
# When to try again: a busy server did not answer in time; a down one is a
# container restarting, ~80 s to reload the model onto the GPU.
_BUSY_RETRY_S = 20
_DOWN_RETRY_S = 60
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
# What else a PDF's title metadata is often left holding: a scanner's stamp
# ("2025-01-02 (1)" on every decree of 2026-10-06), a word processor's
# "Microsoft Word - x.docx", a file name.
_JUNK_METADATA_TITLE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\b|\(\d+\)$|^microsoft \w+ - |\.(pdf|docx?|odt|rtf)$|^(untitled|scan\d*)",
    re.IGNORECASE,
)
# The lines every Vietnamese official document opens with -- the national
# motto, its slogan, the number, the place and date. Never the title.
_PREAMBLE = re.compile(
    r"^(cộng hò?a xã hội chủ nghĩa việt nam|độc lập\s*-\s*tự do\s*-\s*hạnh phúc|số\s*:"
    r"|.*,\s*ngày \d+ tháng \d+ năm \d{4})",
    re.IGNORECASE,
)
# A legal document names its type on a line of its own, its summary right
# below. Laws and their kin carry a name in capitals instead of a sentence.
_LEGAL_TYPES = {
    "luật", "bộ luật", "hiến pháp", "pháp lệnh", "nghị định", "nghị quyết", "thông tư",
    "thông tư liên tịch", "quyết định", "chỉ thị", "lệnh",
}  # fmt: skip
_NAMED_TYPES = {"luật", "bộ luật", "hiến pháp", "pháp lệnh"}
_LEGAL_NUMBER = re.compile(r"\bsố\s*:?\s*(\d+/\d{4}/[\w.-]+)", re.IGNORECASE)
# What follows the summary: the legal bases, the enacting line, the body.
_AFTER_SUMMARY = re.compile(
    r"^(căn cứ|theo đề nghị|quốc hội ban hành|chương |điều \d|phần |luật .* số \d)",
    re.IGNORECASE,
)
_SUMMARY_MAX_CHARS = 400


class OcrUnavailable(Exception):
    """The OCR server, not a page, is what failed. Raised instead of saving
    the batch's pages degraded: degraded, a scan's pages fall back to an
    empty text layer, and the document reads as blank.

    Not the document's failure either: the stage defers on it, like a rate
    limit, after `countdown` seconds -- see worker/pipeline/errors.py."""

    def __init__(self, message: str, countdown: float) -> None:
        super().__init__(message)
        self.countdown = countdown


class ParseContinues(Exception):
    """The parse's time slice is over, with pages still to read. Not a
    failure: everything read so far is checkpointed, and the task requeues
    itself to go on -- see `deadline` in parse_document."""


def _http_client() -> httpx.Client:
    return httpx.Client()


class _ServerTrouble(Exception):
    def __init__(self, reason: str, busy: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.busy = busy


def _read_page(client: httpx.Client, ocr_url: str, image) -> str | None:
    """One page's markdown, or None when the server refused the page itself
    (a 4xx -- e.g. more tokens than the model takes). Raises _ServerTrouble
    when the server did not answer properly: no answer in time, no
    connection, a 5xx.

    Calls vLLM directly rather than through Chandra's client, which turns
    every failure -- a refused connection, a 500, a timeout -- into the same
    error=True it uses for a page, leaving S1 to guess which it was. The
    guess went wrong twice: on 2026-10-06 a dead server's pages were saved
    blank, and on 2026-10-07 a wedged one answered /health in 0.4 s while a
    5-token request hung for minutes. What Chandra contributes is kept as it
    is: its renderer, prompt, image sizing, markdown parser, and its retry
    of a page the model loops on."""
    from chandra.model import util as chandra_util
    from chandra.model import vllm as chandra_vllm
    from chandra.output import parse_markdown
    from chandra.prompts import PROMPT_MAPPING
    from chandra.settings import settings as chandra_settings

    image_url = (
        f"data:image/png;base64,{chandra_vllm.image_to_base64(chandra_util.scale_to_fit(image))}"
    )
    body = {
        "model": chandra_settings.VLLM_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": PROMPT_MAPPING["ocr_layout"]},
                ],
            }
        ],
        "max_tokens": chandra_settings.MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "top_p": 0.1,
    }
    raw = ""
    for attempt in range(_OCR_MAX_RETRIES + 1):
        if attempt:
            # Chandra's remedy for a model stuck repeating itself.
            body |= {"temperature": min(0.2 * attempt, 0.8), "top_p": 0.95}
        try:
            response = client.post(
                f"{ocr_url}/v1/chat/completions", json=body, timeout=_PAGE_TIMEOUT_S
            )
        except httpx.TimeoutException as exc:
            raise _ServerTrouble(
                f"no answer for a page within {_PAGE_TIMEOUT_S}s", busy=True
            ) from exc
        except httpx.TransportError as exc:
            raise _ServerTrouble(f"unreachable ({type(exc).__name__})", busy=False) from exc
        if response.status_code >= 500:
            # vLLM answers 503 once its engine process is gone.
            raise _ServerTrouble(f"HTTP {response.status_code}", busy=False)
        if response.status_code >= 400:
            return None
        raw = response.json()["choices"][0]["message"]["content"] or ""
        looping = chandra_util.detect_repeat_token(raw) or (
            len(raw) > 50 and chandra_util.detect_repeat_token(raw, cut_from_end=50)
        )
        if not looping:
            break
    return parse_markdown(raw, include_images=False)


def is_scanned(total_text_chars: int, page_count: int) -> bool:
    """No text layer means an image. page_count == 0 counts as scanned
    rather than a division by zero."""
    if page_count == 0:
        return True
    return total_text_chars / page_count < SCANNED_CHARS_PER_PAGE


def _plain(line: str) -> str:
    """A line of page markdown as text: no heading marks, no emphasis, no
    HTML tags, composed Unicode -- OCR and PyMuPDF output compared alike."""
    line = re.sub(r"<[^>]+>", " ", _HEADING_MARKUP.sub("", line.strip()))
    return unicodedata.normalize("NFC", line.strip(" *_\t"))


def _sentence_case(text: str) -> str:
    text = text.lower().replace("việt nam", "Việt Nam")
    return text[:1].upper() + text[1:]


def _legal_title(markdown: str) -> str | None:
    """ "Nghị định 168/2024/NĐ-CP quy định xử phạt ..." or "Luật Đường bộ",
    from the opening of page one -- or None if it does not read as a
    Vietnamese legal document."""
    # An HTML block on a scan's first page is a stamp or a box -- the
    # portal's "received" stamp sat between nd-151's type line and its
    # summary -- never part of the title. None marks it to be stepped over.
    lines = [
        None if raw.lstrip().startswith("<") else _plain(raw) for raw in markdown.splitlines()[:60]
    ]
    number = None
    for i, line in enumerate(lines):
        if line is None:
            continue
        if number is None and (found := _LEGAL_NUMBER.search(line)):
            number = found.group(1)
        kind = line.lower()
        if kind not in _LEGAL_TYPES:
            continue

        summary: list[str] = []
        for following in lines[i + 1 :]:
            if following is None:
                continue
            if not following:
                if summary:
                    break
                continue
            if _AFTER_SUMMARY.match(following) or _PREAMBLE.match(following):
                break
            # A law's name is set in capitals, and the prose under it --
            # "Luật Đường bộ số 35/2024/QH15 ngày ..." -- is not part of it.
            if kind in _NAMED_TYPES and not following.isupper():
                break
            summary.append(following)
        text = " ".join(summary)[:_SUMMARY_MAX_CHARS].strip()
        if not text:
            return None
        if kind in _NAMED_TYPES:
            return f"{kind.capitalize()} {_sentence_case(text)}"
        text = _sentence_case(text) if text.isupper() else text
        head = f"{kind.capitalize()} {number}" if number else kind.capitalize()
        return f"{head} {text[:1].lower()}{text[1:]}"
    return None


def extract_title(metadata: dict, pages: list[dict]) -> str | None:
    """A legal document's own type and summary, then PDF metadata, then the
    first h1, then the first line of page one that is not the national
    motto. Returns None when there is none of these -- the caller falls
    back to the filename, the one thing guaranteed to exist.

    The legal title goes first, ahead of metadata: on 2026-10-06 every
    decree carried its scanner's date stamp there, and every one of them
    names itself on page one."""
    legal = _legal_title(pages[0]["markdown"]) if pages else None
    if legal:
        return legal

    title = (metadata or {}).get("title", "").strip()
    if title and title != _REPORTLAB_PLACEHOLDER_TITLE and not _JUNK_METADATA_TITLE.search(title):
        return title

    for page in pages:
        found = _H1.search(page["markdown"])
        if found:
            return found.group(1).strip()

    for line in (pages[0]["markdown"] if pages else "").splitlines():
        # A lower-level heading opening the page is still a heading: its
        # text is the title, its markup is not.
        text = _HEADING_MARKUP.sub("", line.strip())
        if text and not _PREAMBLE.match(_plain(line)):
            return text[:200]

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


def _ocr(
    data: bytes,
    page_numbers: list[int],
    ocr_url: str,
    on_batch: Callable[[dict[int, str | None]], None] | None = None,
    deadline: float | None = None,
) -> dict[int, str | None]:
    """Chandra's markdown for each page, or None for a page it still failed
    on after its own retries. `on_batch` gets everything read so far after
    every batch -- the caller's checkpoint. Past `deadline` (a
    time.monotonic() value), raises ParseContinues between batches.

    Raises OcrUnavailable if the OCR server, not a page, failed: an
    outage is a stage failure to retry later, not a page to degrade.
    Degraded, every page of a scan falls back to its empty text layer, and
    the document would be dead-lettered as blank -- permanently, for what
    was a restart. Only a page the server refused is degraded."""
    if not page_numbers:
        return {}

    from chandra.input import load_pdf_images

    out: dict[int, str | None] = {}
    # A batch at a time: a page rendered for the model is ~12 MB, so a
    # 500-page scan rendered all at once would not fit in the worker.
    for start in range(0, len(page_numbers), _OCR_BATCH_PAGES):
        batch = page_numbers[start : start + _OCR_BATCH_PAGES]
        # Chandra's own renderer, so pages reach the model exactly as it was
        # trained to see them: at least 192 DPI, form fields flattened.
        images = load_pdf_images(data, page_range=[n - 1 for n in batch])
        with _http_client() as client, ThreadPoolExecutor(len(images)) as pool:
            futures = [pool.submit(_read_page, client, ocr_url, image) for image in images]
        results, trouble = [], None
        for future in futures:
            try:
                results.append(future.result())
            except _ServerTrouble as exc:
                trouble = trouble or exc
                results.append(None)
        if trouble:
            # The whole batch goes, not just its failed pages: a page that
            # did come back is cheap to read again on the retry.
            raise OcrUnavailable(
                f"OCR server {trouble.reason} at {ocr_url}",
                countdown=_BUSY_RETRY_S if trouble.busy else _DOWN_RETRY_S,
            )
        for number, markdown in zip(batch, results, strict=True):
            out[number] = markdown
        if on_batch:
            on_batch(dict(out))
        # Checked only after a batch, so every slice reads at least one.
        more = start + _OCR_BATCH_PAGES < len(page_numbers)
        if more and deadline is not None and time.monotonic() >= deadline:
            raise ParseContinues(f"slice over after {len(out)} of {len(page_numbers)} pages")
    return out


def parse_document(
    object_key: str,
    store: ObjectStore,
    ocr_url: str | None,
    ocr_all_pages: bool = False,
    done: dict[int, str | None] | None = None,
    on_batch: Callable[[dict[int, str | None]], None] | None = None,
    deadline: float | None = None,
    on_start: Callable[[int, int, int], None] | None = None,
) -> dict:
    """`done` is OCR an earlier, interrupted run already paid for -- those
    pages are not sent again, a None among them included: a page the server
    refused is refused the same way twice. `on_batch` gets `done` plus everything read since, after
    each batch.

    `deadline` (a time.monotonic() value) ends the call with ParseContinues
    once it passes, between batches -- the caller resumes from `on_batch`'s
    checkpoint. That is what keeps one delivery of a long scan under
    RabbitMQ's consumer_timeout.

    `on_start` gets (page_count, pages already read, pages to read by OCR)
    once the PDF is open and before any OCR -- the moment page_count is
    known, which on a scan is up to half an hour before the result."""
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

    done = {n: done[n] for n in needs_ocr if n in (done or {})}
    todo = [n for n in needs_ocr if n not in done]
    if on_start:
        on_start(document.page_count, len(done), len(needs_ocr))

    def checkpoint(read: dict[int, str | None]) -> None:
        on_batch({**done, **read})

    ocr = {
        **done,
        **_ocr(data, todo, ocr_url, on_batch=checkpoint if on_batch else None, deadline=deadline),
    }
    for page_number, markdown in ocr.items():
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
