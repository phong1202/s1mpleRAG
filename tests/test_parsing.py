import hashlib
import io
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

from app.exceptions import AppException, ErrorCode
from worker.steps.chunking import chunk_document
from worker.steps.parsing import extract_title, is_scanned, parse_document

FIXTURES = Path(__file__).parent / "fixtures"
STYLES = getSampleStyleSheet()
BOLD_BODY = ParagraphStyle("bold_body", parent=STYLES["BodyText"], fontName="Helvetica-Bold")
BODY = "Noi dung quy dinh chi tiet ve hoa don dien tu va chung tu cua doanh nghiep. " * 6


def test_scanned_detection_uses_chars_per_page():
    assert is_scanned(total_text_chars=50, page_count=3) is True
    assert is_scanned(total_text_chars=5000, page_count=3) is False


def test_scanned_detection_handles_zero_pages():
    """Dividing by zero here is an easy crash to hit and a hard one to trace."""
    assert is_scanned(total_text_chars=0, page_count=0) is True


def test_title_falls_back_from_metadata_to_heading_to_first_line():
    """documents.title, from Task 8b. The order matters: PDF metadata is the
    only source an author set on purpose."""
    pages = [{"markdown": "# Decree 123/2020\n\nbody"}]

    assert extract_title({"title": "Annual Report 2024"}, pages) == "Annual Report 2024"
    assert extract_title({}, pages) == "Decree 123/2020"
    assert extract_title({}, [{"markdown": "\n\nNghi dinh so 123\n\nbody"}]) == "Nghi dinh so 123"
    assert extract_title({}, []) is None


def test_reportlab_placeholder_title_is_not_a_real_title():
    """'(anonymous)' is reportlab's own hardcoded default (pdfdoc.py) when
    no title was set -- not just our fixtures, any real-world document a
    ReportLab-based tool produced without an explicit title carries the
    exact same string. Trusting it as an author's real title would put a
    fake one on every such document."""
    pages = [{"markdown": "# Real Heading\n\nbody"}]

    assert extract_title({"title": "(anonymous)"}, pages) == "Real Heading"


def test_a_document_opening_on_a_lower_heading_gets_its_text_as_title():
    """No metadata title and no h1, but the first line is an h2 -- as both
    OCR output and typographic headings produce. The title is the heading's
    text, not "## " in front of it."""
    pages = [{"markdown": "## Chuong 1 - Bao cao\n\nbody"}]

    assert extract_title({}, pages) == "Chuong 1 - Bao cao"


def test_clean_text_is_parsed_by_pymupdf_alone(store, uploaded):
    key = uploaded("clean_text.pdf")

    result = parse_document(key, store, ocr_url=None)

    assert result["page_count"] == 1
    assert all(p["source"] == "pymupdf" for p in result["pages"])
    assert all(p["confidence"] == 1.0 for p in result["pages"])


def test_encrypted_pdf_raises_a_permanent_error(store, uploaded):
    key = uploaded("encrypted.pdf")

    with pytest.raises(AppException) as exc:
        parse_document(key, store, ocr_url=None)

    assert exc.value.error is ErrorCode.PDF_ENCRYPTED


def test_malformed_pdf_raises_the_malformed_error_not_encrypted(store, uploaded):
    """A corrupt file and a password-protected one are different failures
    with different remediations -- mislabeling one as the other sends
    whoever reads documents.last_error looking for a password that would
    not have helped."""
    key = uploaded("malformed.pdf")

    with pytest.raises(AppException) as exc:
        parse_document(key, store, ocr_url=None)

    assert exc.value.error is ErrorCode.PDF_MALFORMED


def test_a_document_over_the_page_limit_is_rejected(store, uploaded, monkeypatch):
    """clean_text.pdf is genuinely 1 page, so the cap is set to 0 --
    anything at all exceeds it -- rather than to its real page count."""
    monkeypatch.setattr("worker.steps.parsing.MAX_PAGE_COUNT", 0)
    key = uploaded("clean_text.pdf")

    with pytest.raises(AppException) as exc:
        parse_document(key, store, ocr_url=None)

    assert exc.value.error is ErrorCode.PDF_TOO_LARGE


@pytest.fixture
def upload_pdf(store):
    """Builds a PDF from reportlab flowables, puts it in MinIO, and returns
    its key -- deleting every one it made afterward."""
    keys: list[str] = []

    def _upload(*flowables) -> str:
        buffer = io.BytesIO()
        SimpleDocTemplate(buffer, pagesize=A4, invariant=1).build(list(flowables))
        key = f"raw/{hashlib.sha256(buffer.getvalue()).hexdigest()}.pdf"
        store.put(key, buffer.getvalue())
        keys.append(key)
        return key

    yield _upload
    for key in keys:
        store.delete(key)


def _headings(result: dict) -> list[str]:
    return [
        line
        for page in result["pages"]
        for line in page["markdown"].splitlines()
        if line.startswith("#")
    ]


def test_larger_type_becomes_markdown_headings_ranked_by_size(store, upload_pdf):
    """Plain get_text() carries no structure, so every digital PDF came out
    of S1 with no heading at all -- and S2 with heading_path empty for
    every parent, which is what Phase 2's browse tool builds its table of
    contents from."""
    key = upload_pdf(
        Paragraph("Chuong I - Quy dinh chung", STYLES["Heading1"]),
        Paragraph("Dieu 1. Pham vi", STYLES["Heading2"]),
        Paragraph(BODY, STYLES["BodyText"]),
    )

    result = parse_document(key, store, ocr_url=None)

    assert _headings(result) == ["# Chuong I - Quy dinh chung", "## Dieu 1. Pham vi"]
    parents = chunk_document(result)["parents"]
    assert parents[0]["heading_path"] == "Chuong I - Quy dinh chung > Dieu 1. Pham vi"


def test_bold_structural_lines_at_body_size_nest_by_their_rank(store, upload_pdf):
    """Decrees commonly set Chuong and Dieu in bold at body size: size alone
    sees no heading. The structural opening ranks them, Chuong over Dieu."""
    key = upload_pdf(
        Paragraph("Chuong I", BOLD_BODY),
        Paragraph("Dieu 1. Pham vi dieu chinh", BOLD_BODY),
        Paragraph(BODY, STYLES["BodyText"]),
    )

    result = parse_document(key, store, ocr_url=None)

    assert _headings(result) == ["# Chuong I", "## Dieu 1. Pham vi dieu chinh"]


def test_body_text_is_not_mistaken_for_a_heading(store, upload_pdf):
    """A sentence that merely opens like an article, and a short bold line
    with no structural opening, both stay body text: either rule alone would
    turn every cross-reference or bold label into a section."""
    key = upload_pdf(
        Paragraph("Dieu 5 cua Luat nay quy dinh ro trach nhiem cua cac ben.", STYLES["BodyText"]),
        Paragraph("Luu y", BOLD_BODY),
        Paragraph(BODY, STYLES["BodyText"]),
    )

    assert _headings(parse_document(key, store, ocr_url=None)) == []


def test_a_document_without_headings_gains_none(store, uploaded):
    key = uploaded("clean_text.pdf")

    assert _headings(parse_document(key, store, ocr_url=None)) == []


@pytest.fixture
def ocr_calls(monkeypatch):
    """Stands in for Chandra: records which pages were sent, and answers
    each with markdown -- or with None, Chandra's "failed after retries", for
    the page numbers listed in `ocr_calls.fail`."""
    calls = {"pages": [], "fail": set()}

    def fake_ocr(data, page_numbers, ocr_url):
        calls["pages"] = list(page_numbers)
        return {n: None if n in calls["fail"] else f"# Page {n} by OCR" for n in page_numbers}

    monkeypatch.setattr("worker.steps.parsing._ocr", fake_ocr)
    return calls


def test_a_scan_sends_every_page_to_ocr(store, uploaded, ocr_calls):
    key = uploaded("scanned.pdf")

    result = parse_document(key, store, ocr_url="http://ocr")

    assert ocr_calls["pages"] == list(range(1, result["page_count"] + 1))
    assert {p["source"] for p in result["pages"]} == {"chandra"}


def test_a_plain_digital_page_skips_ocr_unless_every_page_is_asked_for(store, uploaded, ocr_calls):
    """clean_text has a text layer, no table and no image: PyMuPDF alone is
    enough -- unless OCR_ALL_PAGES asks for Chandra's reading of everything,
    which is the only way a borderless table PyMuPDF cannot see gets read."""
    key = uploaded("clean_text.pdf")

    assert parse_document(key, store, ocr_url="http://ocr")["pages"][0]["source"] == "pymupdf"
    assert ocr_calls["pages"] == []

    result = parse_document(key, store, ocr_url="http://ocr", ocr_all_pages=True)
    assert ocr_calls["pages"] == [1]
    assert result["pages"][0]["source"] == "chandra"


def test_a_page_ocr_fails_on_keeps_its_own_text_at_confidence_zero(store, uploaded, ocr_calls):
    """A degraded page beats a dead document -- as long as the degradation
    shows, which is what confidence 0 is for."""
    ocr_calls["fail"] = {1}
    key = uploaded("clean_text.pdf")

    page = parse_document(key, store, ocr_url="http://ocr", ocr_all_pages=True)["pages"][0]

    assert page["source"] == "pymupdf"
    assert page["confidence"] == 0.0
    assert "Doanh thu" in page["markdown"]


def test_an_unreachable_ocr_server_fails_the_stage_rather_than_the_pages(store, uploaded):
    """Degrading every page of a scan to its empty text layer would read as
    a blank document and dead-letter it -- permanently, for an outage. The
    stage has to fail, so it retries once the server is back."""
    import httpx

    key = uploaded("scanned.pdf")

    with pytest.raises(httpx.ConnectError):
        parse_document(key, store, ocr_url="http://127.0.0.1:1")


@pytest.fixture
def chandra(monkeypatch):
    """Fakes Chandra's library one level below `ocr_calls`, so _ocr itself
    runs: pages listed in `fail` come back as Chandra's error=True, and
    `health` lists what successive GET /health calls answer -- a status
    code, or an exception to raise. Once the list is empty, /health is 200.

    The distinction under test lives in _ocr: Chandra's library turns a
    dead server, a 500 and a timeout into the very same error=True it uses
    for a page the model could not read."""
    from types import SimpleNamespace

    import httpx

    state = {"fail": set(), "health": []}

    class FakeManager:
        def __init__(self, method):
            pass

        def generate(self, items, **kwargs):
            # load_pdf_images is faked to hand back the 0-based page index
            # as the "image", so each item knows which page it is.
            return [
                SimpleNamespace(error=True, markdown="")
                if item.image + 1 in state["fail"]
                else SimpleNamespace(error=False, markdown=f"# Page {item.image + 1} by OCR")
                for item in items
            ]

    def fake_get(url, timeout):
        outcome = state["health"].pop(0) if state["health"] else 200
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, request=httpx.Request("GET", url))

    monkeypatch.setattr("chandra.model.InferenceManager", FakeManager)
    monkeypatch.setattr("chandra.input.load_pdf_images", lambda data, page_range: list(page_range))
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("worker.steps.parsing._PROBE_PAUSE_S", 0)
    return state


def _down(exc_type):
    import httpx

    return [200] + [exc_type("down", request=httpx.Request("GET", "http://ocr/health"))] * 9


def test_an_ocr_server_lost_mid_document_fails_the_stage_not_the_pages(store, uploaded, chandra):
    """2026-10-06: vLLM was OOM-killed mid-run, every page after that came
    back error=True, and 166 pages were saved blank under a parse that
    "succeeded". The preflight check had passed, long before."""
    import httpx

    from worker.steps.parsing import OcrUnavailable

    chandra["fail"] = {1, 2, 3, 4}
    chandra["health"] = _down(httpx.ConnectError)
    key = uploaded("topics.pdf")

    with pytest.raises(OcrUnavailable, match="unreachable"):
        parse_document(key, store, ocr_url="http://ocr", ocr_all_pages=True)


@pytest.mark.parametrize(
    ("health", "reason"),
    [
        ([200, 503, 503, 503], "engine"),  # vLLM's /health once EngineCore is dead
        ("timeout", "not answering"),  # alive but too busy to answer
    ],
)
def test_a_dead_engine_or_a_silent_server_is_an_outage_too(
    store, uploaded, chandra, health, reason
):
    import httpx

    from worker.steps.parsing import OcrUnavailable

    chandra["fail"] = {2}
    chandra["health"] = _down(httpx.ReadTimeout) if health == "timeout" else health
    key = uploaded("topics.pdf")

    with pytest.raises(OcrUnavailable, match=reason):
        parse_document(key, store, ocr_url="http://ocr", ocr_all_pages=True)


def test_a_page_failing_on_a_healthy_server_is_degraded_alone(store, uploaded, chandra):
    """The other side of the probe: the server answers, so the failure
    belongs to that page. It keeps its own text at confidence 0, and every
    other page keeps what OCR read."""
    chandra["fail"] = {2}
    key = uploaded("topics.pdf")

    pages = parse_document(key, store, ocr_url="http://ocr", ocr_all_pages=True)["pages"]

    assert [(p["source"], p["confidence"]) for p in pages] == [
        ("chandra", 1.0),
        ("pymupdf", 0.0),
        ("chandra", 1.0),
        ("chandra", 1.0),
    ]
