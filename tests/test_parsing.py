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

    def fake_ocr(data, page_numbers, ocr_url, on_batch=None, deadline=None):
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


def test_an_unreachable_ocr_server_fails_the_stage_rather_than_the_pages(
    store, uploaded, monkeypatch
):
    """Degrading every page of a scan to its empty text layer would read as
    a blank document and dead-letter it -- permanently, for an outage. The
    stage has to fail, so it retries once the server is back.

    Through Chandra's real library, not a fake: what is under test is that
    a refused connection, which the library reports exactly like an
    unreadable page, still comes out as an outage."""
    from worker.steps.parsing import OcrUnavailable

    monkeypatch.setattr("worker.steps.parsing._PROBE_PAUSE_S", 0)
    key = uploaded("scanned.pdf")

    with pytest.raises(OcrUnavailable, match="unreachable"):
        parse_document(key, store, ocr_url="http://127.0.0.1:1")


@pytest.fixture
def chandra(monkeypatch):
    """Fakes Chandra's library one level below `ocr_calls`, so _ocr itself
    runs: pages listed in `fail` come back as Chandra's error=True, and
    `health` lists what successive GET /health calls answer -- a status
    code, or an exception to raise. Once the list is empty, /health is 200.
    `fail_once` pages fail only the first call they are in -- a server
    that went down and came back inside one batch. `sent` records every page
    handed to the model.

    The distinction under test lives in _ocr: Chandra's library turns a
    dead server, a 500 and a timeout into the very same error=True it uses
    for a page the model could not read."""
    from types import SimpleNamespace

    import httpx

    state = {"fail": set(), "fail_once": set(), "health": [], "sent": []}

    class FakeManager:
        def __init__(self, method):
            pass

        def generate(self, items, **kwargs):
            state["sent"].extend(item.image + 1 for item in items)
            failing = state["fail"] | state["fail_once"]
            state["fail_once"] = set()
            # load_pdf_images is faked to hand back the 0-based page index
            # as the "image", so each item knows which page it is.
            return [
                SimpleNamespace(error=True, markdown="")
                if item.image + 1 in failing
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

    return [exc_type("down", request=httpx.Request("GET", "http://ocr/health"))] * 9


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
        ([503, 503, 503], "engine"),  # vLLM's /health once EngineCore is dead
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


def test_an_interrupted_parse_resumes_from_its_checkpoint(store, uploaded, chandra, monkeypatch):
    """OCR results used to live in memory until the whole file was done, so
    any retry -- a crash, a redelivery, an outage -- read the scan again from
    page 1: ~25 GPU minutes for the 111-page decree. Each batch is handed to
    on_batch as it lands, and a rerun given those pages as `done` sends only
    the rest."""
    import httpx

    from worker.steps.parsing import OcrUnavailable

    monkeypatch.setattr("worker.steps.parsing._OCR_BATCH_PAGES", 2)
    key = uploaded("topics.pdf")
    saved = {}

    chandra["fail"] = {3}
    chandra["health"] = _down(httpx.ConnectError)
    with pytest.raises(OcrUnavailable):
        parse_document(key, store, "http://ocr", ocr_all_pages=True, on_batch=saved.update)
    assert saved == {1: "# Page 1 by OCR", 2: "# Page 2 by OCR"}

    chandra.update(fail=set(), health=[], sent=[])
    result = parse_document(key, store, "http://ocr", ocr_all_pages=True, done=dict(saved))

    assert chandra["sent"] == [3, 4]
    assert [p["markdown"] for p in result["pages"]] == [f"# Page {n} by OCR" for n in range(1, 5)]


def test_a_page_already_failed_on_a_healthy_server_is_not_sent_again(store, uploaded, chandra):
    """None in the checkpoint is a verdict, not a gap: a page the model kept
    looping on costs minutes each time, and would fail the same way."""
    key = uploaded("topics.pdf")

    pages = parse_document(key, store, "http://ocr", ocr_all_pages=True, done={1: None})["pages"]

    assert chandra["sent"] == [2, 3, 4]
    assert (pages[0]["source"], pages[0]["confidence"]) == ("pymupdf", 0.0)


def test_a_busy_server_is_retried_sooner_than_a_dead_one(store, uploaded, chandra):
    """Busy is vLLM alive but its event loop stalled on a burst of images --
    seconds. Down is a container restarting -- ~80 s to reload the model."""
    import httpx

    from worker.steps.parsing import OcrUnavailable

    key = uploaded("topics.pdf")
    chandra["fail"] = {1}
    countdowns = {}
    for name, exc_type in (("busy", httpx.ReadTimeout), ("down", httpx.ConnectError)):
        chandra["health"] = _down(exc_type)
        with pytest.raises(OcrUnavailable) as raised:
            parse_document(key, store, "http://ocr", ocr_all_pages=True)
        countdowns[name] = raised.value.countdown

    assert 0 < countdowns["busy"] < countdowns["down"]


def test_no_health_check_runs_before_a_batch_that_succeeds(store, uploaded, chandra):
    """The preflight /health call was the one that timed out on 2026-10-06:
    asked of a server too busy to answer, about pages it would have read."""
    import httpx

    chandra["health"] = _down(httpx.ReadTimeout)
    key = uploaded("topics.pdf")

    pages = parse_document(key, store, "http://ocr", ocr_all_pages=True)["pages"]

    assert {p["source"] for p in pages} == {"chandra"}


def test_pages_lost_to_a_restart_inside_one_batch_are_read_again(store, uploaded, chandra):
    """2026-10-07, chandra killed mid-run: the retry's first batch went out
    while vLLM was still reloading, every page got a connection error, and
    by the time /health was asked, 5 s later, it answered 200 -- so 8 pages
    were kept blank as if the model had failed them. A healthy answer after
    the fact does not prove the errors were the pages': they are sent once
    more before that verdict."""
    chandra["fail_once"] = {1, 2, 3, 4}
    key = uploaded("topics.pdf")

    pages = parse_document(key, store, "http://ocr", ocr_all_pages=True)["pages"]

    assert {(p["source"], p["confidence"]) for p in pages} == {("chandra", 1.0)}
    assert chandra["sent"] == [1, 2, 3, 4, 1, 2, 3, 4]


def test_a_parse_past_its_deadline_stops_after_the_batch_in_hand(
    store, uploaded, chandra, monkeypatch
):
    """RabbitMQ takes a message back from a consumer that holds it unacked
    past consumer_timeout (30 min), and Celery then exits: on 2026-10-06
    that ended the 111-page decree, and the worker with it. A parse runs in
    slices well under that: past its deadline it stops between batches --
    never before finishing one, so every slice makes progress -- with the
    batch already checkpointed."""
    from worker.steps.parsing import ParseContinues

    monkeypatch.setattr("worker.steps.parsing._OCR_BATCH_PAGES", 2)
    key = uploaded("topics.pdf")
    saved = {}

    with pytest.raises(ParseContinues):
        parse_document(
            key, store, "http://ocr", ocr_all_pages=True, on_batch=saved.update, deadline=0
        )

    assert chandra["sent"] == [1, 2]
    assert saved == {1: "# Page 1 by OCR", 2: "# Page 2 by OCR"}


def test_a_deadline_reached_on_the_last_batch_still_finishes(store, uploaded, chandra):
    """Nothing is left to continue with, so there is no slice to end."""
    key = uploaded("topics.pdf")

    result = parse_document(key, store, "http://ocr", ocr_all_pages=True, deadline=0)

    assert chandra["sent"] == [1, 2, 3, 4]
    assert len(result["pages"]) == 4
