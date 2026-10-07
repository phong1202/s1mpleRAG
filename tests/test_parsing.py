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


def test_an_unreachable_ocr_server_fails_the_stage_rather_than_the_pages(store, uploaded):
    """Degrading every page of a scan to its empty text layer would read as
    a blank document and dead-letter it -- permanently, for an outage. The
    stage has to fail, so it retries once the server is back.

    Over a real socket, no fakes: nothing listens on port 1."""
    from worker.steps.parsing import OcrUnavailable

    key = uploaded("scanned.pdf")

    with pytest.raises(OcrUnavailable, match="unreachable"):
        parse_document(key, store, ocr_url="http://127.0.0.1:1")


@pytest.fixture
def chandra(monkeypatch):
    """A fake vLLM behind httpx.MockTransport, so _ocr's own HTTP code runs.
    Per page: `fail` is refused with a 400 -- the page's own fault; `down`
    gets a refused connection, `silent` a read timeout, `broken` a 503 --
    the server's fault; `repeat_once` first comes back looping. `sent`
    records every page request, `requests` each one's JSON body and read
    timeout.

    Chandra's renderer is faked to hand back the 0-based page index as the
    "image", which travels as the data URL's payload."""
    import json

    import httpx

    state = {
        "fail": set(),
        "down": set(),
        "silent": set(),
        "broken": set(),
        "repeat_once": set(),
        "sent": [],
        "requests": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        page = int(body["messages"][0]["content"][0]["image_url"]["url"].rsplit(",", 1)[1]) + 1
        state["sent"].append(page)
        state["requests"].append((page, body, request.extensions["timeout"]["read"]))
        if page in state["down"]:
            raise httpx.ConnectError("connection refused", request=request)
        if page in state["silent"]:
            raise httpx.ReadTimeout("timed out", request=request)
        if page in state["broken"]:
            return httpx.Response(503)
        if page in state["fail"]:
            return httpx.Response(400, json={"error": "prompt too long"})
        if page in state["repeat_once"]:
            state["repeat_once"].discard(page)
            text = "<p>" + "lặp lại " * 400 + "</p>"
        else:
            text = f'<div data-label="Section-Header"><h1>Page {page} by OCR</h1></div>'
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "worker.steps.parsing._http_client", lambda: httpx.Client(transport=transport)
    )
    monkeypatch.setattr("chandra.input.load_pdf_images", lambda data, page_range: list(page_range))
    monkeypatch.setattr("chandra.model.util.scale_to_fit", lambda image: image)
    monkeypatch.setattr("chandra.model.vllm.image_to_base64", lambda image: str(image))
    return state


def test_an_ocr_server_lost_mid_document_fails_the_stage_not_the_pages(store, uploaded, chandra):
    """2026-10-06: vLLM was OOM-killed mid-run, every page after that came
    back as Chandra's error=True, and 166 pages were saved blank under a
    parse that "succeeded"."""
    from worker.steps.parsing import OcrUnavailable

    chandra["down"] = {1, 2, 3, 4}
    key = uploaded("topics.pdf")

    with pytest.raises(OcrUnavailable, match="unreachable"):
        parse_document(key, store, ocr_url="http://ocr", ocr_all_pages=True)


@pytest.mark.parametrize(
    ("trouble", "reason"),
    [("broken", "HTTP 503"), ("silent", "no answer")],
)
def test_a_dead_engine_or_a_silent_server_is_an_outage_too(
    store, uploaded, chandra, trouble, reason
):
    """503 is vLLM's answer once its engine process is gone. Silence is
    what 2026-10-07 looked like: a GPU shared with a game, /health still
    answering 200 in 0.4 s, and a 5-token request hanging for minutes."""
    from worker.steps.parsing import OcrUnavailable

    chandra[trouble] = {2}
    key = uploaded("topics.pdf")

    with pytest.raises(OcrUnavailable, match=reason):
        parse_document(key, store, ocr_url="http://ocr", ocr_all_pages=True)


def test_a_page_the_server_refuses_is_degraded_alone(store, uploaded, chandra):
    """The server answered, and said no to that page: the failure is the
    page's. It keeps its own text at confidence 0, and every other page
    keeps what OCR read."""
    chandra["fail"] = {2}
    key = uploaded("topics.pdf")

    pages = parse_document(key, store, ocr_url="http://ocr", ocr_all_pages=True)["pages"]

    assert [(p["source"], p["confidence"]) for p in pages] == [
        ("chandra", 1.0),
        ("pymupdf", 0.0),
        ("chandra", 1.0),
        ("chandra", 1.0),
    ]


def test_every_page_request_has_a_bounded_wait(store, uploaded, chandra):
    """Chandra's own client waits 600 s and retries twice, per attempt: a
    wedged server could hold one batch for over an hour, past the time
    slice and RabbitMQ's 30-minute ack timeout alike."""
    from worker.steps.parsing import _PAGE_TIMEOUT_S

    key = uploaded("topics.pdf")

    parse_document(key, store, ocr_url="http://ocr", ocr_all_pages=True)

    assert {timeout for _, _, timeout in chandra["requests"]} == {_PAGE_TIMEOUT_S}


def test_a_looping_page_is_read_again_warmer(store, uploaded, chandra):
    """Chandra's own remedy for a model stuck repeating itself, kept as it
    was: ask again with a higher temperature."""
    chandra["repeat_once"] = {3}
    key = uploaded("topics.pdf")

    pages = parse_document(key, store, ocr_url="http://ocr", ocr_all_pages=True)["pages"]

    assert pages[2]["markdown"] == "# Page 3 by OCR"
    temperatures = [body["temperature"] for page, body, _ in chandra["requests"] if page == 3]
    assert temperatures[0] == 0.0 and temperatures[1] > 0.0


def test_an_interrupted_parse_resumes_from_its_checkpoint(store, uploaded, chandra, monkeypatch):
    """OCR results used to live in memory until the whole file was done, so
    any retry -- a crash, a redelivery, an outage -- read the scan again from
    page 1: ~25 GPU minutes for the 111-page decree. Each batch is handed to
    on_batch as it lands, and a rerun given those pages as `done` sends only
    the rest."""
    from worker.steps.parsing import OcrUnavailable

    monkeypatch.setattr("worker.steps.parsing._OCR_BATCH_PAGES", 2)
    key = uploaded("topics.pdf")
    saved = {}

    chandra["down"] = {3, 4}
    with pytest.raises(OcrUnavailable):
        parse_document(key, store, "http://ocr", ocr_all_pages=True, on_batch=saved.update)
    assert saved == {1: "# Page 1 by OCR", 2: "# Page 2 by OCR"}

    chandra.update(down=set(), sent=[])
    result = parse_document(key, store, "http://ocr", ocr_all_pages=True, done=dict(saved))

    assert sorted(chandra["sent"]) == [3, 4]
    assert [p["markdown"] for p in result["pages"]] == [f"# Page {n} by OCR" for n in range(1, 5)]


def test_a_page_already_failed_on_a_healthy_server_is_not_sent_again(store, uploaded, chandra):
    """None in the checkpoint is a verdict, not a gap: a page the server
    refused would be refused the same way twice."""
    key = uploaded("topics.pdf")

    pages = parse_document(key, store, "http://ocr", ocr_all_pages=True, done={1: None})["pages"]

    assert sorted(chandra["sent"]) == [2, 3, 4]
    assert (pages[0]["source"], pages[0]["confidence"]) == ("pymupdf", 0.0)


def test_a_busy_server_is_retried_sooner_than_a_dead_one(store, uploaded, chandra):
    """Busy is a server that did not answer in time -- seconds to minutes.
    Down is a container restarting -- ~80 s to reload the model."""
    from worker.steps.parsing import OcrUnavailable

    key = uploaded("topics.pdf")
    countdowns = {}
    for name, trouble in (("busy", "silent"), ("down", "down")):
        chandra.update(silent=set(), down=set())
        chandra[trouble] = {1}
        with pytest.raises(OcrUnavailable) as raised:
            parse_document(key, store, "http://ocr", ocr_all_pages=True)
        countdowns[name] = raised.value.countdown

    assert 0 < countdowns["busy"] < countdowns["down"]


def test_a_restart_inside_one_batch_defers_it_rather_than_degrading_it(store, uploaded, chandra):
    """2026-10-07, chandra killed mid-run: the retry's first batch went out
    while vLLM was still reloading and /health answered 200 five seconds
    later -- under a probe-after-the-fact design, 8 pages were kept blank.
    A refused connection is the server's, whatever /health says next."""
    from worker.steps.parsing import OcrUnavailable

    chandra["down"] = {1, 2, 3, 4}
    key = uploaded("topics.pdf")

    with pytest.raises(OcrUnavailable):
        parse_document(key, store, "http://ocr", ocr_all_pages=True)


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

    assert sorted(chandra["sent"]) == [1, 2]
    assert saved == {1: "# Page 1 by OCR", 2: "# Page 2 by OCR"}


def test_a_deadline_reached_on_the_last_batch_still_finishes(store, uploaded, chandra):
    """Nothing is left to continue with, so there is no slice to end."""
    key = uploaded("topics.pdf")

    result = parse_document(key, store, "http://ocr", ocr_all_pages=True, deadline=0)

    assert sorted(chandra["sent"]) == [1, 2, 3, 4]
    assert len(result["pages"]) == 4


# Page 1 of real documents, as S1 produced it on 2026-10-06/07 -- scans as
# Chandra's markdown, digital ones as PyMuPDF's. Trimmed after the opening.
_ND_168_PAGE_1 = """CHÍNH PHỦ

CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

Số: 168/2024/NĐ-CP

Hà Nội, ngày 26 tháng 12 năm 2024

NGHỊ ĐỊNH

Quy định xử phạt vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông \
đường bộ; trừ điểm, phục hồi điểm giấy phép lái xe

<table><tr><td>CÔNG THÔNG TIN ĐIỆN TỬ CHÍNH PHỦ</td></tr><tr><td>ĐẾN GHI: C</td></tr></table>

Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015;"""

_ND_238_PAGE_1 = """CHÍNH PHỦ

CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

Số: 238/2026/NĐ-CP

Hà Nội, ngày 26 tháng 6 năm 2026

### NGHỊ ĐỊNH

**Sửa đổi, bổ sung một số điều của Nghị định số 168/2024/NĐ-CP ngày 26 tháng 12 năm 2024 của \
Chính phủ quy định xử phạt vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao \
thông đường bộ; trừ điểm, phục hồi điểm giấy phép lái xe**

*Căn cứ Luật Tổ chức Chính phủ số 63/2025/QH15;*"""

_LUAT_DUONG_BO_PAGE_1 = """CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
LUẬT
ĐƯỜNG BỘ
Luật Đường bộ số 35/2024/QH15 ngày 27 tháng 6 năm 2024 của Quốc hội,
có hiệu lực kể từ ngày 01 tháng 01 năm 2025, được sửa đổi, bổ sung bởi:
Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam;
### Chương I
NHỮNG QUY ĐỊNH CHUNG"""

_LUAT_TRAT_TU_PAGE_1 = """CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
LUẬT
TRẬT TỰ, AN TOÀN GIAO THÔNG ĐƯỜNG BỘ
Luật Trật tự, an toàn giao thông đường bộ số 36/2024/QH15 ngày 27 tháng
6 năm 2024 của Quốc hội, có hiệu lực kể từ ngày 01 tháng 01 năm 2025, được
Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam;
Quốc hội ban hành Luật Trật tự, an toàn giao thông đường bộ1.
## Chương I"""

# The summary wrapped over five bold lines, as Chandra set it.
_ND_236_PAGE_1 = """CHÍNH PHỦ

CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

Số: 236/2026/NĐ-CP

Hà Nội, ngày 26 tháng 6 năm 2026

**NGHỊ ĐỊNH**

**Sửa đổi, bổ sung một số điều của Nghị định số 151/2024/NĐ-CP
ngày 15 tháng 11 năm 2024 của Chính phủ quy định chi tiết một số điều
và biện pháp thi hành Luật Trật tự, an toàn giao thông đường bộ
được sửa đổi, bổ sung bởi Nghị định số 184/2025/NĐ-CP
ngày 01 tháng 7 năm 2025 của Chính phủ**

*Căn cứ Luật Tổ chức Chính phủ số 63/2025/QH15;*"""

# The portal's "received" stamp, which Chandra reads as a table, sits
# between the type line and the summary here.
_ND_151_PAGE_1 = """CHÍNH PHỦ

CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

Số: 151/2024/NĐ-CP

Hà Nội, ngày 15 tháng 11 năm 2024

NGHỊ ĐỊNH

<table><tr><td colspan="2">CỘNG THÔNG TIN ĐIỆN TỬ CHÍNH PHỦ</td></tr><tr><td>ĐẾN</td>\
<td>Gửi: .....<br/>Ngày: 10.11.2024</td></tr></table>

Quy định chi tiết một số điều và biện pháp thi hành
Luật Trật tự, an toàn giao thông đường bộ

Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015;"""


@pytest.mark.parametrize(
    ("metadata", "page_1", "title"),
    [
        (
            {"title": "2025-01-02 (1)"},
            _ND_168_PAGE_1,
            "Nghị định 168/2024/NĐ-CP quy định xử phạt vi phạm hành chính về trật tự, an toàn "
            "giao thông trong lĩnh vực giao thông đường bộ; trừ điểm, phục hồi điểm giấy phép "
            "lái xe",
        ),
        (
            {"title": "2026-07-01 (1)"},
            _ND_238_PAGE_1,
            "Nghị định 238/2026/NĐ-CP sửa đổi, bổ sung một số điều của Nghị định số 168/2024/NĐ-CP "
            "ngày 26 tháng 12 năm 2024 của Chính phủ quy định xử phạt vi phạm hành chính về trật "
            "tự, an toàn giao thông trong lĩnh vực giao thông đường bộ; trừ điểm, phục hồi điểm "
            "giấy phép lái xe",
        ),
        ({"title": ""}, _LUAT_DUONG_BO_PAGE_1, "Luật Đường bộ"),
        ({"title": ""}, _LUAT_TRAT_TU_PAGE_1, "Luật Trật tự, an toàn giao thông đường bộ"),
        (
            {"title": "2026-06-30 (1)"},
            _ND_236_PAGE_1,
            "Nghị định 236/2026/NĐ-CP sửa đổi, bổ sung một số điều của Nghị định số 151/2024/NĐ-CP "
            "ngày 15 tháng 11 năm 2024 của Chính phủ quy định chi tiết một số điều và biện pháp "
            "thi hành Luật Trật tự, an toàn giao thông đường bộ được sửa đổi, bổ sung bởi Nghị "
            "định số 184/2025/NĐ-CP ngày 01 tháng 7 năm 2025 của Chính phủ",
        ),
        (
            {"title": "2024-12-10 (1)"},
            _ND_151_PAGE_1,
            "Nghị định 151/2024/NĐ-CP quy định chi tiết một số điều và biện pháp thi hành Luật "
            "Trật tự, an toàn giao thông đường bộ",
        ),
    ],
    ids=["nd-168", "nd-238", "luat-duong-bo", "luat-trat-tu", "nd-236", "nd-151"],
)
def test_a_vietnamese_legal_document_is_titled_by_its_type_and_summary(metadata, page_1, title):
    """2026-10-06 gave these "2025-01-02 (1)" -- the scanner's metadata --
    and "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", the national motto that opens
    every one of them. A legal document names itself on page 1: its type on
    a line of its own, its summary right after, its number above."""
    assert extract_title(metadata, [{"markdown": page_1}]) == title


@pytest.mark.parametrize(
    "junk", ["2024-12-10 (1)", "2026-06-30", "Microsoft Word - ND168.docx", "scan0001.pdf"]
)
def test_metadata_that_is_a_date_or_a_file_name_is_not_a_title(junk):
    pages = [{"markdown": "# Báo cáo tài chính quý 3\n\nNội dung."}]

    assert extract_title({"title": junk}, pages) == "Báo cáo tài chính quý 3"


def test_the_national_motto_is_never_a_title():
    """The last-resort first line skips the lines every Vietnamese official
    document opens with."""
    pages = [
        {
            "markdown": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\n"
            "Số: 12/BC-UBND\nHà Nội, ngày 1 tháng 2 năm 2026\nBáo cáo tình hình kinh tế"
        }
    ]

    assert extract_title({}, pages) == "Báo cáo tình hình kinh tế"
