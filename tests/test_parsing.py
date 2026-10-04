import hashlib
import io
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

from app.exceptions import AppException, ErrorCode
from worker.chunking import chunk_document
from worker.parsing import extract_title, is_scanned, parse_document

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

    result = parse_document(key, store, docling_url=None)

    assert result["page_count"] == 1
    assert all(p["source"] == "pymupdf" for p in result["pages"])
    assert all(p["confidence"] == 1.0 for p in result["pages"])


def test_encrypted_pdf_raises_a_permanent_error(store, uploaded):
    key = uploaded("encrypted.pdf")

    with pytest.raises(AppException) as exc:
        parse_document(key, store, docling_url=None)

    assert exc.value.error is ErrorCode.PDF_ENCRYPTED


def test_malformed_pdf_raises_the_malformed_error_not_encrypted(store, uploaded):
    """A corrupt file and a password-protected one are different failures
    with different remediations -- mislabeling one as the other sends
    whoever reads documents.last_error looking for a password that would
    not have helped."""
    key = uploaded("malformed.pdf")

    with pytest.raises(AppException) as exc:
        parse_document(key, store, docling_url=None)

    assert exc.value.error is ErrorCode.PDF_MALFORMED


def test_a_document_over_the_page_limit_is_rejected(store, uploaded, monkeypatch):
    """clean_text.pdf is genuinely 1 page, so the cap is set to 0 --
    anything at all exceeds it -- rather than to its real page count."""
    monkeypatch.setattr("worker.parsing.MAX_PAGE_COUNT", 0)
    key = uploaded("clean_text.pdf")

    with pytest.raises(AppException) as exc:
        parse_document(key, store, docling_url=None)

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

    result = parse_document(key, store, docling_url=None)

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

    result = parse_document(key, store, docling_url=None)

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

    assert _headings(parse_document(key, store, docling_url=None)) == []


def test_a_document_without_headings_gains_none(store, uploaded):
    key = uploaded("clean_text.pdf")

    assert _headings(parse_document(key, store, docling_url=None)) == []
