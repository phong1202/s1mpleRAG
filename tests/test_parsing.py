from pathlib import Path

import pytest

from app.exceptions import AppException, ErrorCode
from worker.parsing import extract_title, is_scanned, parse_document

FIXTURES = Path(__file__).parent / "fixtures"


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
