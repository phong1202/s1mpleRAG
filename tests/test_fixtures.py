import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest
from pypdf import PdfReader

FIXTURES = Path(__file__).parent / "fixtures"
NAMES = ["clean_text", "tables", "multi_column", "scanned", "encrypted", "malformed"]


def test_all_six_fixtures_exist():
    missing = [n for n in NAMES if not (FIXTURES / f"{n}.pdf").exists()]
    assert missing == []


def test_clean_text_has_a_text_layer():
    doc = pymupdf.open(FIXTURES / "clean_text.pdf")
    chars = sum(len(page.get_text()) for page in doc)
    assert chars / doc.page_count >= 100, "clean_text must clear the scan-detection threshold"


def test_scanned_has_no_text_layer():
    """This is the fixture that forces the "route the whole document to OCR"
    branch."""
    doc = pymupdf.open(FIXTURES / "scanned.pdf")
    chars = sum(len(page.get_text()) for page in doc)
    assert chars / doc.page_count < 100


def test_encrypted_is_actually_encrypted():
    assert PdfReader(FIXTURES / "encrypted.pdf").is_encrypted


def test_malformed_cannot_be_opened():
    # `pytest.raises(Exception)` would pass on a missing file too, so this
    # asserts the file is there and that the failure is a data failure.
    path = FIXTURES / "malformed.pdf"
    assert path.exists(), "fixture has not been generated"
    with pytest.raises(pymupdf.FileDataError):
        pymupdf.open(path)


def test_regenerating_produces_identical_bytes():
    """The six PDFs are committed, so a generator emitting fresh bytes on
    every run would surface as a phantom binary diff for anyone who ran it.

    Comparing two consecutive runs is not enough to catch this: MuPDF writes a
    random trailer /ID, and two runs inside the same second can agree by
    chance. Regenerating in place is safe precisely because the output is
    reproducible -- if this test fails it also leaves the tree dirty, which is
    the correct signal.
    """
    before = {n: (FIXTURES / f"{n}.pdf").read_bytes() for n in NAMES}
    subprocess.run([sys.executable, str(FIXTURES / "generate.py")], check=True, capture_output=True)
    changed = [n for n in NAMES if (FIXTURES / f"{n}.pdf").read_bytes() != before[n]]
    assert changed == []
