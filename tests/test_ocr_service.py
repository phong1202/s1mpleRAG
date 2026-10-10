"""Runs against the real Chandra server from `docker compose up -d chandra`.

The first start downloads ~10 GB of weights and LOOKS EXACTLY LIKE A HANG --
wait for `docker compose ps chandra` to report healthy before running this.

Chandra is generative, not a character-by-character reader: given the
fixtures' Vietnamese written without diacritics, it returns proper
Vietnamese ("quy" comes back "quý"). Words are therefore compared with the
accents stripped -- and the numbers exactly, since a model that rewrites
words must not be found rewriting figures.
"""

import hashlib
import unicodedata
from pathlib import Path

import httpx
import pymupdf
import pytest

from app.config import get_settings
from shared.storage import get_public_store
from worker.steps.parsing import parse_document

OCR = get_settings().ocr_url
FIXTURES = Path(__file__).parent / "fixtures"


def _plain(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in stripped if not unicodedata.combining(c)).lower()


def _ocr(data: bytes, **kwargs) -> dict:
    store = get_public_store()
    key = f"raw/{hashlib.sha256(data).hexdigest()}.pdf"
    store.put(key, data)
    try:
        return parse_document(key, store, ocr_url=OCR, **kwargs)["pages"][0]
    finally:
        store.delete(key)


@pytest.fixture(scope="module")
def tables_page() -> dict:
    """One OCR pass shared by the table tests -- each costs GPU seconds."""
    return _ocr((FIXTURES / "tables.pdf").read_bytes(), ocr_all_pages=True)


def test_health_answers_only_once_the_model_is_loaded():
    """vLLM answers /health only after the weights are on the GPU -- the
    same check S1 makes before sending a page."""
    assert httpx.get(f"{OCR}/health", timeout=10).status_code == 200


def test_a_borderless_table_comes_back_as_a_table(tables_page):
    """The page Docling flattened into one run of words ("Quy Doanh thu Chi
    phi Q1 38.1 8.0 ...") -- the reason for switching. Every cell has to
    land in its own row."""
    html = tables_page["markdown"]

    assert tables_page["source"] == "chandra"
    assert "<table>" in html
    assert "<td>Q1</td><td>38.1</td><td>8.0</td>" in html
    assert "<td>Q2</td><td>39.4</td><td>8.1</td>" in html


def test_the_words_and_figures_around_the_table_survive(tables_page):
    text = _plain(tables_page["markdown"])

    assert "doanh thu quy 3 nam 2024" in text
    assert "41.7" in text and "8.2" in text


def test_a_scan_with_no_text_layer_is_read_back_as_its_words():
    """The first topics.pdf page rendered to an image: PyMuPDF finds no
    text at all, so every word here came from the model."""
    page = pymupdf.open(FIXTURES / "topics.pdf")[0]
    scan = pymupdf.open()
    scan.new_page(width=page.rect.width, height=page.rect.height).insert_image(
        page.rect, stream=page.get_pixmap(dpi=150).tobytes("jpg")
    )

    result = _ocr(scan.tobytes())

    assert result["source"] == "chandra"
    assert "doanh thu quy 3 nam 2024" in _plain(result["markdown"])
    assert "41.7" in result["markdown"]
