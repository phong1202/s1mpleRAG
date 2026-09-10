"""Runs against the real docling_service from `docker compose up -d docling`.

Model load takes real time (minutes on CPU, faster on GPU) -- wait for
`docker compose logs -f docling` to show models_loaded before running this.
"""

import hashlib
from pathlib import Path

import httpx
import pytest

from app.config import get_settings
from shared.storage import get_public_store

DOCLING = get_settings().docling_url


@pytest.fixture(scope="module")
def uploaded_tables_pdf():
    data = (Path(__file__).parent / "fixtures" / "tables.pdf").read_bytes()
    key = f"raw/{hashlib.sha256(data).hexdigest()}.pdf"
    get_public_store().put(key, data)
    yield key
    get_public_store().delete(key)


def test_health_is_green_only_after_models_are_loaded():
    """The healthcheck has to wait for the model to load, not just for the
    port to open -- otherwise a worker fires requests at a service that
    isn't ready yet, and you end up debugging the wrong thing."""
    response = httpx.get(f"{DOCLING}/health", timeout=10)

    assert response.status_code == 200
    assert response.json()["models_loaded"] is True


def test_parse_returns_markdown_for_requested_pages(uploaded_tables_pdf):
    response = httpx.post(
        f"{DOCLING}/parse",
        json={"object_key": uploaded_tables_pdf, "pages": [1]},
        timeout=300,
    )

    assert response.status_code == 200
    pages = response.json()["pages"]
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert len(pages[0]["markdown"]) > 0
    assert 0.0 <= pages[0]["confidence"] <= 1.0


@pytest.fixture(scope="module")
def uploaded_three_page_pdf():
    """A dedicated 3-page fixture, not one of tests/fixtures/*.pdf: this
    checks docling_service's own per-page slicing, not parsing semantics,
    and each page needs content distinct enough to prove it."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph

    path = Path(__file__).parent / "fixtures" / "_docling_three_page_probe.pdf"
    styles = getSampleStyleSheet()
    doc = BaseDocTemplate(str(path), pagesize=A4, invariant=1)
    doc.addPageTemplates([PageTemplate(id="t", frames=[Frame(50, 50, 495, 742, id="f")])])
    flow = []
    for i in range(1, 4):
        flow.append(
            Paragraph(
                f"Unique content for page number {i}, matches no other page.", styles["Normal"]
            )
        )
        flow.append(PageBreak())
    doc.build(flow)

    data = path.read_bytes()
    path.unlink()
    key = f"raw/{hashlib.sha256(data).hexdigest()}.pdf"
    get_public_store().put(key, data)
    yield key
    get_public_store().delete(key)


def test_parse_returns_distinct_markdown_and_confidence_per_page(uploaded_three_page_pdf):
    """Each page must get its own slice of the document and its own score --
    returning the whole document's text and a fixed confidence for every
    entry would silently duplicate content and discard a real signal."""
    response = httpx.post(
        f"{DOCLING}/parse",
        json={"object_key": uploaded_three_page_pdf, "pages": [1, 2, 3]},
        timeout=300,
    )

    assert response.status_code == 200
    pages = response.json()["pages"]
    assert [p["page"] for p in pages] == [1, 2, 3]

    markdowns = [p["markdown"] for p in pages]
    assert len(set(markdowns)) == 3, (
        "each page must carry its own text, not the whole document repeated"
    )
    for i, markdown in enumerate(markdowns, start=1):
        assert f"page number {i}" in markdown

    for p in pages:
        assert 0.0 <= p["confidence"] <= 1.0
