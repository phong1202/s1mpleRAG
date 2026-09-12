from worker.chunking import (
    chunk_document,
    detect_language,
    sanitize,
    split_sections,
)


def test_sanitize_collapses_blank_runs():
    assert sanitize("a\n\n\n\n\nb") == "a\n\nb"


def test_sanitize_skips_fenced_code_blocks():
    """A table's pipe-patching regex would mangle code. A code block must
    pass through untouched."""
    source = "văn bản\n\n```python\nx = [1|2|3]\n```\n\nvăn bản"

    assert "x = [1|2|3]" in sanitize(source)


def test_chunk_index_is_deterministic():
    """This is the natural key that keeps S5 idempotent. Any nondeterminism
    here -- dict ordering, set ordering, a timestamp -- would silently break
    retry safety and only surface much later, as a duplicate row."""
    parsed = {
        "page_count": 2,
        "pages": [
            {
                "page": 1,
                "markdown": "# Tiêu đề\n\n" + "câu văn. " * 200,
                "source": "pymupdf",
                "confidence": 1.0,
            },
            {
                "page": 2,
                "markdown": "## Mục hai\n\n" + "câu khác. " * 200,
                "source": "pymupdf",
                "confidence": 1.0,
            },
        ],
    }

    first = chunk_document(parsed)
    second = chunk_document(parsed)

    assert [c["chunk_index"] for c in first["children"]] == [
        c["chunk_index"] for c in second["children"]
    ]
    assert first == second


def test_children_carry_a_page_number_and_a_parent_index():
    parsed = {
        "page_count": 1,
        "pages": [
            {
                "page": 1,
                "markdown": "# T\n\n" + "câu. " * 300,
                "source": "pymupdf",
                "confidence": 1.0,
            }
        ],
    }

    result = chunk_document(parsed)

    assert all(c["page_number"] == 1 for c in result["children"])
    assert all(
        c["parent_index"] in {p["chunk_index"] for p in result["parents"]}
        for c in result["children"]
    )


def test_chunks_under_twenty_tokens_are_dropped():
    """Dropped BEFORE anything gets billed by the token at S3/S4."""
    parsed = {
        "page_count": 1,
        "pages": [{"page": 1, "markdown": "ngắn", "source": "pymupdf", "confidence": 1.0}],
    }

    assert chunk_document(parsed)["children"] == []


def test_parent_chunks_stay_within_the_token_band():
    parsed = {
        "page_count": 1,
        "pages": [
            {
                "page": 1,
                "markdown": "# T\n\n" + "từ " * 3000,
                "source": "pymupdf",
                "confidence": 1.0,
            }
        ],
    }

    parents = chunk_document(parsed)["parents"]

    assert all(500 <= p["token_count"] <= 1000 for p in parents[:-1])


def test_split_sections_carries_the_heading_path():
    """The heading path is what lets Phase 2's browse tool build a table of
    contents. A deeper heading extends it; a shallower one truncates it."""
    markdown = "# Chuong II\n\nintro\n\n## Dieu 19\n\nbody\n\n# Chuong III\n\nmore"

    paths = [path for path, _ in split_sections(markdown)]

    assert paths == ["Chuong II", "Chuong II > Dieu 19", "Chuong III"]


def test_split_sections_keeps_text_that_precedes_any_heading():
    """A page that's all body with no heading at all must not disappear --
    the common case for PyMuPDF output, which has no markdown structure."""
    assert split_sections("just body text") == [("", "just body text")]


def test_language_is_detected_on_the_parent_and_inherited_by_children():
    """Detected once on the parent, not per child: 150 tokens is too little
    to classify reliably."""
    parsed = {
        "page_count": 1,
        "pages": [
            {
                "page": 1,
                "markdown": "# Invoices\n\n" + "The seller issues an invoice. " * 60,
                "source": "pymupdf",
                "confidence": 1.0,
            }
        ],
    }

    result = chunk_document(parsed)

    assert result["parents"][0]["language"] == "en"
    assert result["parents"][0]["heading_path"] == "Invoices"
    assert {c["language"] for c in result["children"]} == {"en"}


def test_unknown_languages_collapse_to_other():
    """A closed set, same discipline as a taxonomy category: an open one
    would accumulate 'vie', 'vi-VN' and 'vietnamese' within a week."""
    assert detect_language("Lorem ipsum dolor sit amet consectetur") == "other"
