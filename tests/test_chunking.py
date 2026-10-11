from worker.steps.chunking import (
    chunk_document,
    detect_language,
    sanitize,
    split_document,
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


def test_a_section_running_across_a_page_break_keeps_its_heading_path():
    """An article in a decree routinely runs onto the next page, which then
    opens mid-sentence with no heading of its own. Splitting page by page
    with a fresh heading stack filed that continuation under no section at
    all."""
    parsed = {
        "page_count": 2,
        "pages": [
            {
                "page": 1,
                "markdown": "# Chuong II\n\n## Dieu 19\n\n" + "noi dung dieu khoan. " * 120,
                "source": "pymupdf",
                "confidence": 1.0,
            },
            {
                "page": 2,
                "markdown": "tiep theo dieu khoan. " * 120,
                "source": "pymupdf",
                "confidence": 1.0,
            },
        ],
    }

    parents = chunk_document(parsed)["parents"]

    assert {p["heading_path"] for p in parents if p["page_start"] == 2} == {"Chuong II > Dieu 19"}


def _paths(*pages: str) -> list[str]:
    return [path for page in split_document(list(pages)) for path, _ in page]


def test_a_numbered_heading_counts_whatever_markup_the_parser_gave_it():
    """One decree, parsed in one run, held the same kind of line as h2, h3, h4,
    h5, bold and plain text. Only '#' lines were read as headings, so 29 of its
    54 articles were missed and their chunks inherited the wrong heading."""
    page = (
        "##### Điều 5. Mot\nnoi dung 5\n\n"
        "**Điều 6. Xử phạt người điều khiển xe ô tô**\nnoi dung 6\n\n"
        "Điều 7. Xử phạt người điều khiển xe mô tô\nnoi dung 7\n"
    )

    assert _paths(page) == [
        "Điều 5. Mot",
        "Điều 6. Xử phạt người điều khiển xe ô tô",
        "Điều 7. Xử phạt người điều khiển xe mô tô",
    ]


def test_depth_comes_from_how_headings_recur_not_from_the_markdown_level():
    """`# Điều 2` followed by `## Chương II` used to file the whole chapter
    under Article 2. A chapter never falls twice between two articles, so it
    cannot be a subdivision of one."""
    page = (
        "# Điều 1. A\nx\n\n# Điều 2. B\nx\n\n"
        "## Chương II\n\n## HÀNH VI VI PHẠM\n\n"
        "#### Mục 1. QUY TẮC\ny\n\n**Điều 6. Xe ô tô**\nz\n\n"
        "#### Mục 2. KHÁC\n\n**Điều 7. C**\nz\n\n"
        "## Chương III\n\n## THI HÀNH\n\n**Điều 8. D**\nz\n"
    )

    paths = _paths(page)

    assert "Chương II. HÀNH VI VI PHẠM > Mục 1. QUY TẮC > Điều 6. Xe ô tô" in paths
    assert paths[-1] == "Chương III. THI HÀNH > Điều 8. D"


def test_no_list_of_words_is_involved():
    """The kinds are whatever the document numbers. Nothing here names them."""
    page = (
        "# Part 1\n\n## Article 1. Scope\nx\n\n## Article 2. Terms\ny\n\n"
        "# Part 2\n\n## Article 3. Fees\nz\n"
    )

    assert _paths(page) == [
        "Part 1",
        "Part 1 > Article 1. Scope",
        "Part 1 > Article 2. Terms",
        "Part 2",
        "Part 2 > Article 3. Fees",
    ]


def test_a_reference_or_a_quoted_article_is_not_a_heading():
    """ "Điều 24 của Luật này" is a reference, and an amending decree quotes
    whole articles of the decree it amends. Neither opens a section."""
    page = (
        "**Điều 1. Sửa đổi, bổ sung Điều 9**\n"
        "Điều 24 của Luật này được áp dụng.\n"
        "“Điều 9. Cơ sở dữ liệu\n"
        "a) theo quy định tại Điều 6. của Nghị định này\n\n"
        "**Điều 2. Hiệu lực**\nx\n"
    )

    assert _paths(page) == ["Điều 1. Sửa đổi, bổ sung Điều 9", "Điều 2. Hiệu lực"]


def test_the_open_article_carries_over_a_page_break():
    first = "**Điều 5. A**\nx\n\n**Điều 6. Xe ô tô**\nkhoan 1\n"

    assert _paths(first, "khoan 9, tiep theo trang truoc\n")[-1] == "Điều 6. Xe ô tô"


def test_a_document_with_no_numbered_headings_still_follows_its_markdown():
    assert _paths("# A\nx\n\n## B\ny\n\n# C\nz\n") == ["A", "A > B", "C"]


def test_a_child_carries_the_heading_of_its_parent():
    """S3 and S4 see children only. Without the heading on the child, the
    stage that writes its context cannot know which article it sits in."""
    body = " ".join(["noi dung"] * 400)
    page = f"**Điều 5. A**\nx\n\n**Điều 6. Xe ô tô**\n{body}"

    chunks = chunk_document({"pages": [{"page": 1, "markdown": page}]})

    assert chunks["children"]
    assert {c["heading_path"] for c in chunks["children"]} == {"Điều 6. Xe ô tô"}


def _outline(*pages: str) -> list[str]:
    return [path for page in split_document(list(pages)) for path, _ in page]


def test_headings_the_ocr_glued_onto_one_line_are_each_read():
    """`**Mục 3****TÊN MỤC****Điều 18. …**` came out of OCR as one line, and
    Article 18's chunks were filed under Article 17."""
    page = (
        "### Điều 16. A\nx\n\n### Điều 17. Bảo vệ môi trường\ny\n\n"
        "**Mục 3****VI PHẠM KHÁC****Điều 18. Giấy phép lái xe**\nz\n\n"
        "### Mục 4. HẾT\n\n### Điều 19. B\nw\n"
    )

    paths = _outline(page)

    assert any(p.endswith("Điều 18. Giấy phép lái xe") for p in paths)
    assert not any("Điều 17" in p and "Điều 18" in p for p in paths)


def test_a_title_bolded_separately_from_its_marker_is_still_a_heading():
    page = (
        "### Chương II\n\n### MỘT\n\n### Điều 5. A\nx\n\n### Điều 6. B\nx\n\n"
        "### **Chương III** **THUÊ TÀU BAY**\n\n### Điều 7. C\ny\n"
    )

    assert _outline(page)[-1] == "Chương III THUÊ TÀU BAY > Điều 7. C"


def test_a_plain_article_with_a_letter_suffix_continues_the_series():
    """Amended laws insert 10a after 10 and 98h after 98g."""
    page = "### Điều 9. A\nx\n\n### Điều 10. B\ny\n\nĐiều 10a. Chen\nz\n\nĐiều 11. C\nw\n"

    assert _outline(page) == ["Điều 9. A", "Điều 10. B", "Điều 10a. Chen", "Điều 11. C"]


def test_a_sentence_that_starts_with_the_next_article_number_is_not_a_heading():
    """ "Điều 24 của Luật này." wrapped onto its own line right after Article 23
    and became a heading; the context sentences written for it were wrong."""
    page = (
        "### Điều 22. A\nx\n\n### Điều 23. B\ntheo quy định tại\nĐiều 24 của Luật này.\n"
        "Điều 24 Nghị định này có hiệu lực.\n\n### Điều 25. C\ny\n"
    )

    assert _outline(page) == ["Điều 22. A", "Điều 23. B", "Điều 25. C"]


def test_the_first_heading_of_a_kind_may_be_plain_text():
    page = "Điều 1. Phạm vi\nx\n\n### Điều 2. B\ny\n\n### Điều 3. C\nz\n"

    assert _outline(page)[0] == "Điều 1. Phạm vi"


def test_bold_captions_do_not_take_over_a_report_that_has_its_own_headings():
    """Two bold "Table N" captions made a kind, and every markdown heading of
    the report stopped being one."""
    page = (
        "# Annual Report\n\n## Introduction\nx\n\n**Table 1**\nt\n\n"
        "## Methods\ny\n\n**Table 2**\nt\n\n## Results\nz\n"
    )

    assert _outline(page) == [
        "Annual Report",
        "Annual Report > Introduction",
        "Annual Report > Methods",
        "Annual Report > Results",
    ]


def test_an_unnumbered_heading_stays_a_sub_level_of_the_numbered_one_above_it():
    page = "# Chapter 1\n\n## Overview\nx\n\n## Details\ny\n\n# Chapter 2\n\n## Scope\nz\n"

    assert _outline(page) == [
        "Chapter 1",
        "Chapter 1 > Overview",
        "Chapter 1 > Details",
        "Chapter 2",
        "Chapter 2 > Scope",
    ]


def test_an_appendix_after_the_last_article_is_not_filed_under_it():
    """Forms after the final article were all filed under "Điều 39. Trách
    nhiệm thi hành"."""
    page = (
        "## Chương I\n\n## CHUNG\n\n### Điều 1. A\nx\n\n### Điều 2. B\nx\n\n"
        "## Chương II\n\n## THI HÀNH\n\n### Điều 3. C\nx\n\n### Điều 4. Trách nhiệm thi hành\ny\n\n"
        "### Phụ lục I\nf1\n\n### Phụ lục II\nf2\n"
    )

    paths = _outline(page)

    assert paths[-2:] == ["Phụ lục I", "Phụ lục II"]


def test_sections_only_in_the_last_chapter_still_sit_inside_it():
    page = (
        "## Chương I\n\n## CHUNG\n\n### Điều 1. A\nx\n\n### Điều 2. B\nx\n\n"
        "## Chương II\n\n## CỤ THỂ\n\n### Mục 1. MỘT\n\n### Điều 3. C\nx\n\n"
        "### Mục 2. HAI\n\n### Điều 4. D\ny\n"
    )

    assert _outline(page)[-1] == "Chương II. CỤ THỂ > Mục 2. HAI > Điều 4. D"


def test_bold_forms_after_the_last_heading_stand_on_their_own():
    """Twenty pages of forms after a decree's final article were only ever
    bold, so they were not structure, and every chunk of them was filed under
    "Trách nhiệm thi hành"."""
    page = (
        "### Điều 1. A\nx\n\n### Điều 2. Trách nhiệm thi hành\ny\n\n"
        "**Mẫu số 01**\nf1\n\n**Mẫu số 02.** Văn bản đề nghị\nf2\n"
    )

    assert _outline(page)[-2:] == ["Mẫu số 01", "Mẫu số 02. Văn bản đề nghị"]


def test_bold_captions_between_numbered_headings_stay_out_of_the_outline():
    """What makes a bold kind an appendix is that it only begins once every
    other heading is behind it. A caption inside a chapter is not that."""
    page = "# Chapter 1\nx\n\n**Table 1**\nt\n\n# Chapter 2\ny\n\n**Table 2**\nt\n"

    assert _outline(page) == ["Chapter 1", "Chapter 2"]


def test_sections_in_two_chapters_keep_their_chapter():
    """Two chapters between one chapter's last section and the next one's
    first looked like chapters subdividing sections, so a section closed its
    own chapter: 233 chunks of two real decrees lost the chapter they sit in."""
    page = (
        "# Chapter 1\n\n## Section 1\na\n\n## Section 2\nb\n\n# Chapter 2\nc\n\n"
        "# Chapter 3\n\n## Section 1\nd\n\n## Section 2\ne\n"
    )

    assert _outline(page)[-2:] == ["Chapter 3 > Section 1", "Chapter 3 > Section 2"]


def test_a_clause_tagged_as_a_heading_ends_at_the_next_clause():
    """OCR tagged clause 1 of an article as a heading and left clauses 2 to 9
    as text, and all of them were filed under "1. Lịch bay căn cứ:"."""
    page = (
        "### Điều 35. A\nx\n\n### Điều 36. Lịch bay\n\n#### 1. Lịch bay căn cứ:\n"
        "a) nhu cầu thị trường;\nb) năng lực khai thác.\n\n"
        "2. Hãng hàng không có trách nhiệm bồi thường.\n\n3. Hành khách được hỗ trợ.\n"
    )

    sections = [(path, body.split()[0]) for path, body in split_document([page])[0]]

    assert sections[-2:] == [
        ("Điều 36. Lịch bay > 1. Lịch bay căn cứ:", "####"),
        ("Điều 36. Lịch bay", "2."),
    ]


def test_an_amount_at_the_start_of_a_line_is_not_the_next_clause():
    page = (
        "### Điều 1. A\nx\n\n### Điều 2. Mức phạt\n\n#### 1. Phạt tiền từ\n"
        "1.000.000 đồng đến 2.000.000 đồng đối với hành vi sau đây.\n"
    )

    assert _outline(page)[-1] == "Điều 2. Mức phạt > 1. Phạt tiền từ"


def test_a_heading_no_deeper_than_the_numbered_one_closes_it():
    """ "Configuration" was filed under "Step 2 Install", the last step of the
    section before it: the unnumbered heading's own level was never read."""
    page = (
        "## Installation\n\n### Step 1 Download\na\n\n### Step 2 Install\nb\n\n"
        "## Configuration\nc\n\n## Fiscal Year 2024\nd\n\n## Outlook\ne\n"
    )

    paths = _outline(page)

    assert "Configuration" in paths and "Outlook" in paths
    assert not any(" > Configuration" in path or " > Outlook" in path for path in paths)


def test_a_quoted_article_stays_under_the_article_that_quotes_it():
    """An amending decree quotes the article it rewrites, as a heading at the
    same level as its own. The quotation mark is what says it belongs inside."""
    page = (
        "#### Điều 24. Sửa đổi, bổ sung Điều 61\nx\n\n#### “Điều 61. Thẩm quyền\ny\n\n"
        "#### Điều 25. Sửa đổi, bổ sung Điều 62\nz\n"
    )

    assert _outline(page)[1] == "Điều 24. Sửa đổi, bổ sung Điều 61 > “Điều 61. Thẩm quyền"


def test_a_form_keeps_the_title_written_after_its_bold_marker():
    """ "**Mẫu số 04b.** Văn bản chấp thuận" lost its title, and the letterhead
    on the next line was taken for one: "Mẫu số 04b.. BỘ CÔNG AN"."""
    page = (
        "### Điều 1. A\nx\n\n### Điều 2. Trách nhiệm thi hành\ny\n\n"
        "**Mẫu số 01.** Đơn đề nghị\nBỘ CÔNG AN\nf1\n\n"
        "**Mẫu số 02.** Văn bản chấp thuận\nBỘ CÔNG AN\nf2\n"
    )

    assert _outline(page)[-2:] == ["Mẫu số 01. Đơn đề nghị", "Mẫu số 02. Văn bản chấp thuận"]


def test_a_sentence_after_a_bold_marker_is_not_its_title():
    page = (
        "**Điều 1.** Nghị định này có hiệu lực từ ngày ký.\n\n"
        "**Điều 2.** Các Bộ trưởng chịu trách nhiệm thi hành.\n"
    )

    assert _outline(page) == ["Điều 1.", "Điều 2."]


def test_a_number_may_end_in_any_letter():
    """ "04đ" was not a number, so the form was filed under "04d"; "VIIa" was
    not one either, and the chapter's articles had no chapter. A series may
    also skip a letter: Vietnamese has no "f"."""
    page = (
        "## Chương VII\n\n## MỘT\n\n### Điều 98d. A\nx\n\n### Điều 98đ. B\ny\n\n"
        "### Điều 98e. C\nz\n\nĐiều 98g. D\nw\n\n"
        "## Chương VIIa\n\n## HAI\n\n### Điều 99. E\nv\n"
    )

    assert _outline(page)[2:] == [
        "Chương VII. MỘT > Điều 98đ. B",
        "Chương VII. MỘT > Điều 98e. C",
        "Chương VII. MỘT > Điều 98g. D",
        "Chương VIIa. HAI",
        "Chương VIIa. HAI > Điều 99. E",
    ]


def test_a_table_row_in_capitals_is_not_the_name_of_the_heading_above():
    page = "## Table 1\n| ID | NAME |\n| 1 | A |\n\n## Table 2\n| ID | NAME |\n"

    assert _outline(page) == ["Table 1", "Table 2"]


def test_asterisks_inside_text_are_left_alone():
    """Splitting glued headings rewrote every "****", a masked card number
    included."""
    page = "### Điều 1. A\nCard: **** **** **** 1234\n\n### Điều 2. B\ny\n"

    assert "**** **** **** 1234" in split_document([page])[0][0][1]


def test_a_paragraph_tagged_as_a_heading_is_not_one():
    page = "### Điều 1. A\nx\n\n### Điều 2. B\n\n#### " + "rất dài " * 60 + "\ny\n"

    assert _outline(page)[-1] == "Điều 2. B"


def test_a_numbered_section_is_not_an_item_of_the_step_above_it():
    """A clause tagged as a heading belongs to its article whatever level the
    OCR gave it -- but "# 3. Results" is no item of "## Step 2": no list was
    running under that step for it to continue."""
    page = (
        "# 1. Introduction\na\n\n# 2. Methods\n\n## Step 1 Collect\nb\n\n"
        "## Step 2 Clean\nc\n\n# 3. Results\nd\n"
    )

    assert _outline(page)[-1] == "3. Results"


def test_a_clause_heading_shallower_than_its_article_still_belongs_to_it():
    page = (
        "#### Điều 25. A\nx\n\n#### Điều 26. Thuê, mua tàu bay\n1. Một.\n2. Hai.\n\n"
        "### 3. Quy định về thuê tàu bay\ny\n"
    )

    assert _outline(page)[-1] == "Điều 26. Thuê, mua tàu bay > 3. Quy định về thuê tàu bay"
