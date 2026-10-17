"""R5: from the name a question uses to one document, or to none."""

import uuid

import pytest

from app.repositories.document_lookup import Named, best_match

TITLES = {
    "duong-bo": ("Luật Đường bộ", "luat-duong-bo_vbhn-49-2026.pdf"),
    "trat-tu": (
        "Luật Trật tự, an toàn giao thông đường bộ",
        "luat-trat-tu-an-toan-giao-thong-duong-bo_vbhn-55-2026.pdf",
    ),
    "duong-thuy": (
        "Luật Giao thông đường thủy nội địa",
        "luat-giao-thong-duong-thuy-noi-dia_vbhn-56-2026.pdf",
    ),
    "nd-168": (
        "Nghị định 168/2024/NĐ-CP quy định xử phạt vi phạm hành chính về trật tự, "
        "an toàn giao thông",
        "nd-168-2024_xu-phat-vi-pham-giao-thong-duong-bo.pdf",
    ),
    "nd-238": (
        "Nghị định 238/2026/NĐ-CP sửa đổi, bổ sung một số điều của Nghị định số 168/2024/NĐ-CP",
        "nd-238-2026_sua-doi-nd-168-2024.pdf",
    ),
    "nd-208": (
        "Nghị định 208/2026/NĐ-CP về vận tải hàng không",
        "nd-208-2026_van-tai-hang-khong.pdf",
    ),
    "untitled": (None, "WS_挨拶.pdf"),
}
IDS = {key: uuid.UUID(int=i) for i, key in enumerate(TITLES, start=1)}
DOCS = [Named(IDS[k], title, filename) for k, (title, filename) in TITLES.items()]


def _resolve(hint):
    found = best_match(hint, DOCS)
    return next((k for k, v in IDS.items() if v == found), None)


@pytest.mark.parametrize(
    "hint, expected",
    [
        ("Luật Đường bộ", "duong-bo"),
        ("luật đường bộ", "duong-bo"),
        ("Luật Trật tự, an toàn giao thông đường bộ", "trat-tu"),
        ("Luật Giao thông đường thủy nội địa", "duong-thuy"),
        ("Nghị định 168", "nd-168"),
        ("NĐ 168/2024", "nd-168"),
        ("Nghị định 238", "nd-238"),
        ("Nghị định về vận tải hàng không", "nd-208"),
        ("luật đường thủy", "duong-thuy"),
    ],
)
def test_a_name_finds_its_document_and_not_the_one_that_quotes_it(hint, expected):
    """Two traps: "Luật Đường bộ" is also the tail of the traffic-order law's
    title, and the decree amending 168 names 168 in its own title."""
    assert _resolve(hint) == expected


@pytest.mark.parametrize(
    "hint",
    # The last names a law this library does not hold, in words four of which
    # are in the waterway law's title.
    ["", "Bộ luật Dân sự", "thông tư 12", "luật giao thông đường bộ"],
)
def test_a_name_that_matches_nothing_clearly_finds_nothing(hint):
    assert _resolve(hint) is None


def test_two_documents_close_in_score_are_not_guessed_between():
    """Rather no document than the wrong one: the question is then searched
    across the corpus, which can still answer it."""
    docs = [
        Named(uuid.UUID(int=1), "Nghị định về xử phạt A", "a.pdf"),
        Named(uuid.UUID(int=2), "Nghị định về xử phạt B", "b.pdf"),
    ]

    assert best_match("Nghị định về xử phạt", docs) is None
