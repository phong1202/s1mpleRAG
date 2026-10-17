"""From the name a question gives a document to the document, or to none (R5).

"Tóm tắt Luật Đường bộ" names a law; the browse tool needs its id. A library
holds tens of documents, not millions, so the titles are scored here in Python
rather than through the title index.

Two traps shape the score. "Luật Đường bộ" is also the tail of "Luật Trật tự,
an toàn giao thông đường bộ"; and the decree amending Decree 168 names 168 in
its own title. A document names itself at the head of its title, and another
document only further along -- so a word of the name found among the first
words of a title counts for more than the same word later on. Two documents
that still score close are not guessed between.
"""

import re
import unicodedata
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document

# How many leading words of a title are its name, and how much more a word
# found there counts. Set against the corpus' own titles (plan R5, Task 3).
_HEAD_WORDS = 4
_HEAD_WEIGHT = 3.0
# The best document must beat the second by this factor, or none is chosen.
_CLEAR_LEAD = 1.2
# And it must carry most of the name: a single shared word is not a match.
_MIN_SHARE = 0.6


@dataclass(frozen=True)
class Named:
    id: uuid.UUID
    title: str | None
    filename: str


def _words(text: str, fold: bool = False) -> list[str]:
    """Lower-case words. Diacritics are kept: in Vietnamese "tư", "tự" and
    "tu" are different words. `fold` drops them, for comparing with a filename
    written without accents."""
    text = unicodedata.normalize("NFC", text.casefold())
    if fold:
        text = unicodedata.normalize("NFKD", text.replace("đ", "d"))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"\w+", text)


def _score(name: str, doc: Named) -> float:
    if doc.title:
        asked, title = _words(name), _words(doc.title)
        head, rest = set(title[:_HEAD_WORDS]), set(title[_HEAD_WORDS:])
    else:
        # No title to read: the filename, compared without diacritics.
        asked, head, rest = _words(name, fold=True), set(), set(_words(doc.filename, fold=True))
    if not asked:
        return 0.0
    weights = [_HEAD_WEIGHT if w in head else 1.0 if w in rest else 0.0 for w in asked]
    share = sum(1 for w in weights if w) / len(asked)
    if share < _MIN_SHARE:
        return 0.0
    # A word of the name the title lacks is evidence against it, twice over:
    # "luật giao thông đường bộ" shares four words with the waterway law, and
    # the one it lacks ("bộ") is the one that says it is a different law.
    return sum(weights) / len(asked) * share**2


def best_match(name: str, documents: list[Named]) -> uuid.UUID | None:
    if not _words(name):
        return None
    ranked = sorted(((_score(name, d), d.id) for d in documents), reverse=True)
    if not ranked or ranked[0][0] == 0:
        return None
    if len(ranked) > 1 and ranked[0][0] < _CLEAR_LEAD * ranked[1][0]:
        return None
    return ranked[0][1]


async def find_document(session: AsyncSession, name: str) -> uuid.UUID | None:
    """Among the documents ingested completely; None when no single one is
    clearly what the name refers to."""
    rows = await session.execute(
        select(Document.id, Document.title, Document.filename).where(Document.status == "COMPLETED")
    )
    return best_match(name, [Named(*row) for row in rows])
