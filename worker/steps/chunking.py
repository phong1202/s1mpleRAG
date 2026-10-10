"""S2 -- Structure. Pure CPU, no network calls. The only stage where a
rerun costs a few milliseconds.
"""

import re

import py3langid as langid
import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")

PARENT_MIN, PARENT_MAX = 500, 1000
CHILD_MIN, CHILD_MAX = 100, 200
DROP_BELOW = 20
LANGUAGES = {"vi", "en"}

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_BLANK_RUN = re.compile(r"\n{3,}")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def sanitize(markdown: str) -> str:
    """Collapses blank-line runs -- but never touches a fenced code block:
    a table's pipe-patching regex would otherwise mangle code syntax."""
    blocks = _FENCE.split(markdown)
    fences = _FENCE.findall(markdown)

    cleaned = [_BLANK_RUN.sub("\n\n", b).strip() for b in blocks]

    out = []
    for i, block in enumerate(cleaned):
        out.append(block)
        if i < len(fences):
            out.append(fences[i])
    return "\n\n".join(x for x in out if x)


def _split_by_tokens(text: str, target_min: int, target_max: int) -> list[str]:
    """Never exceeds target_max -- except a single word that alone already
    does, since a word can't be split without breaking it. The final piece,
    if under target_min, is merged back into the one before it: a chunk a
    bit over the ceiling beats an orphan under the floor.

    Tallies token counts word by word instead of re-encoding the whole
    growing buffer on every word -- that used to take a 200-page document
    ~25s to chunk, instead of ~0.4s."""
    words = text.split()
    chunks: list[list[str]] = []
    token_counts: list[int] = []
    current: list[str] = []
    current_tokens = 0

    for word in words:
        word_tokens = count_tokens((" " if current else "") + word)
        if current and current_tokens + word_tokens > target_max:
            chunks.append(current)
            token_counts.append(current_tokens)
            current, current_tokens = [], 0
            # This word now starts a fresh chunk, so it carries no leading
            # space -- re-cost it without one, or the tally silently drifts
            # low by exactly the space-vs-no-space token gap (3 vs 1 for a
            # word with combining diacritics, observed on "từ").
            word_tokens = count_tokens(word)
        current.append(word)
        current_tokens += word_tokens

    if current:
        chunks.append(current)
        token_counts.append(current_tokens)

    if len(chunks) > 1 and token_counts[-1] < target_min:
        chunks[-2].extend(chunks.pop())

    return [" ".join(c) for c in chunks]


def split_sections(markdown: str, stack: list[str] | None = None) -> list[tuple[str, str]]:
    """Splits on markdown headings, carrying the heading path down the
    tree. Spec §6 S2 puts headings ahead of paragraphs because a heading
    boundary is a real semantic boundary; a token-count cutoff is an
    arbitrary one.

    The heading line itself stays in the body it opens: heading_path is
    metadata, but the words in the heading also belong to content -- what
    the keyword index reads.

    `stack` holds the headings still open, and is updated in place: pass the
    same list for consecutive pages, and text opening a page continues the
    section the previous page ended in."""
    sections: list[tuple[str, str]] = []
    stack = [] if stack is None else stack
    path, buffer = " > ".join(stack), []

    for line in markdown.splitlines():
        found = _HEADING.match(line)
        if not found:
            buffer.append(line)
            continue
        if buffer:
            sections.append((path, "\n".join(buffer).strip()))
        level = len(found.group(1))
        del stack[level - 1 :]  # a shallower heading truncates the path
        stack.append(found.group(2).strip())
        path = " > ".join(stack)
        buffer = [line]

    if buffer:
        sections.append((path, "\n".join(buffer).strip()))

    # Text preceding the first heading keeps an empty path instead of being
    # dropped -- PyMuPDF output with no headings at all is the common case.
    return [(path, body) for path, body in sections if body]


def detect_language(text: str) -> str:
    """Called once per parent; children inherit it. A 150-token child is
    too little to classify reliably, and Vietnamese prose mixed with
    English terminology is exactly what throws it off."""
    code, _ = langid.classify(text)
    return code if code in LANGUAGES else "other"


def chunk_document(parsed: dict) -> dict:
    """chunk_index is assigned in document order, deterministically -- the
    natural key that keeps S5 idempotent."""
    parents, children = [], []
    parent_index = child_index = 0
    # One stack for the whole document: an article runs across page breaks.
    open_headings: list[str] = []

    for page in parsed["pages"]:
        text = sanitize(page["markdown"])
        if not text:
            continue

        for heading_path, section in split_sections(text, open_headings):
            for parent_text in _split_by_tokens(section, PARENT_MIN, PARENT_MAX):
                parent_tokens = count_tokens(parent_text)
                if parent_tokens < DROP_BELOW:
                    continue
                language = detect_language(parent_text)
                parents.append(
                    {
                        "chunk_index": parent_index,
                        "content": parent_text,
                        "token_count": parent_tokens,
                        "page_start": page["page"],
                        "page_end": page["page"],
                        "heading_path": heading_path or None,
                        "language": language,
                    }
                )

                for child_text in _split_by_tokens(parent_text, CHILD_MIN, CHILD_MAX):
                    child_tokens = count_tokens(child_text)
                    if child_tokens < DROP_BELOW:
                        continue
                    children.append(
                        {
                            "chunk_index": child_index,
                            "parent_index": parent_index,
                            "content": child_text,
                            "token_count": child_tokens,
                            "page_number": page["page"],
                            "language": language,
                        }
                    )
                    child_index += 1
                parent_index += 1

    return {"parents": parents, "children": children}
