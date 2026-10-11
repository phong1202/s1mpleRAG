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


_MARKUP = re.compile(r"^[#\s*_]+|[\s*_]+$")
_NUMBERED = re.compile(r"^((?:[^\W\d_]+\s+){1,2})(\d+[a-z]?|[IVXLCDM]+)(?=[\s.:)]|$)")
_TITLE_MAX = 300


def _marked(line: str) -> bool:
    text = line.strip()
    return bool(_HEADING.match(text)) or (
        text.startswith("**") and text.endswith("**") and len(text) > 4
    )


def _kind(line: str) -> tuple[str, str, str] | None:
    """(kind, number, bare title) of a numbered heading line."""
    bare = _MARKUP.sub("", line)
    found = _NUMBERED.match(bare)
    if not found:
        return None
    return found.group(1).strip().casefold(), found.group(2), bare[:_TITLE_MAX]


def learn(lines: list[str]) -> tuple[set[str], set[tuple[str, str]]]:
    """The kinds that recur among marked lines, and the pairs (a, b) meaning
    'a is not a subdivision of b'.

    A kind subdivides another only if somewhere two or more of it sit between
    one heading of the other and the next: four sections inside one chapter.
    A chapter never does that to an article -- at most one chapter falls
    between two consecutive articles -- so a chapter is never filed under one,
    whatever '#' level the parser gave either."""
    seen = [k[0] for line in lines if _marked(line) and (k := _kind(line))]
    kinds = {k for k in seen if seen.count(k) >= 2}
    seen = [k for k in seen if k in kinds]
    outside: set[tuple[str, str]] = set()
    for a in kinds:
        for b in kinds - {a}:
            most = run = 0
            started = False
            for k in seen:
                if k == b:
                    started, run = True, 0
                elif k == a and started:
                    run += 1
                    most = max(most, run)
            if most < 2:
                outside.add((a, b))
    return kinds, outside


def split_document(pages: list[str]) -> list[list[tuple[str, str]]]:
    """Per page, the (heading_path, body) sections."""
    lines = [line for page in pages for line in page.splitlines()]
    kinds, outside = learn(lines)
    if not kinds:
        # No numbered headings to go by: the markdown levels are all there is.
        open_headings: list[str] = []
        return [split_sections(page, open_headings) for page in pages]
    stack: list[tuple[str, str]] = []  # (kind, title)
    last_number: dict[str, str] = {}
    out = []
    untitled = None
    for page in pages:
        sections, buffer = [], []
        path = " > ".join(t for _, t in stack)
        for line in page.splitlines():
            k = _kind(line)
            marked = _marked(line)
            opens = bool(k) and k[0] in kinds and (marked or _follows(last_number.get(k[0]), k[1]))
            if opens:
                if buffer:
                    sections.append((path, "\n".join(buffer).strip()))
                kind, number, title = k
                if any(kd == kind for kd, _ in stack):
                    del stack[[kd for kd, _ in stack].index(kind) :]
                while stack and (kind, stack[-1][0]) in outside:
                    stack.pop()
                stack.append((kind, title))
                last_number[kind] = number
                untitled = (
                    len(stack) - 1
                    if not title[_NUMBERED.match(title).end() :].strip(" .:")
                    else None
                )
                path, buffer = " > ".join(t for _, t in stack), [line]
                continue
            if untitled is not None and line.strip():
                if marked or line.strip().isupper():
                    kd, t = stack[untitled]
                    stack[untitled] = (kd, f"{t}. {_MARKUP.sub('', line)}"[:_TITLE_MAX])
                    path = " > ".join(t for _, t in stack)
                untitled = None
            buffer.append(line)
        if buffer:
            sections.append((path, "\n".join(buffer).strip()))
        out.append([(p, b) for p, b in sections if b])
    return out


def _follows(previous: str | None, number: str) -> bool:
    """A plain line continues a numbered series only as its very next item."""
    return (
        bool(previous)
        and previous.isdigit()
        and number.isdigit()
        and int(number) == int(previous) + 1
    )


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
    # The whole document at once: which headings are structure, and how they
    # nest, is learned from all of them, and an article runs across page breaks.
    pages = [(page["page"], sanitize(page["markdown"])) for page in parsed["pages"]]
    outline = split_document([text for _, text in pages])

    for (page_number, _), sections in zip(pages, outline, strict=True):
        for heading_path, section in sections:
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
                        "page_start": page_number,
                        "page_end": page_number,
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
                            "page_number": page_number,
                            "language": language,
                        }
                    )
                    child_index += 1
                parent_index += 1

    return {"parents": parents, "children": children}
