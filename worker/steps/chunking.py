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
_SERIAL = re.compile(r"(\d+)([a-z]?)")
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_TITLE_MAX = 300


def _clean(text: str) -> str:
    """A heading without its markup: "### **Chương III** **THUÊ**" is "Chương III THUÊ"."""
    return " ".join(_MARKUP.sub("", text).replace("*", " ").split())


def _marked(line: str) -> str | None:
    """ "#" for a markdown heading, "b" for a line that is bold from end to end
    or opens with a bold numbered marker ("**Mẫu số 04a.** Văn bản..."), else None."""
    text = line.strip()
    if _HEADING.match(text):
        return "#"
    if text.startswith("**") and len(text) > 4:
        if text.endswith("**"):
            return "b"
        opening = text[2:].split("**", 1)[0]
        if len(opening) <= 120 and _NUMBERED.match(opening.strip()):
            return "b"
    return None


def _kind(line: str) -> tuple[str, str, str, str] | None:
    """(kind, number, the character after the number, title) of a line that
    reads "words + number". The kind is the words."""
    text = line.strip()
    if text.startswith("**") and not text.endswith("**") and not _HEADING.match(text):
        text = text[2:].split("**", 1)[0]  # only the bold opening is the heading
    bare = _clean(text)
    found = _NUMBERED.match(bare)
    if not found:
        return None
    return (
        found[1].strip().casefold(),
        found[2],
        bare[found.end() : found.end() + 1],
        bare[:_TITLE_MAX],
    )


def _serial(number: str) -> tuple[int, str] | None:
    """ "10a" is (10, "a"); "IV" is (4, "")."""
    arabic = _SERIAL.fullmatch(number)
    if arabic:
        return int(arabic[1]), arabic[2]
    if not number or any(ch not in _ROMAN for ch in number):
        return None
    total = 0
    for ch, after in zip(number, number[1:] + " ", strict=True):
        value = _ROMAN[ch]
        total += -value if _ROMAN.get(after, 0) > value else value
    return total, ""


def _follows(previous: str | None, number: str) -> bool:
    """Whether `number` is the very next item of a series: 10 then 11, 10 then
    10a, 98g then 98h, 10a then 11 -- or the first item of a series not opened
    yet. A plain line is taken for a heading on no weaker evidence."""
    now = _serial(number)
    if now is None:
        return False
    before = _serial(previous) if previous else None
    if before is None:
        return now == (1, "")
    (n1, s1), (n2, s2) = before, now
    if n2 == n1 + 1 and not s2:
        return True
    return n2 == n1 and bool(s2) and (s2 == "a" if not s1 else ord(s2) == ord(s1) + 1)


def learn(lines: list[str]) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    """What the document's own headings say about its structure:

    kinds    the kinds that are structure. A kind has to recur, and to appear
             at least once as a markdown heading -- unless the document has no
             markdown headings at all, when bold lines are all there is. Two
             bold "Table N" captions in a report with its own headings are not
             chapters.
    dotted   the kinds whose number is followed by a full stop ("Điều 6.").
             A plain line of such a kind needs the full stop too, which is what
             separates a heading from "Điều 24 của Luật này".
    outside  the pairs (a, b) meaning 'a is never filed under b'.

    Two things put a under no b. One: b is itself a subdivision of a -- two or
    more b sit between one a and the next, as articles do inside a chapter, so
    a chapter is never filed under an article. Two: a only ever appears after
    the last b, and nothing that lives inside b appears again once a has
    begun -- appendices after the final article. Sections that exist only in
    the final chapter differ exactly there: articles go on under them."""
    marked = [(mark, kind) for line in lines if (mark := _marked(line)) and (kind := _kind(line))]
    names = [kind[0] for _, kind in marked]
    as_heading = {kind[0] for mark, kind in marked if mark == "#"}
    no_markdown = not any(_HEADING.match(line.strip()) for line in lines)
    kinds = {n for n in names if names.count(n) >= 2 and (n in as_heading or no_markdown)}
    dots = [kind[0] for _, kind in marked if kind[2] == "."]
    dotted = {n for n in kinds if 2 * dots.count(n) > names.count(n)}
    seen = [n for n in names if n in kinds]

    def subdivides(inner: str, outer: str) -> bool:
        # the longest run of `inner` strictly between two consecutive `outer`
        best = run = 0
        started = False
        for name in seen:
            if name == outer:
                if started:
                    best = max(best, run)
                started, run = True, 0
            elif name == inner and started:
                run += 1
        return best >= 2

    outside: set[tuple[str, str]] = set()
    for a in kinds:
        for b in kinds - {a}:
            if subdivides(b, a):
                outside.add((a, b))
                continue
            last_b = len(seen) - 1 - seen[::-1].index(b)
            first_a = seen.index(a)
            if first_a > last_b:
                inside_b = {c for c in kinds - {a, b} if subdivides(c, b)}
                if not inside_b & set(seen[first_a:]):
                    outside.add((a, b))
    return kinds, dotted, outside


def split_document(pages: list[str]) -> list[list[tuple[str, str]]]:
    """Per page, the (heading_path, body) sections.

    Numbered headings of a recurring kind form the tree; an unnumbered markdown
    heading is a sub-level of whichever numbered heading is open."""
    # OCR sometimes glues consecutive bold headings into one line.
    pages = [page.replace("****", "**\n**") for page in pages]
    kinds, dotted, outside = learn([line for page in pages for line in page.splitlines()])
    if not kinds:
        # No numbered headings to go by: the markdown levels are all there is.
        open_headings: list[str] = []
        return [split_sections(page, open_headings) for page in pages]

    stack: list[tuple[str, str]] = []  # the open numbered headings: (kind, title)
    # Markdown headings open beneath the innermost numbered one, with their level.
    under: list[tuple[int, str]] = []
    last_number: dict[str, str] = {}
    untitled: int | None = None
    marker_level = 0
    out = []

    def path() -> str:
        return " > ".join([title for _, title in stack] + [title for _, title in under])

    for page in pages:
        sections: list[tuple[str, str]] = []
        buffer: list[str] = []
        current = path()
        for line in page.splitlines():
            mark = _marked(line)
            numbered = _kind(line)
            if numbered and numbered[0] in kinds:
                kind, number, after, title = numbered
                bare = not title[_NUMBERED.match(title).end() :].strip(" .:")
                # A plain line has to earn it: the next number of its series,
                # written the way that kind's headings are written.
                written_alike = after == "." if kind in dotted else bare
                if mark or (written_alike and _follows(last_number.get(kind), number)):
                    if buffer:
                        sections.append((current, "\n".join(buffer).strip()))
                    open_kinds = [open_kind for open_kind, _ in stack]
                    if kind in open_kinds:
                        del stack[open_kinds.index(kind) :]
                    while stack and (kind, stack[-1][0]) in outside:
                        stack.pop()
                    stack.append((kind, title))
                    under.clear()
                    last_number[kind] = number
                    untitled = len(stack) - 1 if bare else None
                    own = _HEADING.match(line.strip())
                    marker_level = len(own.group(1)) if own else 0
                    current, buffer = path(), [line]
                    continue
            if untitled is not None and line.strip():
                # "Chương II" on one line, its name on the next. A markdown
                # heading deeper than the marker's own is a sub-heading, not
                # a name -- unless it is in capitals, as such names are.
                as_markdown = _HEADING.match(line.strip())
                deeper = bool(as_markdown) and len(as_markdown.group(1)) > marker_level > 0
                named = line.strip("#* _").isupper() or (mark and not deeper)
                if named and untitled < len(stack):
                    kind, title = stack[untitled]
                    stack[untitled] = (kind, f"{title}. {_clean(line)}"[:_TITLE_MAX])
                    current = path()
                    untitled = None
                    buffer.append(line)
                    continue
                untitled = None
            found = _HEADING.match(line.strip())
            if found:
                if buffer:
                    sections.append((current, "\n".join(buffer).strip()))
                level = len(found.group(1))
                while under and under[-1][0] >= level:
                    under.pop()
                under.append((level, _clean(found.group(2))))
                current, buffer = path(), [line]
                continue
            buffer.append(line)
        if buffer:
            sections.append((current, "\n".join(buffer).strip()))
        out.append([(heading, body) for heading, body in sections if body])
    return out


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
                            "heading_path": heading_path or None,
                            "language": language,
                        }
                    )
                    child_index += 1
                parent_index += 1

    return {"parents": parents, "children": children}
