# Phase 2, R2: hybrid search (keywords + vectors, fused with RRF)

> **For whoever executes this:** do the tasks in order, each one by TDD (red test → code → green test). Do not commit on your own: each task takes one snapshot, and commits happen when the repo owner says so. Any step that calls the real OpenAI needs agreement first.

**Goal:** add a keyword-search source that runs beside vector search, fuse the two lists with RRF, then measure `recall@10` again on the very same question set as R1.

**Architecture:** R0 left two seams ready: the `retrievers` list and the `fuse` function in `app/core/deps.py`. R2 adds one retriever and replaces `fuse`; `pipeline.py`, `contracts.py`, the database schema and the write path do not change. The flag `RETRIEVAL_BM25_ENABLED` (already there, default `false`) turns both on together; with the flag off the system runs exactly as R1.

**Tech Stack:** Postgres full-text search (`child_chunks.tsv`, the GIN index `ix_child_chunks_tsv`, there since Phase 1), SQLAlchemy async, pytest.

**Spec:** `docs/retrieval-design.md` §5.2 (keyword search), §5.3 (RRF), §8 (measurement).

## Global Constraints

- No new column, no re-ingest. (It originally also said "no migration"; see the part on how codes are split, at the end.)
- With the flag off the results must be identical to R1 measured on the same corpus: that is what makes a before/after comparison mean anything.
- The question set `eval/dataset.yaml` does not change (spec §8.4).
- No change to `web/`. The `POST /query` contract does not change, so the FE has nothing to do.
- The two retrievers share one `AsyncSession`: they run one after the other, no `gather`.

## Measured before the plan was written (SQL on the real corpus, 2,982 chunks, no OpenAI call)

Keyword search alone, scored with the eval set's labels (the right page is in the first 10 results):

| Search method | Hits in the top 10 | MRR | Questions returning nothing |
| --- | --- | --- | --- |
| The SQL in the spec: `websearch_to_tsquery` (AND) + `ts_rank_cd` | 0/20 | 0.000 | 19 |
| Words joined with OR + `ts_rank_cd` | 9/20 | 0.119 | 0 |
| Words joined with OR + `ts_rank` | 12/20 | 0.334 | 0 |
| **Words joined with OR + `ts_rank(…, 1)`** (divided by the log of the length) | **15/20** | **0.397** | 0 |
| As above, with the document title and the article heading added to the searched text | 13/20 | 0.301 | 0 |
| As above, dropping words that occur in more than 20% of the chunks | 12/20 | 0.302 | 0 |

Three things follow, and all three differ from the spec:

1. **AND does not work for natural-language questions.** The user asks "thì bị xử phạt thế nào", and no chunk contains every one of those words. The words have to be joined with OR.
2. **`ts_rank` does better than `ts_rank_cd`** on this corpus, contrary to the argument in spec §5.2.
3. **Adding the article heading to the keyword index does not help** (13 against 15, within the noise of 20 questions). So R2 does not change the schema.

Keyword search finds a08, a13 and a20, three of the five questions that vector search currently misses (a08, a13, a14, a16, a20). What it is worth after fusion can only be known by running the eval (Task 5).

One query takes 7–13 ms. It matches about 70% of the chunks because the words are joined with OR, so the cost grows linearly with the corpus; the threshold for looking at it again is `retrieve` going over 200 ms in the trace.

The eval set has only 20 questions, so the table above is enough to rule out a broken option (AND), and not enough to tune parameters. The plan does not search for any further parameter on this set.

## Out of scope

- **Bilingual sub-queries.** The ladder table puts them at R2, but spec §6.1 puts them in the rewrite node, which is an LLM call and belongs to R4. The corpus is currently Vietnamese plus 6 Japanese chunks, and the eval set has no cross-language question to measure with. Moved to R4.
- **Real BM25** (`pg_search`): needs a separate Postgres image. To be considered only if questions containing a code are still missed after R2.
- **Japanese, Chinese:** the `simple` configuration splits words on whitespace, so keyword search finds almost nothing there. Those chunks can still be found by vector.
- Reranking (R3), rewrite (R4).

## Files

| File | Work |
| --- | --- |
| `app/repositories/chunk_repository.py` | add `KeywordHit`, `keyword_search()` |
| `app/core/retrievers/bm25.py` | new: `BM25Retriever` |
| `app/core/stages/fusion.py` | new: `reciprocal_rank_fusion()` (a pure function) and the `fuse` stage |
| `app/core/deps.py` | wire in the retriever and `fuse` by the flag; run the retrievers one after the other |
| `eval/run.py` | print the flags that are on along with the results |
| `tests/helpers.py` | add `seed_texts()` |
| `tests/test_chunk_repository.py`, `tests/test_bm25_retriever.py`, `tests/test_fusion.py`, `tests/test_hybrid_search.py` | tests |
| `docs/retrieval-design.md`, `docs/system-design.md`, `README.md` | Task 6 |

---

### Task 1: keyword search in the repository

**Interface produced:**

```python
@dataclass(frozen=True)
class KeywordHit:
    child_id: uuid.UUID
    parent_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    child_content: str
    score: float
    filename: str

class ChunkRepository:
    async def keyword_search(self, query: str, over_fetch: int, filters: Filters) -> list[KeywordHit]: ...
```

- [x] **Step 1: add `seed_texts` to `tests/helpers.py`**: one document, one parent, one child per paragraph, each on a page of its own; it takes `language`.

- [x] **Step 2: write the failing tests** in `tests/test_chunk_repository.py`:

| Test | Proves |
| --- | --- |
| `test_keyword_search_finds_the_chunk_that_quotes_a_code` | asking `04/SS-HĐĐT`, the chunk that contains exactly that code comes first |
| `test_a_question_matches_without_every_word_being_present` | a natural-language question still returns results even though the chunk contains only one of its words (OR, not AND) |
| `test_a_question_with_no_searchable_word_finds_nothing` | `"?!..."` returns `[]`, no error |
| `test_english_chunks_are_matched_through_english_stemming` | a `language="en"` chunk contains "vehicles", and asking "vehicle examination" still matches |
| `test_keyword_search_honours_the_document_filter` | filtering on `document_id` returns only that document's chunks |
| `test_keyword_hits_come_best_first_and_stop_at_over_fetch` | scores descend, and the number of results does not exceed `over_fetch` |

- [x] **Step 3: run, see red** (`AttributeError: keyword_search`).

- [x] **Step 4: implement.** The SQL:

```sql
WITH q AS (
  SELECT (SELECT string_agg(quote_literal(lex), ' | ')::tsquery
          FROM   unnest(tsvector_to_array(to_tsvector('simple', :q))) AS lex) AS plain,
         (SELECT string_agg(quote_literal(lex), ' | ')::tsquery
          FROM   unnest(tsvector_to_array(to_tsvector('english', :q))) AS lex) AS english
), candidates AS (
  SELECT c.id, c.parent_id, c.document_id, c.page_number, c.content,
         ts_rank(c.tsv, CASE WHEN c.language = 'en' THEN q.english ELSE q.plain END, 1) AS score
  FROM   child_chunks c, q
  WHERE  ((c.language = 'en' AND c.tsv @@ q.english)
       OR (c.language IS DISTINCT FROM 'en' AND c.tsv @@ q.plain))
    AND  (CAST(:category AS text) IS NULL OR c.category = CAST(:category AS text))
    AND  (CAST(:document_id AS uuid) IS NULL OR c.document_id = CAST(:document_id AS uuid))
  ORDER  BY score DESC, c.id
  LIMIT  :over_fetch
)
SELECT c.id AS child_id, c.parent_id, c.document_id, c.page_number,
       c.content AS child_content, c.score, d.filename
FROM   candidates c
JOIN   documents d ON d.id = c.document_id
ORDER  BY c.score DESC, c.id
```

Why it is written this way:

- The question is tokenised by `to_tsvector` itself, with the same configuration as the `tsv` column, so words in the question and words in the index are normalised the same way. The words are joined with `|`.
- The `tsv` column uses the `english` configuration for English chunks and `simple` for the rest, so the question is built into two tsqueries and each row is compared with the one in its own configuration. The two `OR` branches are written separately so that Postgres can use the GIN index for both.
- When the question has no word, `string_agg` returns `NULL`, the `@@` condition becomes `NULL`, and the result is empty.
- `c.id` is in the `ORDER BY` so that two chunks with equal scores always come out in the same order.

- [x] **Step 5: run, see green.** `uv run pytest tests/test_chunk_repository.py -q`
- [x] **Step 6: snapshot `r2-task-1`.**

### Task 2: `BM25Retriever`

**Uses:** `ChunkRepository.keyword_search`. **Produces:** `BM25Retriever(session)` with `source = "bm25"` and `async def search(plan: Plan) -> list[Candidate]`.

The name stays `bm25` because `Source`, the config flag and the spec have called it that since R0; the docstring says plainly that this is Postgres's `ts_rank`, with no IDF.

- [x] **Step 1: failing tests** in `tests/test_bm25_retriever.py`:

| Test | Proves |
| --- | --- |
| `test_candidates_are_ranked_from_one_within_the_source` | `rank` runs 1, 2, 3… with no gap; `source == "bm25"` |
| `test_sub_queries_are_merged_without_duplicating_a_child` | when two sub-queries match the same chunk, that chunk appears once and keeps the higher score |
| `test_candidate_carries_a_doc_ref_with_the_child_page` | `ref` is a `DocRef`, `page_number` is the child's page |

- [x] **Step 2: red** (`ModuleNotFoundError`). **Step 3: implement**, on the same pattern as `VectorRetriever`: group by `child_id`, keep the highest score, sort by `(-score, child_id)`, number `rank` from 1. **Step 4: green.** **Step 5: snapshot `r2-task-2`.**

### Task 3: RRF

**Produces** in `app/core/stages/fusion.py`:

```python
def reciprocal_rank_fusion(
    candidates: list[Candidate], k: int, top_k: int
) -> tuple[list[Candidate], dict[str, int]]:
    """The fused list (already cut to top_k) and a count of where the kept
    candidates came from, for example {"bm25+vector": 4, "vector": 5, "bm25": 1}."""

async def fuse(query: Query, candidates: list[Candidate], trace: Trace) -> list[Candidate]:
    """Stage: reads k and top_k from settings, records a `fuse_detail` node in the trace."""
```

Rules:

- A candidate's key is its `child_id` (for a `WebRef`, the `url`).
- Score = the sum of `1 / (k + rank)` over the sources that have it. Each source already guarantees that a chunk appears only once (Task 2 and the existing test of `VectorRetriever`).
- Sorted by descending score; on equal scores the candidate with the better original rank comes first, then the key. The order does not depend on the input order.
- The candidate returned is the copy from the source that ranked it highest, with a new `rank` from 1 and `score` set to the RRF score.

- [x] **Step 1: failing tests** in `tests/test_fusion.py`:

| Test | Proves |
| --- | --- |
| `test_agreement_between_sources_beats_leading_one_list` | the table in spec §5.3: B (vector rank 5, bm25 rank 2) comes above A and C (each at the top of one list) |
| `test_fused_ranks_run_from_one_and_scores_are_the_rrf_sums` | `rank` 1…n; B's score equals `1/65 + 1/62` |
| `test_the_list_is_cut_to_top_k` | 15 candidates, `top_k=10` → 10 |
| `test_one_source_alone_keeps_its_order` | with vector alone the order does not change |
| `test_the_order_does_not_depend_on_the_input_order` | shuffle the input, the result is identical |
| `test_the_summary_counts_where_the_kept_candidates_came_from` | the counts are right |

- [x] **Step 2: red. Step 3: implement. Step 4: green. Step 5: snapshot `r2-task-3`.**

### Task 4: wire into `build_stages`

- [x] **Step 1: failing tests** in `tests/test_hybrid_search.py`, through `POST /query` with `query_client`:

| Test | Proves |
| --- | --- |
| `test_with_the_flag_off_only_vectors_are_searched` | `retrieve_detail.per_source` has only `vector`; there is no `fuse_detail` node |
| `test_with_the_flag_on_both_sources_are_searched` | `per_source` has both `vector` and `bm25`; `fuse_detail` is there |
| `test_a_chunk_both_sources_agree_on_leads_the_context` | the stub's vectors carry no meaning; the only chunk that matches the keyword `04/SS-HĐĐT` is voted for by both sources and so leads `expand_detail` |

- [x] **Step 2: red.**
- [x] **Step 3: change `app/core/deps.py`:**
  - `if settings.retrieval_bm25_enabled: retrievers.append(BM25Retriever(session))`
  - replace `asyncio.gather` with a sequential loop, and correct the comment so it gives the right reason (one shared session);
  - `fuse=fusion.fuse if settings.retrieval_bm25_enabled else noops.fuse`.
- [x] **Step 4: green; run the whole suite** (`uv run pytest --ignore=tests/test_ocr_service.py -q`), `ruff check`, `ruff format --check`.
- [x] **Step 5: snapshot `r2-task-4`.**

### Task 5: measure

- [x] **Step 1:** `eval/run.py` prints one more line at the top of the results, for example `stages           bm25=on`, so that a recorded figure always comes with the rung of the ladder that was measured. Test in `tests/test_eval_harness.py`.
- [x] **Step 2 (no OpenAI call):** `uv run python -m eval.run --check-labels` → `labels 20 reachable`.
- [x] **Step 3 (calls OpenAI: 20 embeddings, needs the repo owner's agreement):** `RETRIEVAL_BM25_ENABLED=true uv run python -m eval.run`. The baseline to compare against is R1 measured again just before, on the same corpus (flag off).
- [x] **Step 4:** record the results per kind of question, and the list of missed questions. Every result is recorded, including one that shows no improvement. If `recall@10` does not go up, the flag keeps its default of `false` and the spec records why.

### Task 6: documentation

- [x] `docs/retrieval-design.md`: §5.2 rewritten (the results row in §8.7 waits for Task 5) from the real SQL and the measurement table above (AND → OR, `ts_rank_cd` → `ts_rank`), §3 records that bilingual sub-queries move to R4, §8.7 gains the R2 results row.
- [x] `docs/system-design.md` (the Phase 2 section) and `README.md`: R2's status, how to turn the flag on.
- [x] The English version of this plan: `docs/plans/phase-2-r2.en.md`.
- [x] Snapshot `r2-task-6`.

## Definition of Done

1. The whole suite is green, `ruff check` and `ruff format --check` are clean.
2. Flag off: no node in the trace differs from R1.
3. Flag on: the trace has `per_source` with two sources, and `fuse_detail`.
4. R2's `recall@10` and MRR are written into spec §8.7, per kind of question, beside R1's figures.
5. Spec §5.2 matches the SQL that is running.

## Outcome (2026-10-11)

Measured on the freshly re-ingested corpus (2,986 chunks), two runs back to back:

| | `recall@10` | `MRR` | Missed |
| --- | --- | --- | --- |
| R1, flag off | 0.700 | 0.463 | a06, a08, a10, a13, a14, a16 |
| R2, flag on | 0.750 | 0.542 | a08, a13, a14, a16, a18 |

By kind: questions containing a code go from 0.750 / 0.425 to 1.000 / 0.750; ordinary questions from 0.750 to 0.875; follow-ups stay at 0.250; summary questions go from 1.000 down to 0.750 (a18 lost).

Overall it is ahead by exactly one question, which is within the noise of 20 questions. Where R2 clearly works is the questions containing a code, which is what it is for. Keyword search ranks a08 and a13 at 5 and 6, but they still fall out of the top 10 after fusion: that part belongs to reranking (R3).

The flag still defaults to `false`; turning it on or not is the repo owner's decision.

Remaining: the R2 code has not had an independent review.

**How codes are split (outside the original plan, approved by the repo owner 2026-10-11).** Postgres's tokeniser reads `168/2024/NĐ-CP` as a file path and cuts at the first non-ASCII letter: the index holds `168/2024/n`, `đ-cp`, `đ`, `cp`, so asking "nghị định 168" does not match. Migration `a0454bfe1fe8` rebuilds the `tsv` column on `translate(content, '/-', '  ')`, and the query applies the same replacement to the question. The "no migration" constraint at the top of the plan no longer holds for this part. The test `test_a_code_is_found_from_any_part_of_it` went red → green; the whole suite of 398 is green. The dev database has had the migration run, with no re-ingest.

The price is noise: `2024` and `cp` occur in a great many chunks; of the first 10 results for "168/2024" only 4 quote that decree. Keyword search on its own over the eval set is still 15/20, before and after. The R2 figures after fusion were measured again after this change: still 0.750 / 0.542, with the same list of misses.

The flag `RETRIEVAL_BM25_ENABLED=true` is turned on in the dev machine's `.env` (at the repo owner's request); the default in the code and in `.env.example` is still `false`.

Branch: from R2 onward the work is on `phase-3`, branched from `phase-2`.
