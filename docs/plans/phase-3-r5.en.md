# Phase 3, R5: the router and a tool that reads a document's outline

> **For whoever executes this:** do the tasks in order, each one by TDD (red test → code → green test). Do not commit on your own; when told to, group the work into a few large commits. Any step that calls the real OpenAI needs agreement first.

**Goal:** a question such as "Tóm tắt Luật Đường bộ" is answered from the outline of that very document (its chapters, its articles), not from the 10 scattered passages that vector search happened to pick up. Spec §8.4 records exactly this point: "vector search structurally cannot do this".

**Architecture:** R0 already left a `plan` stage (no-op), `Plan.strategy` / `Plan.document_hint`, and `Source = "browse"` in place. R5 replaces `plan` with a router, adds a retriever, `BrowseRetriever`, that reads `parent_chunks.heading_path`, and has the later stages keep "browse" candidates as they are. `pipeline.py` does not change. The flag `RETRIEVAL_ROUTER_ENABLED` (already there, default `false`) turns all of it on.

**Spec:** `docs/retrieval-design.md` §6.2 (router), §3 (R5: "Strategy distribution + browse accuracy"), §8.7 (summary questions that could not yet be measured).

## A decision that departs from the spec, needing the repo owner's approval

Spec §6.2 gives the router **a model call of its own**. This plan **adds no call**: the router is code, reading the result of R4's rewrite step, which already classifies the question (`SUMMARIZE_DOC` for 3/4 of the eval set's summary questions). The rewrite step only needs one more field: which document the question mentions (`document_hint`).

- Reason: each model call adds about 1.3 s to every question (measured at R4). Two calls just to decide two adjacent things is paying twice for the same reading of the question.
- The trade-off: the router cannot decide anything the rewrite step does not see. With the current set of strategies (`HYBRID`, `BROWSE_DOC`, `NO_RETRIEVAL`), every signal lies in the intent and the document name.
- `VECTOR_ONLY` is not used: no signal shows which questions should skip keyword search.

## Global Constraints

- With the flag off: the results are identical to R4.
- The router never breaks a request: if the document cannot be identified, it falls back to `HYBRID`, and the trace records the reason.
- The trace records the `strategy` of every question, and the `query_traces.strategy` column (there since R1) is filled in. Spec §6.2: a router that always returns `HYBRID` is a bug that only this distribution shows.
- No model call beyond the rewrite step that already exists.
- The eval question set does not change; only ways of scoring are added. No change to `web/`; the `POST /query` contract does not change.

## Out of scope

- **`METADATA`** ("thư viện có những tài liệu nào?"): the FE already shows the list of documents, and the eval set has no question of this kind. Noted for later.
- **Summarising several documents at once** and comparing two documents.
- **Scoring the quality of a summary** (needs a model as judge, belongs to R7).
- **R5.5** (a ReAct agent for the reading tool, spec §3): tried only once R5 has figures.

## Files

| File | Work |
| --- | --- |
| `app/core/contracts.py` | `Query.document_hint` |
| `app/core/stages/rewrite.py` | `RewriteResult.document_hint`, the prompt |
| `app/core/stages/route.py` | new: the router, in code |
| `app/repositories/document_lookup.py` | new: from the name in the question to one document (the `ix_documents_title_fts` index has been there since Phase 1) |
| `app/core/retrievers/browse.py` | new: a document's outline as candidates |
| `app/core/deps.py`, `app/core/stages/fusion.py`, `app/core/parent_expander.py` | `BROWSE_DOC` runs only the reading tool; "browse" candidates pass through fusion, rerank and expansion unchanged |
| `app/services/query_service.py` | record `strategy` in the trace |
| `eval/run.py` | outline coverage and document-selection accuracy |
| the matching tests; documentation in Task 7 | |

---

### Task 1: the rewrite step says which document the question mentions

- `RewriteResult.document_hint: str`: the name, number or subject of **one** document the question mentions, written exactly as the user wrote it; an empty string if there is none. (Structured outputs in strict mode require every field, so an empty string stands in for `null`.)
- `Query.document_hint` carries it to the router.
- [x] **Failing tests:** "Tóm tắt Luật Đường bộ" → `document_hint == "Luật Đường bộ"` goes into `Query`; a question that mentions no document → `None`; the `rewrite_detail` trace has `document_hint`.
- [x] **Implement, tests green.**

### Task 2: the router

| Intent | `document_hint` | Strategy |
| --- | --- | --- |
| `CHITCHAT` | — | `NO_RETRIEVAL` (as in R4) |
| `SUMMARIZE_DOC` | present, or the client sent `document_id` | `BROWSE_DOC` |
| every other case | — | `HYBRID` |

- Trace: a `route_detail` node with `strategy`, `document_hint`, and a one-sentence reason.
- [x] **Failing tests:** each row of the table; a summary request with no known document → `HYBRID` with the reason "no document named".
- [x] **Implement, tests green.**

### Task 3: from the name in the question to one document

- If the client sent `document_id` (the user has a document open), use it.
- Otherwise, search for `document_hint` in the titles and file names of the documents that have finished ingesting, using the `ix_documents_title_fts` index, joining words with OR as in R2. Accept only when the top document is clearly ahead of the second; two documents with close scores (for example "Nghị định 151" and "Nghị định 236 sửa đổi Nghị định 151") count as not identified.
- [x] **Failing tests:** "Luật Đường bộ" gives the right law, not "Luật Trật tự, an toàn giao thông đường bộ"; "Nghị định 168" gives NĐ 168, not NĐ 238 (which amends NĐ 168); a name that matches nothing → `None`; two candidates with close scores → `None`.
- [x] **Measure on the real database (free):** with the names of the ten documents and their common abbreviations, what threshold for "clearly ahead" makes no mistakes.
- [x] **Implement, tests green.**

### Task 4: the outline reading tool

For one document, return, in order:

1. An opening candidate: the document's title and its first parent (usually the legal bases and the article on the scope of regulation).
2. Each top-level section (a chapter, or an equivalent part) as one candidate: the section's name and the list of its subsections' names (articles), without their content. `ref` points to the section's first parent, so the citation opens the page where the chapter starts.

- The outline of the largest document (NĐ 168: 70 heading paths), with the repeated parts of the paths dropped, still fits within the context's 8,000-token budget.
- Candidates with `source="browse"` are not reordered by RRF, not cut by the reranker, and do not have their content replaced by parent expansion.
- A document with no headings at all (for example the 3-page Japanese file): only the opening candidate is returned.
- [x] **Failing tests** on seed data with two chapters: the right order, the right subsection names, `ref` on the right page; passing through `fuse` and `expand` unchanged; a document with no headings.
- [x] **Wire into `deps.py`:** `BROWSE_DOC` runs the reading tool instead of the retrievers; if no document is found, fall back to `HYBRID` and the trace records it.
- [x] **Implement, tests green; the whole suite green.**

### Task 5: an eval that can score summary questions

- **Outline coverage** for the `summarize` kind of question: of the top-level sections of the right document, how many are present in the context. This is the measure that spec §8.7 said R5 needs.
- **Right document chosen:** for questions that go through `BROWSE_DOC`, whether the chosen document is the one in the label.
- **Strategy distribution** on Set A and Set B.
- [x] **Failing tests, implement, tests green.**
- [x] **Measure the R4 baseline (free, from the cache):** the outline coverage of the 4 summary questions before there is a router.

### Task 6: measure

- [x] **(calls OpenAI, needs agreement)** The rewrite prompt changes, so the old cache cannot be used: 28 rewrite calls, about 1 cent.
- [x] Compare with R4: `recall@10` and MRR per kind of question (no kind may go down), outline coverage, right document chosen, strategy distribution.
- [x] **(calls OpenAI, needs agreement)** `--full` for the 4 summary questions, to read the answers first-hand: 4 answer calls.

### Task 7: documentation

- [x] `docs/retrieval-design.md`: §6.2 records the decision to build the router in code; §8.7 adds the R5 row and the new measures.
- [x] `README.md`, `docs/system-design.md`.
- [x] The English version of this plan.

## Definition of Done

1. The whole suite is green, `ruff` is clean.
2. Flag off: the results are as in R4.
3. Document not identified: the question is still answered by `HYBRID`, and the trace records the reason.
4. The outline coverage of summary questions, the document-selection accuracy and the strategy distribution are written into spec §8.7 beside R4's.

## Outcome (2026-10-17)

The repo owner approved the router in code ("tạm thời theo m gợi ý đi").

| | `recall@10` | `MRR` | Outline coverage (summary questions) | Right document chosen |
| --- | --- | --- | --- | --- |
| R4 | 0.900 | 0.580 | 0.440 | — |
| **R5** | **0.900** | **0.624** | **1.000** | **4/4** |

- All 4 summary questions went through `BROWSE_DOC` and found the right document from the name the question used. The MRR of the summary kind went from 0.656 to 0.875, because the outline opens with page 1 of the document, which is also the gold page. The other kinds of question did not change.
- **First run:** a19 ("Nghị định về vận tải hàng không nói về những nội dung gì?") was not recognised as a summary question; coverage 0.857. The definition of `SUMMARIZE_DOC` in the prompt was widened to "summarise or outline a document, or what a document covers or is about" (the prompt's own words): one general definition, not one written for this question alone.
- **Reading the 4 answers first-hand:** the right chapters, with a citation for each chapter; 2.2–4.8 s for the whole pipeline. The answers tend to end with a sentence saying that the sources only have the names of the chapters and articles. That comes from the answer step's rule to "say clearly when the sources are not enough", and has not been tuned yet.
- **Set B:** the router chose `HYBRID` for all 8 questions. Since the prompt gained the `document_hint` field, the rewrite step marks only 1/5 of the out-of-scope questions as `OUT_OF_SCOPE` (4/5 before). This intent is not used before R7; R7 must measure it again.
- Time of the rewrite step: a median of 1.4 s.
- Cost of the run: 56 rewrites, 2 embeddings, 4 answers.

Departures from the plan:

- No score threshold for telling documents apart was needed: a name is scored against the first words of the title (a document names itself at the start of its title), and words in the name that the title lacks are penalised heavily (the square of the match ratio). On the 10 real documents, every way of naming them that was tried gave the right document, and ambiguous names gave no document.
- Tests now turn off every stage flag by default, whatever the dev machine's `.env` turns on. Reason: with rerank turned on in `.env`, the R2 tests called into the real rerank container.
