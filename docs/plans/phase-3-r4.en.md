# Phase 3, R4: understanding follow-up questions (rewrite + intent)

> **For whoever executes this:** do the tasks in order, each one by TDD (red test → code → green test). Do not commit on your own; when told to, group the work into a few large commits. Any step that calls the real OpenAI needs agreement first.

**Goal:** a follow-up question ("Thế nếu người ngồi sau không đội thì sao?") is rewritten into a question that stands on its own before searching, using the conversation history the server stores. Measured with the `recall@10` of the `followup` kind of question, currently 0.500 at R3.

**Architecture:** R0 already left a `rewrite` stage (no-op) and `Query.history` in place. R4 replaces `rewrite` with one call to `gpt-6-luna` that returns a `RewriteResult` (spec §6.1), and adds two tables, `conversations` / `messages`, so that the server keeps the history itself (the spec's decision no. 9). `pipeline.py` does not change. The flag `RETRIEVAL_INTENT_ENABLED` (already there, default `false`) turns all of it on.

**Spec:** `docs/retrieval-design.md` §6.1 (rewrite + intent), decision no. 9 (conversations stored on the server side), §8.4 (follow-up questions).

## Global Constraints

- With the flag off, the results are identical to R3; the `POST /query` contract only **adds** fields and does not change the old ones, so the current FE keeps working.
- If the rewrite fails (the model returns garbage, or times out), use the original question: the pipeline already has a fallback to the no-op for this stage.
- `standalone` may only use words from the user and from the history; it adds no new concept (spec §6.1: over-broadening, hallucinated context).
- **One** search query by default; split into several (at most 3) only when the question really asks about several things.
- The eval question set does not change. No change to `web/`: the FE part goes through the handoff document.

## Out of scope

- **Acting on intents other than `CHITCHAT`.** `SUMMARIZE_DOC` and `METADATA` belong to R5 (router), `OUT_OF_SCOPE` to R7 (refusal). R4 only records these intents in the trace, so that R5 and R7 can measure on real data.
- **Bilingual sub-queries.** Spec §6.1 asks for cross-language recall to be measured first, but the eval set has no cross-language question. Not done yet.
- **An API to read the conversation history back** (`GET /conversations/{id}`): the FE already keeps its own list of messages.
- **Filtering by documents the model guesses** (`RewriteResult.filters` in the spec): filters are still sent by the interface.

## Files

| File | Work |
| --- | --- |
| `app/core/contracts.py` | `Query.intent` |
| `app/core/stages/rewrite.py` | new: `RewriteResult`, the prompt, `make_rewrite(provider)` |
| `app/core/stages/noops.py` | `plan`: `CHITCHAT` → `NO_RETRIEVAL`, no search |
| `app/core/generator.py` | the answer for `CHITCHAT`: short, no citations, no facts stated |
| `app/core/deps.py` | wire `rewrite` by the flag |
| `app/models/conversation.py`, `app/models/message.py`, `alembic/versions/…` | two new tables |
| `app/repositories/conversation_repository.py` | create a conversation, read the last N turns, write the question turn and the answer turn |
| `app/schemas/query.py`, `app/services/query_service.py`, `app/controllers/query_controller.py` | `conversation_id` in and out; the trace records `standalone` |
| `eval/run.py` | a cache for the rewrite calls; print the intent distribution |
| `docs/fe-query-handoff.md` | the handoff part for the FE session |
| the matching tests | |

---

### Task 1: the `rewrite` stage

**Produces:**

```python
Intent = Literal["CHITCHAT", "SEARCH_CORPUS", "SUMMARIZE_DOC", "METADATA", "OUT_OF_SCOPE"]

class RewriteResult(BaseModel):
    intent: Intent
    standalone: str
    sub_queries: list[str] = Field(default_factory=list, max_length=3)

def make_rewrite(provider: AsyncLLMProvider) -> Callable[[Query, Trace], Awaitable[Query]]: ...
```

- The message sent to the model holds at most the last 6 turns of the history and the current question.
- Returns a `Query` with `text = standalone`, `sub_queries` only when there are 2 or more queries, and `intent`; `filters`, `history` and `conversation_id` stay as they are.
- An empty `standalone` means the original question is used.
- Trace: a `rewrite_detail` node with `intent`, `standalone`, `sub_queries`.
- `Query.refined()` keeps `intent`.

- [x] **Failing tests** (`AsyncStubProvider.script(RewriteResult, …)`): a follow-up question becomes a question that stands on its own; the history is in the message sent to the model, at most 6 turns; with one search query, `sub_queries` is empty; an empty `standalone` means the original question is used; on a model error the pipeline uses the original question (the existing fallback) and the trace records the reason.
- [x] **Implement, tests green.**

### Task 2: `CHITCHAT`

- `plan`: intent `CHITCHAT` → `Plan(strategy="NO_RETRIEVAL", queries=())`. The retrievers, given an empty list of queries, return `[]` without calling embedding or the DB.
- `generate`: intent `CHITCHAT` → its own prompt: answer briefly and politely, saying that it answers from the documents in the library; no citations, no facts stated.
- [x] **Failing tests:** "cảm ơn" with intent `CHITCHAT` does not call `embed_query`, does not query the DB, has an answer, and has no citations.
- [x] **Implement, tests green.**

### Task 3: conversations stored on the server side

- `conversations(id, created_at, updated_at)`; `messages(id, conversation_id → conversations, role, text, created_at)`, with an index on `(conversation_id, created_at)`. The migration only creates the two tables.
- `POST /query` takes `conversation_id` (optional). Without it, a new conversation is created; with one that does not exist, 404. The last 6 turns are read as `Query.history`. Once answered, the question turn and the answer turn are written in the same transaction as the trace.
- A failed request writes no turn: the history holds only what the user actually received.
- The response adds `conversation_id`.
- With the flag off, conversations are still stored (it is cheap, and the FE can use `conversation_id` right away); only the history is not read to rewrite the question.

- [x] **Failing tests:** the first question creates a conversation and returns `conversation_id`; for a second question with that `conversation_id`, the `rewrite` stage receives the history; an unknown `conversation_id` gives 404; a failed request writes no turn; with a long history, only the last 6 turns are taken, in the right order.
- [x] **Implement, tests green; the whole suite green.**

### Task 4: eval

- `--llm-cache PATH`: the results of `complete()` are stored by `(model, schema, message content)`, the same way as `--embed-cache`. Measuring again costs nothing when the questions and the history do not change.
- Print the intent distribution on Set A and Set B. No question in Set A may be `CHITCHAT`, because that would skip searching for a real question.
- [x] **Failing tests, implement, tests green.**

### Task 5: measure

- [x] **(calls OpenAI, needs agreement)** R4: 28 rewrites (Set A and Set B) and embeddings for the rewritten questions, stored in the cache. A few cents.
- [x] Compare with R3 on the same corpus: `recall@10` and MRR per kind of question. The aim is for `followup` to go up while the other kinds do not go down.
- [x] The time added to each question (the `rewrite` node in the trace). If it is too long, the option is to skip the rewrite for the first question of a conversation; decide only after measuring.
- [x] The intent distribution; list which questions were classified wrongly.

### Task 6: documentation

- [x] `docs/retrieval-design.md` §8.7 adds the R4 row; §6.1 records what R4 does and does not do yet.
- [x] `docs/fe-query-handoff.md`: the FE sends `conversation_id` back with the next question, and drops it for a "new conversation"; an opening prompt for the FE session.
- [x] `README.md`, `docs/system-design.md`.
- [x] The English version of this plan.

## Definition of Done

1. The whole suite is green, `ruff` is clean.
2. Flag off: the eval results are as in R3; the API only adds `conversation_id`.
3. Rewrite fails: there is still an answer, using the original question, and the trace records the reason.
4. The `recall@10` of `followup` is written into spec §8.7 beside R3's, together with the added time and the intent distribution.

## Status (2026-10-17)

Tasks 1 to 4 are done: 461 tests green, `ruff` is clean, the dev database has run migration `351b23ad1fd7`. Task 5 is waiting for agreement to call OpenAI. Task 6 still has the R4 row in spec §8.7 and the English version.

Departures from the plan:

- `messages.id` is an increasing number, not a UUID: the two turns of one question are written in the same transaction, where `now()` is the same, so the time cannot order them.
- `query_traces` gets a `conversation_id` column, so that the trace of a question can find the conversation that holds it.
- `RewriteResult` has no default values or length limits in the schema (structured outputs in strict mode require every field); the limit of 3 search queries is applied in code.
- The trace of a failed request is linked only to the conversation the client sent; a conversation that request had just created may have been rolled back along with it.

## Measured outcome (2026-10-17)

Same corpus, same reranker; R4 also turns on the question rewrite step.

| | `recall@10` | `MRR` | Missed |
| --- | --- | --- | --- |
| R3 | 0.850 | 0.583 | a01, a14, a16 |
| **R4** | **0.900** | **0.580** | a01, a15 |

- All four follow-up questions were rewritten correctly; the other 16 questions were kept word for word. The recall of follow-up questions went from 0.500 to 0.750; the other kinds of question did not change. The MRR of follow-up questions is still low (0.237): the right page is in the top 10 but not at the top.
- a15 is still missed: the result is page 17 instead of page 18. Page 17 also states that a vehicle must have a registration certificate, so at least part of the error lies in the label (the repo owner chose to keep this label as it is).
- No question in Set A was taken as small talk. 3/4 summary questions were classified `SUMMARIZE_DOC`.
- **Time:** at `reasoning_effort=low` the rewrite step takes a median of 1.9 s (at most 3.3 s). At `none`: 1.3 s (at most 1.8 s), with the same rewritten questions and the same recall, and it recognises 4/5 of Set B's out-of-scope questions instead of 1/5. So the rewrite step uses `none` (`REWRITE_REASONING_EFFORT`), while the answer step keeps `low`.
- The run cost 4 + 2 embeddings and 56 model calls; the results are stored in the cache (`data/eval-embeddings.json`, `data/eval-llm.json`).

Departures from the plan: `REWRITE_REASONING_EFFORT` was added, and `AsyncLLMProvider.complete` takes `reasoning_effort` per call.
