# Phase 3, R6: checking whether there is enough, and searching again if not

> **For whoever executes this:** do the tasks in order, each one by TDD (red test → code → green test). Do not commit on your own; when told to, group the work into a few large commits. Any step that calls the real OpenAI needs agreement first.

**Goal:** after the search, one model call judges whether the passages already found can answer the main part of the question. If not, the model says what is missing and writes a new search query; the pipeline searches again and adds the results to the context, for at most 3 rounds. The spec calls this the point where the system starts to become agentic.

**Architecture:** the loop has been in `pipeline.py` since R0 (`max_iterations`, accumulating context, with a test that keeps it alive). R6 replaces the `gate` stage (currently a no-op that always says "sufficient") with one call to `gpt-6-luna`, and raises `RETRIEVAL_MAX_ITERATIONS` from 1 to 3. The gate runs only when `RETRIEVAL_MAX_ITERATIONS > 1`; no new flag is added.

**Spec:** `docs/retrieval-design.md` §6.3 (gate), §4.3 (the loop), decision no. 12 (accumulating context), §8.2 (mean iterations, target 1.2–1.6).

## Global Constraints

- `RETRIEVAL_MAX_ITERATIONS=1`: the results are identical to R5.
- If the gate fails, it counts as "sufficient" and the answer uses what is already there (the fallback already in the pipeline). If the rounds run out and something is still missing, **still answer**, saying clearly that the documents may not be enough; never return an error because the search found too little.
- The gate's prompt has the calibration sentence from spec §6.3: "not sufficient" only when the missing part would make the answer **wrong or misleading**. Without this sentence, the gate says "not sufficient" almost every time and every question costs 3 rounds.
- The eval question set does not change. No change to `web/`; the `POST /query` contract does not change.

## Three things to fix or decide first

1. **The pipeline is answering the search query instead of the question.** When the gate says "not sufficient", `query` is replaced by the new search query, and the answer step receives exactly that. The user asks "phạt bao nhiêu", and the model answers a search query the gate wrote instead. R6 fixes this: the new search query is used only for searching; the answer step always receives the question as it stands after the rewrite step (R4). This is the only change in `pipeline.py`.
2. **Which questions do not need the gate.** For chitchat (nothing is searched) and summary questions (the whole outline is already there), searching again cannot help: the gate returns "sufficient" at once, without calling the model.
3. **Where to measure.** From R1 to R5, the eval scored at the output of `fuse` (the 10 candidates of a single search pass). R6 adds search passes, so scoring there shows nothing. R6 scores at the **final context** (the passages actually given to the model, after every round). The R5 baseline is measured again this way (free, from the cache) so that the two figures can be compared.

## Out of scope

- A gate that uses web sources (R8).
- A gate that checks the answer (that is R7; this gate checks the context **before** answering).
- Changing the context's 8,000-token budget. The accumulated context still fits within that budget, so a later round can add something only when the earlier rounds have not used it all.

## Files

| File | Work |
| --- | --- |
| `app/core/stages/gate.py` | new: `GateResult`, a prompt with the calibration sentence, `make_gate(provider)` |
| `app/core/pipeline.py` | the answer step receives the question, not the gate's search query |
| `app/core/generator.py`, `app/core/deps.py` | rounds run out with something still missing: tell the model what is missing so that it says so |
| `app/config.py`, `.env.example` | `GATE_REASONING_EFFORT` |
| `eval/run.py` | score at the final context; mean rounds; the gate's decisions |
| the matching tests; documentation in Task 6 | |

---

### Task 1: the `gate` stage

```python
class GateResult(BaseModel):
    sufficient: bool
    missing: str          # empty when sufficient
    refined_query: str    # empty when sufficient
```

- The model receives the question and the passages in the context (document name, heading, page, content) and returns `GateResult`. No field has a default value (structured outputs in strict mode require every field), so an empty string stands in for `null`.
- `refined_query` must use the words of the question and of the context, adding no new concepts (the same rule as the R4 rewrite step).
- Empty context: "not sufficient" without calling the model, and the new search query is the question itself.
- Intent `CHITCHAT` or `SUMMARIZE_DOC`: "sufficient" without calling the model.
- Trace: `gate_detail` with `sufficient`, `missing`, `refined_query`, and the number of passages considered.
- [x] **Failing tests** (`AsyncStubProvider`): "sufficient" and "not sufficient" go correctly into `GateVerdict`; the prompt has the calibration sentence and every passage in the context; empty context and chitchat/summary questions do not call the model; if the model fails, the pipeline counts it as "sufficient" (the existing fallback).
- [x] **Implement, tests green.**

### Task 2: the pipeline answers the right question

- `answer()` keeps the question as it stands after the rewrite step; the loop searches with the gate's search query; the answer step receives the question.
- The rounds run out with the gate still saying "not sufficient": the answer step is told the missing part (`missing`) so that it says clearly the documents may not fully answer that part.
- [x] **Failing tests:** the gate says "not sufficient" once, then "sufficient" → the answer step receives the original question, not the new search query; the rounds run out still "not sufficient" → the answer prompt has the missing part; the existing loop test is still green.
- [x] **Implement, tests green; the whole suite green.**

### Task 3: wire into `deps.py`

- `gate = make_gate(provider, GATE_REASONING_EFFORT)` when `RETRIEVAL_MAX_ITERATIONS > 1`, otherwise a no-op.
- `GATE_REASONING_EFFORT` defaults to `none`, like the rewrite step; Task 5 measures both `none` and `low`.
- [x] **Failing tests** through `POST /query`: the gate says "not sufficient" once → the trace has 2 rounds, `query_traces.iterations == 2`, and the context has passages from both rounds.
- [x] **Implement, tests green.**

### Task 4: eval

- Run the pipeline's search loop up to the final context (without calling the answer step), then score `recall` and MRR on the passages in the context, in order.
- Print the mean number of rounds, the number of questions for which the gate says "not sufficient", and, for each of those, what is missing and which query it searched again with.
- [x] **Failing tests, implement, tests green.**
- [x] **Measure the R5 baseline again at the final context** (free, from the cache).

### Task 5: measure

- [x] **(calls OpenAI, needs agreement)** R6 with `GATE_REASONING_EFFORT=none`, then `low`. One gate call per question (reading about 4–8 thousand tokens of context), one more for each further round; plus embeddings for the new search queries. An estimated 3–6 cents per run; cached.
- [x] Compare with R5 at the final context: `recall`, MRR per kind of question, mean rounds (spec: healthy is 1.2–1.6), the time added to each question.
- [x] Read every question for which the gate says "not sufficient": whether something really is missing, and whether the new search query is sensible.

### Task 6: documentation

- [x] `docs/retrieval-design.md` §6.3 and §8.7.
- [x] `README.md`, `docs/system-design.md`.
- [x] The English version of this plan.

## Definition of Done

1. The whole suite is green, `ruff` is clean.
2. `RETRIEVAL_MAX_ITERATIONS=1`: as in R5.
3. Gate failure or rounds run out: the question is still answered; the answer step receives the question, not the search query.
4. The mean rounds, `recall` and MRR at the final context, and the added time are written into spec §8.7 beside R5's.

## Outcome (2026-10-17)

Scored at the final context, against R5 scored the same way:

| | Context recall | Context MRR | Mean rounds | Gate time |
| --- | --- | --- | --- | --- |
| R5 (one round) | 0.900 | 0.635 | 1.00 | — |
| R6, gate `none` | 0.900 | 0.635 | 1.20 | median 1.2 s, max 3.4 s |
| R6, gate `low` | 0.900 | 0.635 | 1.15 | median 1.3 s, max 5.2 s |
| R5, rerank off | 0.850 | 0.598 | 1.00 | — |
| R6, rerank off, gate `low` | 0.850 | 0.598 | 1.20 | median 1.3 s |

- **The gate works correctly:** 1.15–1.20 rounds on average, within or close to the spec's healthy range of 1.2–1.6. The gate says "not sufficient" only for questions where something really is missing: a01 (the fine for a car running a red light), a14 (how long a driving licence is revoked for), a13, a11.
- **But no question is rescued.** The gate's search queries do not bring the missing page up. a01 has been pushed down by the CPU reranker since R3, and every repeat search goes through that same reranker. With rerank off, a14's repeat search ("Thời hạn tước quyền sử dụng giấy phép lái xe…") still does not reach page 15, because that phrase appears in almost every article of the decree.
- **At `none`**, the new search query often repeats the question word for word. **At `low`** it is more specific, but the results are the same.
- **Two bugs found and fixed along the way:**
  - The answer step used to receive the gate's last search query instead of the user's question.
  - The first round already used 7,712/8,000 tokens, leaving no room for later rounds. Now each later round gets an extra `CONTEXT_ROUND_EXTRA` (3,000 tokens), and only questions the gate judges to be missing something use it.
- **Conclusion:** what would help these questions is the search itself (a stronger reranker), not more rounds of the same search. `RETRIEVAL_MAX_ITERATIONS` stays at 1 by default: the gate costs one model call per question (about 1.2 s) with no measurable gain on this set. When it is turned on and the rounds run out, the answer says clearly which part the documents may not cover.
- **Cost of the run:** about 70 gate calls and 5 embeddings, all cached.

Departures from the plan: added `Context.rounds` and `CONTEXT_ROUND_EXTRA`; split `gather()` out of `answer()` in `pipeline.py`, so that the eval runs the pipeline's own loop without calling the answer step.
