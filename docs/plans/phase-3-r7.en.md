# Phase 3, R7: checking whether the answer sticks to its sources

> **For whoever executes this:** do the tasks in order, each one by TDD (red test → code → green test). Do not commit on your own; when told to, group the work into a few large commits. Any step that calls the real OpenAI needs agreement first.

**Goal:** after answering, a separate model call reads the answer beside the sources and points out which sentences no source supports. Those sentences are dropped and replaced by a line saying the documents do not cover it. Measured by fabrication rate (spec §8.2) on Set A and Set B, with Set B scored for the first time.

**Architecture:** the `reflect` stage has been there since R0 (a no-op that returns the answer unchanged, with the fallback already in `pipeline.py`). R7 replaces it with `make_reflect(provider)` when `RETRIEVAL_REFLECT_ENABLED=true` (the flag already exists, default `false`). `pipeline.py` changes in only two places: `reflect` also receives the question (to skip chitchat), and the answer step plus the check are split out into `respond()` so that the eval can score both the draft and the final version.

**Spec:** `docs/retrieval-design.md` §6.5 (reflection), §6 (failure policy: if reflect fails, the answer is kept as is), §8.2 (fabrication rate, citation accuracy), §8.4 (Set B).

## Global Constraints

- `RETRIEVAL_REFLECT_ENABLED=false`: the results are identical to now.
- If reflect fails, returns the wrong schema, or points to a sentence that is not in the answer: the answer is kept as is for that part. Never return an error because of the check.
- Drop sentences, do not rewrite them (spec §6.5): rewriting costs another call with no guarantee of being more correct; dropping the whole answer is usually an overreaction.
- Citations are recomputed from the text after sentences are dropped: a `[n]` that appeared only in a dropped sentence loses its citation too.
- The eval question set does not change. No change to `web/`; the `POST /query` contract does not change.

## Three things to decide first

1. **What language the line replacing a dropped sentence is written in.** The answer follows the language of the question, and the pipeline does not know what language that is. Proposal: add a `note` field to `Reflection` (the spec has only `grounded`, `unsupported`): one sentence, in the same language as the answer, saying the documents do not cover the dropped part; empty when nothing is dropped. No second call is needed.
2. **Who scores the fabrication rate.** `reflect` cannot be left to score itself. Proposal: the eval has its own judge prompt (the same `gpt-6-luna`, since the provider has only one model, but at reasoning `high`), and every sentence flagged by the judge or by reflect is printed to be read by hand. 28 answers can be read by hand, and reading by hand is the real check; the judge only shows where to read.
3. **Measure before/after on the same draft.** The answer step does not give the same answer twice, so running the eval twice (reflect off, then on) mixes the difference with noise. The eval generates the draft once, scores it, passes it through `reflect`, then scores the final version. The two figures come as a pair within one run, and the cache makes later runs free.

## Out of scope

- Acting on the `OUT_OF_SCOPE` intent (skipping the search, declining at once). Since R5 the rewriter recognises only 1/5 of the out-of-scope questions, and mistaking a real question for one loses that whole question. R7 measures whether the answer declines correctly; whether to act on the intent is decided from that figure.
- Checking whether a citation points to the right page (`[n]` points to the right source, but that source does not say what the sentence says). Reflect sees part of this because it reads each sentence beside the source, but there is no separate fix step.
- Streaming the answer. Once streaming is in, what has already been sent can no longer be dropped; at that point one has to choose between checking before sending and sending before checking. Noted here so it is not forgotten.

## Files

| File | Work |
| --- | --- |
| `app/core/stages/reflect.py` | new: `Reflection`, prompt, `make_reflect(provider, reasoning_effort)`, dropping sentences |
| `app/core/generator.py` | split out the part that builds the source list so that `reflect` can reuse it (the gate's prompt also copies this part) |
| `app/core/pipeline.py`, `app/core/stages/noops.py` | `reflect(query, answer, context, trace)`; `respond()` returns both the draft and the final version |
| `app/core/deps.py` | wire `reflect` in by the flag |
| `app/config.py`, `.env.example` | `REFLECT_REASONING_EFFORT` |
| `eval/run.py`, `eval/judge.py` | `--full` mode: draft → score → reflect → score, both Set A and Set B |
| `eval/inject.py`, `eval/injections.yaml` | plant one fabricated sentence in each draft, measure how many reflect catches |
| the matching tests; documentation in Task 5 | |

---

### Task 1: the `reflect` stage

```python
class Reflection(BaseModel):
    grounded: bool
    unsupported: list[str]   # the unsupported sentences, verbatim; empty when grounded
    note: str                # one sentence, in the answer's language; empty when grounded
```

- The model receives the question, the sources (with the same document label, heading, and page as the answer step) and the answer with its `[n]`. The prompt says clearly: a sentence is supported by a source when the source says it, even in different words; a sentence saying the documents do not cover something is not a claim; copy sentences verbatim, without editing them.
- Dropping sentences: match after collapsing whitespace; also drop the `[n]` right after the sentence; a bullet line left with only its marker is dropped entirely. A sentence not found in the answer is skipped and recorded in the trace.
- Every sentence dropped: the answer is only the `note`.
- Intent `CHITCHAT`: do not call the model (there are no sources and no claims).
- Trace: `reflect_detail` with `grounded`, `unsupported`, the number of sentences dropped, the unmatched sentences, `asked_model`.
- [x] **Failing tests** (`AsyncStubProvider`): when grounded, the answer and citations are unchanged; an unsupported sentence is dropped along with its `[n]` and the `note` is added, a citation found only in that sentence disappears, a citation still used in another sentence is kept; an unmatched sentence is skipped and recorded in the trace; when everything is dropped only the `note` is left; empty bullets are dropped; chitchat does not call the model; the prompt has every source and the answer; if the model fails, the pipeline returns the draft (the existing fallback).
- [x] **Implement, tests green.**

### Task 2: wire into `deps.py`

- `reflect = make_reflect(provider, REFLECT_REASONING_EFFORT)` when `RETRIEVAL_REFLECT_ENABLED`, otherwise a no-op.
- `REFLECT_REASONING_EFFORT` defaults to `low`; Task 4 measures both `none` and `low`, then settles it.
- [x] **Failing tests** through `POST /query`: reflect drops one sentence → both the response and the message stored in the conversation are the version with it dropped, and `citations` match that text; the trace has `reflect_detail`; with the flag off, no extra model call is made.
- [x] **Implement, tests green; the whole suite green.**

### Task 3: eval

- `eval/judge.py`: `Judgement{declined: bool, unsupported: list[str]}`, a judge prompt (different from reflect's prompt), reasoning `high`.
- `--full`: for each question of Set A and Set B, use the context the search loop already has (no new search), generate a draft, score the draft, pass it through `reflect` (via the pipeline's fallback), score the final version.
- Print:
  - the fabrication rate (answers with at least one sentence the judge finds unsupported) before and after reflect, separately for Set A and Set B;
  - Set B: whether the out-of-scope questions are declined (5), whether the questions that sound like they need the web are answered (3);
  - citation accuracy before and after (dropping sentences can drop correct citations);
  - each sentence reflect dropped, along with whether the judge flagged that sentence in the draft: reflect dropping a sentence the judge finds supported is a sign of an over-drop;
  - reflect's time.
- Remove the second pipeline run in the current `--full` (it searches again from scratch for each question).
- [x] **Failing tests** (no network calls): the fabrication rate and the decline decisions are computed correctly from fake `Judgement`s; before/after use the same draft (the answer step is called only once per question).
- [x] **Implement, tests green.**

### Task 4: measure

- [x] **(calls OpenAI, needs agreement)** `--full` with reflect at `none`, then `low`. Per question: one answer, one reflect, two judgings, each reading about 8 thousand tokens of context; 28 questions. An estimated 0.3–0.6 USD for the first run; later runs pay only for reflect and the judging of the final version. Cached.
- [x] Read by hand every flagged sentence (by reflect or by the judge): real fabrication, over-drop, or the judge being wrong.
- [x] Settle `REFLECT_REASONING_EFFORT` and whether `RETRIEVAL_REFLECT_ENABLED` is on by default, based on: how much the fabrication rate falls, how many sentences are over-dropped, how many seconds are added.

### Task 5: documentation

- [x] `docs/retrieval-design.md` §6.5 and §8.7.
- [x] `README.md`, `docs/system-design.md`.
- [x] The English version of this plan.

## Definition of Done

1. The whole suite is green, `ruff` is clean.
2. `RETRIEVAL_REFLECT_ENABLED=false`: as now.
3. Reflect failure: the question is still answered, with the draft.
4. The fabrication rate before/after, Set B's decline decisions, the number of over-dropped sentences, and the added time are written into spec §8.7.

## Outcome (2026-10-17)

Ran `--full` on 28 questions (Set A 20, Set B 8). One draft per question; the judge scores the draft, then scores the version after the check again.

| | Fabrication, Set A | Fabrication, Set B | Set B correct | Citations on the right page | Over-drops | Check time |
| --- | --- | --- | --- | --- | --- | --- |
| Draft | 0.000 | 0.000 | 7/8 | 0.900 | | |
| First prompt, `none` | 0.000 | 0.000 | 7/8 | 0.850 | 1 (a13) | median 1.4 s |
| First prompt, `low` | 0.100 | 0.000 | 7/8 | 0.800 | 2 (a08, a13) | median 1.5 s, max 6.6 s |
| Revised prompt, `none` | 0.050 | 0.000 | 7/8 | 0.900 | 1 (a05) | median 1.4 s, max 3.1 s |
| **Revised prompt, `low`** | **0.000** | **0.000** | **7/8** | **0.900** | **0** | median 1.5 s, max 5.4 s |

- **There is nothing to catch.** The drafts have no sentence the judge finds unsupported. All 5 out-of-scope questions in Set B are declined by the answer step, even though the rewrite step recognises only 1 of the 5 as out of scope. Set B's only miss is b06: the corpus can answer it, but the answer declines. That is a job for the search and for R8.
- **The first prompt only did harm**, reading each sentence by hand:
  - a13: page 17 states the fine at the head of the clause ("Phạt tiền từ 400.000 đồng đến 600.000 đồng…"), while the act is in point i below it. The check could not join the two, and dropped the only correct answer.
  - a08: dropped a correct answer for leaving out an exception, that is, an omission rather than a fabrication. The replacement line even told the user that "câu trả lời không nêu ngoại lệ này".
  - The "after" fabrication rate at `low` (0.100) comes from those replacement lines themselves.
- **Fixed the prompt:** an omission is not an unsupported claim; a fine stated at the head of a clause applies to every point below it; the replacement line does not mention "the answer". After the fix, `low` drops nothing. `none` still over-drops one sentence in a05. A rule "when every sentence is flagged, keep the draft" was added at one point, but has been removed: it was drawn from those very two cases on the eval set, and it switches reflect off exactly on an answer that is fabricated from start to finish.
- **Planted fabrications: 20/20 caught.** The drafts have nothing to catch, so each Set A draft had one fabricated sentence planted in it (`eval/injections.yaml`, run with `python -m eval.inject`), each one checked to be absent from the sources: 4 sentences assign an act to another act's fine bracket, 5 get a figure wrong, 11 add a rule that does not exist. At `low`, reflect drops all 20 and drops nothing else.
- **Does the clause-head line overfit:** that line was written after the a13 case, so the worry was that it makes reflect lenient towards sentences that put an act in the wrong fine bracket. Measured on its own by removing just that line from the prompt: 19/20 caught, and on a13 reflect drops the correct fine sentence while keeping the fabricated one. The other three wrong-bracket sentences are caught both with and without the line. So it does not make reflect lenient; it helps it read the clause-and-point structure correctly. Limits: the 20 fabrications were written by hand by a language model (Claude), so they may be easier to spot than the answering model's own; only 4 are wrong-bracket sentences; each variant was run once.
- **Conclusion:** `REFLECT_REASONING_EFFORT=low`. `RETRIEVAL_REFLECT_ENABLED` is off by default like every other flag. Turning it on trades about 1.5 s per question for catching fabricated sentences: on this set the real drafts have not fabricated a single sentence, while the planted ones are all caught.
- **Cost of the run:** about 210 model calls and 8 embeddings, all cached.
- **Departures from the plan:** `reflect` also receives the question and `respond()` is split out of `answer()` in `pipeline.py`; added the planted-fabrication measurement (`eval/inject.py`); the eval no longer prints the time of calls taken from the cache.
