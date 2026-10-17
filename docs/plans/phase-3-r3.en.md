# Phase 3, R3: reranking

> **For whoever executes this:** do the tasks in order, each one by TDD (red test → code → green test). Do not commit on your own; when told to, group the work into a few large commits. Any step that calls a paid API (OpenAI, OpenRouter) needs agreement first.

**Goal:** after fusion with RRF, score the candidates again with a cross-encoder and keep the 10 best, so that the right page comes to the top of the list. Measured with `MRR`, because reranking brings in no new candidate, so `recall` cannot go up.

**Architecture:** reranking sits inside the `fuse` stage, right after RRF (spec §4.1: `R2: RRF  R3: +rerank`). The API side has only **one** HTTP client, speaking Cohere's `/rerank` shape: `{model, query, documents, top_n}` → `results[{index, relevance_score}]`. That same shape is served by:

- `reranker` — a new container in the repo, running a small model on CPU. The default; it needs no GPU and no key;
- OpenRouter (`https://openrouter.ai/api/v1/rerank`), Cohere and vLLM — switched through `.env`, with no code change.

**Spec:** `docs/retrieval-design.md` §5.4 (reranking), §8.2 (MRR), §10 (configuration). This plan revises the spec's decision no. 7 (see below).

## Measured before the plan was written

For each question in the eval set: the first 50 candidates from keyword search, scored again, and the rank the right page then has. No paid API call.

| Model | Runs on | Hits in the top 10 | Right at rank 1 | MRR | Time for 50 pairs |
| --- | --- | --- | --- | --- | --- |
| no reranking | — | 15/20 | — | 0.399 | — |
| `Qwen3-Reranker-4B` | GPU, vLLM | 17/20 | 15 | **0.800** | 1.35 s (9 GB VRAM) |
| `Qwen3-Reranker-0.6B` | GPU | 17/20 | 10 | 0.625 | 0.66 s |
| `bge-reranker-v2-m3` | GPU | 17/20 | 9 | 0.607 | 0.19 s |
| `bge-reranker-v2-m3` | CPU | 17/20 | 9 | 0.607 | 10.5 s |
| `mmarco-mMiniLMv2-L12` (118M), ONNX int8 | CPU | 17/20 | 8 | 0.557 | **0.8 s** |
| `ViRanker` | GPU | 16/20 | 8 | 0.544 | 0.15 s |
| `AITeamVN/Vietnamese_Reranker` | GPU | 13/20 | 10 | 0.540 | 0.19 s |

What follows:

- Decision no. 7 (`bge-reranker-v2-m3`, self-hosted, run like Chandra) no longer stands: on CPU it takes 10 seconds, and as for the GPU, Chandra takes nearly all of it while it runs.
- The best model measured (`Qwen3-Reranker-4B`) is too heavy and too slow to self-host. Through an API it is available: OpenRouter, DeepInfra.
- The small ONNX int8 model is the only option that can run at any time on this machine. It is well behind Qwen-4B, but still clearly ahead of no reranking.

So: the CPU container is the default, so that R3 can always run; an API is one line of configuration once the repo owner has a key.

Limits of the table: the candidates come from keyword search, not from the hybrid list; only 20 questions.

## Global Constraints

- With the flag `RETRIEVAL_RERANK_ENABLED` off, the results are identical to R2.
- **If the reranker fails or times out, keep the RRF order**, record it in the trace and the log, and never break the request (spec §5.4).
- Do not hold a DB connection while waiting for the reranker: end the transaction before calling out (the same reason as for `expand`).
- The eval question set does not change. No change to `web/`; the `POST /query` contract does not change.
- The main suite's tests do not need `onnxruntime` and do not download a model.

## Out of scope

- Choosing the final rerank API: needs a key and one measurement run on the hybrid list (Task 5).
- Reranking on GPU, and running Qwen-4B self-hosted.
- Putting the article heading into the text sent to the reranker: not measured yet; R3 sends the child chunk's content, as in the measurement above.

## Files

| File | Work |
| --- | --- |
| `app/config.py`, `.env.example` | add `RETRIEVAL_RERANK_CANDIDATES`, `RERANKER_URL`, `RERANKER_MODEL`, `RERANKER_API_KEY`, `RERANKER_TIMEOUT_S` |
| `app/core/rerank_client.py` | new: an HTTP client in Cohere's shape |
| `app/core/stages/fusion.py` | RRF keeps `RETRIEVAL_RERANK_CANDIDATES`, reranks, cuts to `RETRIEVAL_RERANK_KEEP` |
| `app/core/deps.py` | wire the client into the stage by the flag; end the transaction before reranking |
| `reranker_service/` | new: FastAPI + onnxruntime, `POST /rerank`, `GET /health` |
| `docker-compose.yml` | the `reranker` service |
| `eval/run.py` | cache the question embeddings so that measuring again costs nothing |
| `tests/test_rerank_client.py`, `tests/test_fusion.py`, `tests/test_reranker_service.py`, `tests/test_hybrid_search.py` | tests |
| `docs/retrieval-design.md`, `docs/system-design.md`, `README.md` | Task 6 |

---

### Task 1: the rerank client

**Produces:**

```python
class RerankError(Exception): ...

class RerankClient:
    def __init__(self, url: str, model: str, api_key: str | None, timeout_s: float,
                 transport: httpx.AsyncBaseTransport | None = None) -> None: ...
    async def scores(self, query: str, documents: list[str]) -> list[float]:
        """Scores in the same order as `documents`. A network error, an HTTP error, a timeout
        or a missing index all become RerankError."""
```

- `POST {url}/rerank` with `{"model", "query", "documents", "top_n": len(documents)}`; with a key, add `Authorization: Bearer …`.
- The results come back in score order; the client puts them back in place by `index`, so a missing or duplicated index is an error.
- Empty `documents` returns `[]`, with no network call.

- [x] **Failing tests** (`httpx.MockTransport`, no network): the right body and headers; scores in the original order even when the server returns them in another order; with no key, no header is sent; 500, a timeout, JSON missing an index → `RerankError`; an empty list makes no network call.
- [x] **Implement, tests green.**

### Task 2: the `fuse` stage with reranking

- `reciprocal_rank_fusion` stays as it is. With reranking on, RRF cuts at `RETRIEVAL_RERANK_CANDIDATES` (default 50) instead of `RETRIEVAL_TOP_K`; the reranker scores that many; `RETRIEVAL_RERANK_KEEP` (10) are kept.
- A candidate keeps its original `source`; `score` becomes the rerank score; `rank` is numbered again from 1.
- `RerankError` → the RRF order cut at `RETRIEVAL_TOP_K`; the trace records `rerank_detail` with `fallback` and the reason.
- The trace always has `rerank_detail`: the number of candidates sent, the time, and the RRF ranks of the 10 candidates kept (to show which ones reranking pulled up).

- [x] **Failing tests** with a fake reranker: reranking reverses the order; exactly 10 are kept; on an error it falls back to RRF and the trace records the reason; with the flag off it is the same as R2.
- [x] **Wire into `deps.py`**: the client is created when `RETRIEVAL_RERANK_ENABLED`; `await session.commit()` before calling the reranker; a test checks that no transaction is open while the reranker runs (the same way as `test_no_transaction_is_open_while_the_model_writes`).
- [x] **Tests green; the whole suite green.**

### Task 3: the `reranker` container (CPU)

- `reranker_service/app.py`: FastAPI, `POST /rerank` in Cohere's shape, `GET /health`. The scoring part is a function passed in, so that the tests run with a fake one.
- `reranker_service/onnx_scorer.py`: `onnxruntime` + `tokenizers`; the model `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; the ONNX file is chosen by an environment variable. The default is `model_quint8_avx2.onnx` because it runs on any CPU with AVX2; `model_qint8_avx512_vnni.onnx` is the one measured at 0.8 s on this machine.
- A separate `reranker_service/Dockerfile`; the model is downloaded at build time, so the container starts without the network.
- `docker-compose.yml`: the `reranker` service, with CPU and RAM limits, `restart: unless-stopped`; `api` has `RERANKER_URL=http://reranker:8200`.

- [x] **Failing tests** (`tests/test_reranker_service.py`, a fake scoring function): the right response shape, `index` matching the input, `top_n`, an empty list, `/health`.
- [x] **Implement, tests green.**
- [x] **Measure the two ONNX files on this machine** (no GPU needed, no API call): time for 50 and 100 pairs, MRR on the keyword candidates. Choose the default file from the figures.
- [x] **Build and run the container**, and call `/rerank` once from `api`.

### Task 4: an eval that measures again at no cost

- `eval/run.py --embed-cache data/eval-embeddings.json`: the question embeddings are stored by `(model, dimensions, text)`; the next run reads them from the file. `data/` is already gitignored.
- [x] **Failing tests:** the second run does not call the provider; changing the model or the number of dimensions does not use the old cache.
- [x] **Implement, tests green.**

### Task 5: measure

- [x] **(calls OpenAI one last time: 20 embeddings, needs agreement)** R2 with the cache on, to fill the cache.
- [x] **(free)** R3 with the CPU container.
- [ ] **(needs an OpenRouter key and agreement; under 0.10 USD)** R3 through OpenRouter: `qwen/qwen3-reranker-4b` if the account can use it, then Cohere `rerank-4-fast` and Voyage `rerank-2.5-lite`.
- [x] Record `recall@10`, MRR per kind of question, and the time of the `rerank` node in the trace. If MRR does not go up compared with R2, the flag stays off and the spec records why (spec §8.2: "If MRR does not move, the reranker is not worth a container").

### Task 6: documentation

- [x] `docs/retrieval-design.md`: decision no. 7 rewritten from the measurement table; §5.4 and §10 following the new configuration. (§8.7 waits for Task 5.)
- [x] `docs/system-design.md`, `README.md`: the `reranker` service, how to switch to an API.
- [x] The English version of this plan.

## Definition of Done

1. The whole suite is green, `ruff` is clean; the tests do not download a model and make no network call.
2. Flag off: the trace and the results are as in R2.
3. Reranker off or dead: the request still gets an answer, and the trace records the fallback.
4. Switching between the CPU container and OpenRouter takes only `.env`.
5. The R3 figures (at least the CPU version) are written into spec §8.7 beside R2's.

## Status (2026-10-16)

Tasks 1 to 4 are done; the whole suite is green, `ruff` is clean. Task 5 stopped at its first step, waiting for agreement to call OpenAI. Task 6 still has the R3 row in spec §8.7 and the English version of the plan.

Departures from the plan, with reasons:

- **The container uses sentence-transformers instead of onnxruntime + tokenizers.** The model is not settled, and `gte-multilingual-reranker-base` needs its own code; sentence-transformers loads any cross-encoder, running on ONNX or torch. The price is a heavier image (2.57 GB).
- **The model is downloaded into `/opt/model` at build time and loaded from there.** Loaded by repo name, the library still asks Hugging Face for the list of files at every start, even with offline mode turned on. This bug came to light because the build step loads the model again with the network off.
- **The default ONNX file is the `avx512_vnni` one** (measured at 0.8 s on this machine), not the `avx2` one the plan states, because the `avx2` one has not been measured. On a machine without AVX-512 VNNI, change `RERANKER_LOCAL_ONNX_FILE`.
- **More internationally known models were added to the measurement table** (at the repo owner's request): `gte-multilingual-reranker-base`, `bge-reranker-v2-m3` in int8, `Qwen3-Reranker-0.6B` on CPU.

The container, measured over HTTP: 50 passages 0.62 s, 100 passages 1.2 s, 759 MiB RAM; MRR 0.539 on the keyword candidates (measured directly with onnxruntime it is 0.557).

## Measured outcome (2026-10-17)

Same corpus, same question vectors (the R2 run paid for 20 embeddings and stored them in the cache, the R3 run read them from the cache):

| | `recall@10` | `MRR` | Missed |
| --- | --- | --- | --- |
| R2 | 0.750 | 0.542 | a08, a13, a14, a16, a18 |
| R3, CPU reranker `mmarco-mMiniLMv2` int8 | **0.850** | **0.583** | a01, a14, a16 |

Reranking pulled a08, a13 and a18 in from outside RRF's top 10, so recall goes up too, not only MRR. a01 is lost; the MRR of questions containing a code falls from 0.750 to 0.613. The measurement through an API (OpenRouter) has not been done: there is no key yet.

## After the independent review (2026-10-17)

The reviewer found no critical bug, but 8 important ones. They have been fixed, each with a red → green test:

- **Not every reranker failure kept the RRF order.** A wrong URL (`httpx.InvalidURL`), a number that is too large or JSON nested too deeply escaped the client; the stage then fell back to the pipeline's no-op, that is the first 10 vector results, losing every keyword result. Now the client turns every failure into `RerankError`, and the stage catches every failure so as to keep the RRF order of both sources.
- **NaN and infinity were accepted**, which put the order wrong and kept the trace from being written to Postgres. They are now rejected, along with indices that are not integers.
- **The eval could not tell a real rerank from a fallback.** The eval now reads the model from each question's trace, and refuses to print figures if any question was not reranked.
- **A burst of simultaneous requests made every caller time out**, and requests already abandoned kept running. Now the container scores one request at a time and lets at most 2 requests wait, returning 503 at once beyond that; a request the caller has abandoned is not scored; `/health` needs no worker thread, so it always answers.
- **`RERANKER_TIMEOUT_S` was not a deadline for the whole call** (httpx times each read separately). It is now wrapped in `asyncio.timeout`.
- **The URL and model configuration was easy to get wrong.** `RERANKER_MODEL` is no longer required: left empty, the container uses the model it was built with. `.env.example` and the README say plainly that the three variables are left empty when using the container.
- **`cpus: 8`** kept the container from starting on a machine with fewer than 8 CPUs: removed. Port 8200 is now open only on loopback.
- **Changing the model easily broke the build:** when the repo does not have the chosen ONNX file, a clear error is reported; a note for CPUs without AVX-512 VNNI.
- Small items: the rerank time is not added to the trace twice; a limit on the input size; the container's library versions pinned; the embedding cache is written safely and is still saved when a run fails partway; a check that `RETRIEVAL_RERANK_CANDIDATES ≥ RETRIEVAL_RERANK_KEEP, RETRIEVAL_TOP_K`.

Not fixed: `httpx.AsyncClient` is created anew on every call (one extra TLS handshake with an external API); the `.gitignore` pattern `data/` is not anchored to the repo root (the repo owner keeps this file out of commits).

After the fixes: 444 tests green. R3 measured again from the cache: still 0.850 / 0.583, this time with traces confirming that all 20 questions were really reranked. 30 simultaneous requests sent to the container: 3 answered within 2.4 s, 27 refused within 0.24 s, the next request 0.66 s.

## Model chosen (2026-10-17)

The repo owner chose the current CPU model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, int8, in the `reranker` container. Spec decision 7 and §5.4 record it. The OpenRouter measurement was not done.
