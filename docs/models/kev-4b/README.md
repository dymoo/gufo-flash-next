# Kev-4B (TypeSafe System One)

Kev is an open (Apache-2.0) replacement for TypeSafe's Jev classifier:
`jaredpalmer/kev` trains a LoRA (r16, α32, every linear projection) and a
pointer head on `Qwen/Qwen3.5-4B-Base`. It answers typed questions about a
state in one forward pass: no text is generated. Gufo serves it beside
Qwen3.8 Flash-Next in the same process, on the same GPU, behind the TypeSafe
`POST /v1/systemone` API.

Status: serving in `gufo serve systemone` and beside Flash-Next through
`gufo serve llm --systemone-model FILE`, paced to a share of GPU time while
Flash-Next is busy (measured below). Pinned sources: `jaredpalmer/kev@3e1cd3bb` (reference
implementation), `jaredpalmer/kev-4b@139fdd94` (adapter, `head.pt`),
`Qwen/Qwen3.5-4B-Base@1001bb4d`.

## The model

Text layout (`kev/model.py:encode`), all ids from the Qwen3.5 tokenizer:

```text
<|fim_prefix|> state
per question k:  <|fim_middle|> instructions
                 per option: <|box_start|> option text <|box_end|>
                 <|fim_suffix|>
```

Caller text is tokenized with `<|name|>` rewritten to `<¦name¦>`, so callers
cannot forge delimiters. Question *k* sees the state and its own branch only.
The backbone is hybrid (24 Gated DeltaNet layers, 8 gated-attention layers,
one attention layer every four), so each question runs as a causal row that
continues from the state: attention reads the state's keys and values, and
DeltaNet continues from the state's final recurrent state and conv window.
Branch positions continue from the state length.

Readout: final RMSNorm at `<|fim_suffix|>` (decide) and at every
`<|box_end|>`; `z_i = k(h_opt_i) · q(h_decide) / sqrt(256) / T`, with
`q, k: Linear(2560, 256)` and the calibrated temperature `T` from `head.pt`;
softmax over the question's options.

API mapping (`kev/api.py`): `noul` → options `no[: desc]`, `yes[: desc]`,
answer `p(yes)`; `choice` → `name` or `name: desc`, argmax, probabilities,
confidence `(p_max − 1/K)/(1 − 1/K)`; `score` → rendered levels, expected
level, legend, score confidence. JSON states and instructions are flattened
by `render()` (Python `str()` scalars, `key: value` lines, `- ` list items).
Probabilities are rounded to four decimals. Usage reports input tokens and
the token count of the serialized answers.

## Work per request

| Quantity | Kev-4B |
|---|---|
| Layers | 32 (24 DeltaNet, 8 attention, last layer attention) |
| GEMM parameters | 3.55 B (embedding unused as a head: no vocabulary projection) |
| GEMM work | 7.1 GFLOP per token |
| DeltaNet state | 24 × 32 heads × 128 × 128 fp32 = 50 MB per sequence |
| Attention KV | 8 layers × 4 heads × 256 × 2 × bf16 = 32 KiB per token |

On Strix Halo (measured: 55 TFLOPS FP16 WMMA, 239 GB/s DRAM) a pass is
bandwidth-bound below ~190 tokens and compute-bound above. A typical request
(300-token state, six 50-token questions) is ~600 tokens, ~4.3 TFLOP: about
80–100 ms of the whole GPU at BF16 rates. Throughput therefore comes from
large passes and from never computing a state twice.

Memory: the GTT pool is 100 GiB and Flash-Next production holds 88.7 GiB, so
Kev must fit in ~8 GiB. Q8_0 weights (~4.5 GiB with the embedding) fit; BF16
(~8.4 GiB) does not. Q8_0 passed the quality gate below and is the default.

Kev reserves all its memory at startup and admits work against it:

| Reservation | Size |
|---|---|
| Weights (Q8_0) | 4.5 GiB |
| Pass workspace at 8,192 packed tokens | ~1.2 GiB |
| State bank and prefix cache (KV at 32 KiB/token, 48 MiB DeltaNet per state) | 1.5 GiB cap |

A pass holds at most 8,192 tokens; a longer request runs its state in
chunks and its rows in groups. A state's bank entry costs 48 MiB plus 32 KiB
per token, so a 32,768-token state needs 1.05 GiB; the served state limit is
32,768 tokens (Jev's documented state budget), and the cache evicts
least-recently-used entries to stay under its cap.

## Design

### One fused pass per batch

A Kev pass packs every token of a batch into one varlen buffer: new states
first, then every question row of every request. Each layer runs its GEMMs
once over all packed tokens. Sequence operators run in two phases:

1. **State phase:** conv and DeltaNet over each new state from zero, writing
   the final recurrent state and conv window into the *state bank*; attention
   keys and values for state tokens go to the bank as well.
2. **Row phase:** conv and DeltaNet over each question row, reading its
   state's bank entry as the initial state and never writing a final state.
   Attention for a row reads the state's bank keys (all visible) and its own
   keys (causal).

States served from the prefix cache skip phase 1. Identical states inside
one batch run once. The reference runs a state pass and a row pass
separately (two weight reads); the fused pass reads the weights once and
gives the GEMMs one larger M.

The last layer is attention. Its keys and values are computed for every
token, but query, attention, output projection, residual and MLP run only
for readout tokens (decide and option ends). For state tokens that layer
needs only keys and values. The final norm and pointer head run on readout
rows only.

### Kernels

Reused, already tuned for gfx1151 in `src/models/qwen/hip`: Q8_0 W8A8 WMMA
prefill GEMMs, embedding, RMSNorm, SwiGLU, residual, q/k norm.

New, Kev-specific:

- **Varlen conv + SiLU** with a per-sequence initial window (bank entry or
  zeros) and optional final-window store.
- **Varlen gated delta rule**: one work item per (sequence, value head, row
  group), based on the Flash-Next barrier-free token-serial kernel; the
  initial state is read-only (bank or zero), the final state is stored only
  for state sequences. Gate, beta sigmoid and q/k L2 norm fused.
- **Block-causal prefix attention** (WMMA, head dim 256, GQA 4:1, sigmoid
  output gate fused): queries of all rows sharing a state are tiled together,
  so each tile reads the state's keys once for every row (cascade
  attention); tiles whose rows cannot see each other are skipped.
- **Positions-array RoPE** (partial rotary 64 of 256, θ = 1e7), because
  branch positions restart after the state.
- **Readout**: gather readout rows, final RMSNorm, pointer projections, dot,
  temperature and per-question softmax in one kernel.

A chunked (WY) DeltaNet for long states is a later step, taken only if
profiles show the token-serial state phase dominating long-state passes.

### Batching

A Kev scheduler thread owns one executor. Requests are parsed, rendered,
tokenized and admitted on the HTTP thread. As in the reference, a state past
the limit is truncated (first tokens kept, usage counts the tokens served)
and a question row past state + 8,192 tokens is a 422. The scheduler builds
a pass from everything queued, up to a token budget, whenever the GPU
arbiter grants it. Requests arriving during a pass join the next one. The
prefix cache keeps state bank entries keyed by the exact state token ids,
evicted least-recently-used within a byte budget.

### Sharing the GPU with Flash-Next

Both models live in one `gufo serve llm` process (`--systemone-model FILE`).
Kev loads first so the text model sizes its KV pool from what is left.

The two schedulers keep their own threads and streams, so their kernels
run concurrently. Kev is paced instead of arbitrated: every four layers its
worker drains its stream and, while the text model has decoding or
prefilling sessions, sleeps for `slice * (1 - share) / share`
(`--systemone-share`, default 0.25). With the text model idle Kev runs flat
out. The text scheduler never waits, and no lock is shared.

Measured on the Strix Halo test host (2026-09-28, windows 17 and 18): the 40-agent Flash load
of the Flash-Next benchmarks, with 16 Kev clients sending new six-question
tickets throughout.

| Kev share | Flash decode tok/s | Flash ITL p50 | Flash TTFT p50 | Kev requests/s | Kev latency p50 |
|---|---|---|---|---|---|
| Kev idle | 41.7 | 144 ms | 2.4 s | – | – |
| unpaced | 23.1 | 303 ms | 5.5 s | 10.1 | 1.6 s |
| 0.5 | 30.8 | 215 ms | 3.5 s | 5.2 | 3.1 s |
| **0.25 (default)** | **39.1** | **172 ms** | **3.0 s** | **2.7** | 5.9 s |
| Flash idle | – | – | – | 15.8 | 4.0 s (64 clients) |

Unpaced concurrency does overlap work: Flash keeps 55% and Kev 64% of their
solo rates. But it gives Kev most of the GPU, so pacing trades that overlap
for the text model's latency. A one-question classification under Flash
load takes a few hundred milliseconds at the default share.

### API

- `POST /v1/systemone`: TypeSafe request and response shapes, `model`
  echoed (`kev-latest`, `jev-latest`, and the served id are accepted),
  `x-typesafe-request-id` echoed or generated on every response, errors
  included. The endpoint has its own error path: `{"detail": ...}` with
  401/422/429 (`Retry-After`)/503, also for body parse failures.
- `GET /v1/models`: the OpenAI `data` list plus a TypeSafe `models` list of
  cards, so both SDKs read the same endpoint.
- `/v1/runtime` gains a `systemone` section: pending and its limit,
  requests, rejections, batches, tokens, questions, passes (count, tokens,
  GPU time), the state cache (hits, misses, states), `gpu_share` and
  `paced_ms`. The same counters are exported in `/metrics` as
  `gufo_systemone_*` ([Metrics](../../SERVER.md#metrics)).
- Limits: 128 questions per request; rendered JSON at most 32 levels deep
  and 4 MiB; waiting requests at most half of `--max-connections`.

### llm-router

System One is a router modality: `POST /v1/systemone` with the
router's keys, priorities and metering, routed to local Kev on Gufo. Cloud
Jev (TypeSafe) is a separate, explicit deployment that a key may be allowed
to use; the router never silently swaps Kev for Jev, because their
probabilities differ. The router's own classifier can later point its Jev
mode at local Kev once it passes the classifier qualification gate.

## Quality gate

1. Tokenizer parity: exact token ids against the Hugging Face tokenizer for
   rendered requests, including delimiter strings, other added tokens
   (`<think>` is still recognized inside caller text, as in the reference),
   nested JSON, and Python `str()` spellings of numbers and booleans.
2. Probability parity with the reference implementation (PyTorch, fp32,
   unmerged adapter) on fixed records, including long states and cache hits.
3. `kev.benchmark --remote` on the frozen suites behind the published Kev-4B
   results, for BF16 and Q8_0: accuracy, ECE, Brier and confidence bins,
   with near-tie questions reported separately.

## Results (2026-09-28, Strix Halo, Q8_0)

Quality through `/v1/systemone` with the reference's own scorer
(`python -m kev.benchmark --remote`), against the released checkpoint's
fp32 numbers (`runs/release/kev-4b-r10.json` in the Kev repository):

| Suite (development) | Questions | Reference acc / Brier / ECE | Gufo Q8_0 acc / Brier / ECE |
|---|---|---|---|
| decision-v7 (trained sources) | 1,264 | 0.8726 / 0.1824 / 0.0127 | 0.8695 / 0.1831 / 0.0126 |
| transfer-v4 (new sources) | 656 | 0.8171 / 0.2430 / 0.0416 | 0.8186 / 0.2429 / 0.0390 |

Parity (`kev_parity_test`, 20 adversarial requests against the BF16
reference): token ids identical; probabilities within 0.01, 0.022 on a
10,563-token state; the two argmax differences are reference near-ties
(0.434/0.444, 0.505/0.495). Batched and single answers are bit-identical.

Speed (`kev_bench`, model time; production Flash-Next idle on the same GPU):

| Workload | Result |
|---|---|
| One ticket (~190 tokens, six questions) | 85 ms |
| Batched tickets | ~3,100 tokens/s, ~16 requests/s |
| 10,563-token state, two questions | 4.0 s (43 s on the Qwen27B executor route) |

The dense Q8 GEMMs take ~74% of a pass at ~30 TOPS; the repository's best
dense GEMMs reach 30-35 TFLOPS against a 57 TOPS WMMA peak, so throughput is
near the chip's practical ceiling. `KEV_PROFILE=1` prints per-stage times.

## Building the artifact

The served file is the base with the LoRA merged in fp32, converted by
llama.cpp's converter, with the pointer head and temperature appended. Run
on a host with ~20 GB of free disk and memory, not a workstation:

```sh
python tools/kev/merge_adapter.py Qwen3.5-4B-Base/ kev-4b/ kev-4b-merged-f32/
python llama.cpp/convert_hf_to_gguf.py kev-4b-merged-f32 --outtype q8_0 \
  --outfile kev-4b-backbone-q8_0.gguf     # llama.cpp @ 68d9053a
python tools/kev/pack_gguf.py kev-4b-backbone-q8_0.gguf kev-4b/ kev-4b-q8_0.gguf \
  --adapter jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101 \
  --base Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b
```

The inputs are `Qwen/Qwen3.5-4B-Base@1001bb4d` and
`jaredpalmer/kev-4b@139fdd94` from Hugging Face. `tools/kev/quality.sh`
scores the result with the reference benchmark through a running server,
and `tools/kev/reference_oracle.py` writes the golden file `kev_parity_test`
reads.

## How it was built

1. A baseline on the existing Qwen executor: state prefill, then each
   question row after restoring the recurrent state, hidden states captured
   for the readout. It is slow but simple, and it is the in-repo GPU oracle.
2. The packed pass, kernel by kernel, each checked against the baseline.
3. Serving, scheduling and the router, then measurements with Flash-Next.
