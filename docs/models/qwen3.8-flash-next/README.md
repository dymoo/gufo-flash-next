# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
**UD-IQ4_XS** also loads (IQ3_S/IQ4_XS gate/up, IQ4_NL down experts, Q6_K head
converted to Q8_0 at load): about 18 GiB less GPU memory, with a measured quality
cost (see [Quality](QUALITY.md#serving-options-and-lossy-flags)). UD-Q3_K_XL kernels
exist but are unqualified. Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified. The NVIDIA FP4-Local
converter exposes decoded PLE shapes without loading payloads and decodes each shard only when
GGUF writes its chunk.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. Sampled
requests use deterministic acceptance/cost control for seeded replay; every
greedy request (single or batched) takes its draft width from the model-wide
timed controller (a step-cost model fitted on measured wall time), so only
greedy *statistics* (widths, drafted/accepted counts), never tokens, depend on
timing and on earlier steps. Each request keeps private caches, rollback and
RNG. See [MTP qualification](QUALITY.md). `--mtp-policy adaptive` with
`--mtp-draft-vocab draft-vocab-ranked.txt` (32768 ranked token ids, one per
line; `--mtp-draft-vocab-size`) replaces that controller for greedy single
requests by the combined policy: a compact draft head, a calibrated confidence
gate and longest-match prompt lookup. It is exact (greedy tokens equal AR) and
measured +4% to +46% over the default controller by workload
([Benchmarks](BENCHMARKS.md#single-user-mtp-adaptive-policy)). On the lossy
stack, MTP still helps (+23% prose, +47% code over the stack's AR) but is no
longer bit-identical to AR, because `--gdn-state f16` rounds per verified chain.
The draft vocabulary is ranked by token frequency in a corpus; the list used in
the measurements is built with `tests/models/qwen38_flash_next/spec_vocab.cpp`
(held-out coverage 91–99% at 32K tokens) and lives beside the MTP sidecar.

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144; YaRN extension
is unsupported. KV and indexer pages commit lazily from one pool as sessions grow
(HIP virtual memory), so a slot costs the context it actually holds; admission
claims each session's fixed state plus the shared pool, and only a request that
cannot grow fails when the pool is full. Idle sessions page out to host RAM and
then to `--cache-disk` files when the pool is under pressure.

`--moe-routing` (serve, chat, prompt) trades routing fidelity for fewer expert
bytes per decode step; prefill always routes exactly. `exact` is the default.
`affinity:KEEP:MARGIN` keeps each row's top KEEP experts and fills the other
slots with experts another row keeps when they score within MARGIN logits;
`mass:KEEP:TARGET` drops slots once the exact top-k mass reaches TARGET;
`budget:KEEP:EXPERTS` caps distinct experts per layer and step. The batch-aware
policies make a request's output depend on the requests decoded with it, so
seeded replay then also requires the same co-scheduling. They are unqualified
and, on the device-planned expert kernels, measured no throughput gain
(−8% to +6% at 32–64 sessions); keep `exact`.

## Serving many agents

Coding-agent traffic (long shared system prompts, tool-output appends, about
95% prefix-cache hits per turn) is prefill-bound. The configuration measured in
[Benchmarks](BENCHMARKS.md#agent-serving) is:

```sh
./result/bin/gufo serve --sessions 24 llm --model "$MODEL" --context 131072 \
  --kv-cache tq4 --host-memory-budget 8589934592 \
  --cache-disk /var/cache/gufo --cache-disk-bytes 214748364800
```

- `--sessions` up to 64: decode batches share projections and stream each
  active expert once per step; attention runs one launch per layer.
- `--kv-cache f16|tq4|tq3`: `tq4`/`tq3` store rotated 4/3-bit keys and values
  (3.9×/5.2× smaller than F16; a 131K slot's KV drops from 3.0 to 0.76/0.57 GiB).
  Lossy and opt-in; `f16` is the default.
- `--host-memory-budget BYTES` caps the server's host RAM (half retained
  snapshots, a quarter idle-session paging, a quarter disk staging); set it on
  shared hosts, where the container's view of free memory is misleading.
- `--cache-disk` keeps prefix snapshots on NVMe; system prompts shared by several
  conversations are pinned there and in RAM (`--cache-pin-*`) and survive restarts.
- Fused prefill+decode steps pack several requests' prompt chunks; their size
  adapts to `--max-itl-ms` (default 1500 ms). See
  [SplitFuse scheduling](../../SERVER.md#hip-execution).
  The indexer's block-score scratch holds 512 rows at the model's full 262144
  context (1024 rows at a 131072-token session), fewer than a fused step packs
  (up to the 2048-token prefill capacity): batched block selection scores the
  packed rows in windows of what the scratch holds
  (`SelectBlocksBatch`'s `score_rows`; guarded by `attention_ops_test`).
- MTP helps single users and repetitive text but not this workload at high
  concurrency; serve agents without `--speculative`.
- `--metrics-address HOST:PORT` exposes Prometheus metrics on a separate,
  unauthenticated listener ([Metrics](../../SERVER.md#metrics)).
- `service_tier: "flex"` requests (background and batch work) run only on
  idle compute, at most `--flex-sessions` (default 2) at once, and get an
  immediate 429 otherwise; see the [router contract](../../ROUTER.md).

## Speed options

**The default path is exact.** The campaign's bit-identical wins are on without a
flag (prefetching small decode GEMMs, fused hyper-connection combine/mix and
GatedDeltaNet epilogues that hand Q8_1 activations to the next projection, the
tq4 KV-store prefill epilogue, the multi-session block-selection fix below):
+4% pp2048 and +6.7% tg128 on one session, with unchanged tokens and logprobs
([evidence](QUALITY.md#the-default-path-is-exact), [numbers](BENCHMARKS.md#speed-campaign-stack-october-2026)).

Every option below is off by default and enters the disk-cache identity when set,
so snapshots never cross numerics. All work in `gufo serve ... llm`; the
single-user ones also in `gufo bench`. Measured quality and exact/lossy status
of each are in [Quality](QUALITY.md#speed-campaign-options); the accept envelope
for a lossy option is decode KL <= 0.057, top-1 >= 94%, PPL <= +2% against
UD-Q4_K_XL, plus prefill-row KL <= 0.1 with top-1 >= 90%.

| Option | Status | Effect |
| --- | --- | --- |
| `--prefill-capacity N`, `--prefill-phase-rows N` | exact per session; a packed multi-request prompt batch is not bit-identical (tile choice follows the row count) | prompt batch per forward pass (`8192`/`2048`: each routed expert is streamed once for 4x the rows) |
| `--dense-prefill w4a4` (+ `--dense-prefill-weights DIR`) | lossy (decode KL 0.011 books / 0.005 code) | int4 W4A4 dense projections for prompt batches of >= 2048 rows; `DIR` holds offline GPTQ S4G128 weights, else load-time RTN |
| `--gdn-state f16\|bf16` | lossy (f16: KL 0.007 / 0.003) | half-size GatedDeltaNet recurrent state (decode bandwidth, KV pool); `f32` is the default |
| `--gguf-decode-override PATH`, `--decode-twin-rows N` | lossy, decode only (KL 0.027 / 0.006) | GPTQ Q4_K dense projections and Q5_K head uploaded as decode twins beside the Q8_0 tensors (+1.9 GB), read by decode batches of <= N rows (default 4); prefill unchanged |
| `--mtp-policy adaptive`, `--mtp-draft-vocab FILE`, `--mtp-draft-vocab-size N` | exact (greedy tokens equal AR; sampled requests unchanged) | combined greedy C1 policy: 32K compact draft head, calibrated confidence gate, longest-match prompt lookup |
| `--prefill-residual f16` | lossy; fails the prefill-row budget (top-1 87-89%) | F16 hyper-connection residual between wide prefill combines |
| `--routed-prefill-a4 G`, `--routed-down-w4`, `--routed-w4 G` | lossy; `--routed-w4` fails the prefill-row budget (KL 0.30-0.41) | iu4 routed experts in prompt batches (A4 groups of G = 64/128/256; 4-bit down experts; symmetric int4 experts, decode too) |
| `--gguf-override PATH` | lossy | GGUF whose tensors replace same-named tensors (requantized dense projections; costs pp512 -13% through the MMQ route, so prefer the decode twins) |
| `--experts-used K`, `--router-norm-k N` | lossy; **does not ship for agents** (task level, see Quality) | route each token to its top K experts, weights normalized over the top N probabilities |

### Production configuration

The base model id stays exact; lossy numerics belong to a separately named
model (see [Model ids](#model-ids-variants-and-the-flex-tier)). The exact
production line, with the abliterated variant beside the base model (tested
together: base bit-identical with the adapter loaded, cache isolation, +2%
decode overhead; [Benchmarks](BENCHMARKS.md#adapters-beside-the-base-model)):

```sh
./result/bin/gufo serve --sessions 24 llm --model "$MODEL" --context 131072 \
  --served-model-name workhorse \
  --kv-cache tq4 --host-memory-budget 8589934592 \
  --cache-disk /var/cache/gufo --cache-disk-bytes 214748364800 \
  --adapter workhorse-abliterated=/path/to/orca-abliteration.json
```

`--kv-cache tq4` is the one lossy flag kept on this line (KL 0.007–0.015, 12/12
needles, 54/54 tasks; [Quality](QUALITY.md#serving-options-and-lossy-flags)):
it makes 131K-token agent sessions fit. Drop it for a bit-exact server.

**Lossy stack (process-wide).** Adding
`--dense-prefill w4a4 --dense-prefill-weights "$S4G128_DIR" --gdn-state f16
--gguf-decode-override q4k-gptq2-h5.gguf` passes both quality gates (decode KL
0.031 / 0.010) and measured pp2048 +17%, pp8192 +24% and tg128 +37% on one
session, but on the agent load only decode improves (+6–8%, C8–C32 +10–14%);
prefill and completed turns do not change
([Benchmarks](BENCHMARKS.md#agent-serving-on-the-stack)). These flags apply to
every id the process serves. The `-fast` model below is the per-request
alternative: lossy routing on its own model id while the base id stays exact.

Optional: `--prefill-capacity 8192 --prefill-phase-rows 2048` (exact per session, +5%
pp8192, about 4 GB more scratch, prompt steps about 4x longer; not bit-exact for
packed multi-request prompt batches). It measured no gain on the agent load.
**Upgrading:** `--prefill-phase-rows` was spelled `--prefill-chunk` until this
release, which shadowed the scheduler's `--prefill-chunk` (fused-step prompt
rows, [SERVER.md](../../SERVER.md)) so the scheduler option could not be set.
A configuration that passed `--prefill-chunk 2048` for the macro batch must
switch to `--prefill-phase-rows 2048`; `--prefill-chunk` now sets the
scheduler's rows. Offline artifacts: `s4g128` is a GPTQ S4G128 export indexed by
`tools/qwen-flash/s4_index.py`, and `q4k-gptq2-h5.gguf` a GPTQ
Q4_K decode override; neither is downloadable yet.

## Model ids: variants and the flex tier

One process serves several model ids from the same weights, sessions and
batches, listed by `GET /v1/models`:

| Id | Kind | Numerics |
| --- | --- | --- |
| `workhorse` (`--served-model-name`) | model | exact (plus `--kv-cache tq4` if set) |
| `workhorse-abliterated` (`--adapter NAME=PATH`) | model | the Orca steering adapter on the same weights ([Steering adapters](#steering-adapters)) |
| `workhorse-fast` (`--fast-experts 7`) | model, **experimental** | lossy decode: each decode row keeps its top 7 routed experts with the weights they had among the top 10; prompts stay exact |
| `<any id>-flex` | service tier | the id it suffixes, unchanged; admitted only on idle compute ([Router](../../ROUTER.md#service-tiers)) |

A variant is a **different model**: it has its own outputs, quality numbers and
cache identity, so it never shares a cache entry with another variant. The flex
tier is **scheduling only**: same weights, same arithmetic, same cache entries
as the id it suffixes. That is why fast mode is a model suffix and not a
service tier. It is not "the same model, faster", and it must combine with the
tier: `workhorse-abliterated-fast-flex` is the abliterated fast model at flex.
No served id (the model id or an adapter name) may end in `-flex`, which is
reserved for the alias, or with `--fast-experts` in `-fast`: a literal id such
as `workhorse-fast-flex` would otherwise read as another model's alias. The
server refuses to start with one.

**Fast mode** (`--fast-experts K`) gives every served id a `-fast` model:
`workhorse-fast`, `workhorse-abliterated-fast` and their flex aliases. Its
decode rows keep only their top K routed experts; the dropped slots repeat the
row's top expert at weight 0 (the encoding `--moe-routing` policies use), so
each step streams fewer distinct experts and every expert route runs
unchanged. Prompt rows always route exactly, so a fast request's prompt KV is
the same arithmetic as the base model's, but fast sessions keep their own
cache identity because their generated tokens' KV is not. Fast and
full-routing requests share batches; a fast row speeds up a step only as much
as it shrinks the batch's expert union, so the gain is largest when most rows
are fast.

Measured inputs (the server-wide `--experts-used 7 --router-norm-k 10`
equivalent): books KL 0.061 / top-1 93.4% and +14% C8 / +8% C64 decode in a
pure batch, but coding agents finished fewer tasks with it (65% vs 87%), so
it is for simpler work
([Quality](QUALITY.md#agent-task-quality-top-k-and-adapters)).
**Status:** compiled for gfx1151 but not yet measured on the per-request
path; until a GPU quality ladder and mixed-batch A/B run, treat its numbers as
the server-wide measurements above. Per-request decode twins and F16 GDN state
for the fast model are not built: those options remain process-wide.

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Steering adapters

One process can serve weight variants that differ from the loaded model only
by a rank-1 directional ablation W' = W − α·r·(rᵀW) of the matrices that
write the residual streams. The fork's abliterated variant is OrcaRouter's
Qwen3.8-Flash-Next-Uncensored direction (every writer), served from the base
weights:

```sh
HF_TOKEN=hf_... python3 tools/qwen-flash/adapters/orca_adapter.py --out adapters
./result/bin/gufo serve llm --model "$MODEL" --served-model-name workhorse \
  --adapter workhorse-abliterated=adapters/orca-abliteration.json
```

`orca_adapter.py` rebuilds the adapter from public weights: it range-reads a
few residual writers from Qwen's BF16 release and from OrcaRouter's (gated,
granted on request; the token reads it), recovers the edit direction as their
difference's top singular vector, checks that the edit is rank 1 with strength
1 on every compared writer, and writes the JSON and float32 direction.

On 96 pinned coding tasks with 32 agents the Orca variant matched the base
model (90% of tasks finished, 88.4% correct). huihui-ai's abliterated
Flash-Next, which ablates only block outputs, finished 56% at 62.5% accuracy
and is not served
([Quality](QUALITY.md#agent-task-quality-top-k-and-adapters)).

`--adapter NAME=PATH` (repeatable) reads `gufo steering adapter v1` or `v2`
JSON (`hidden_size`, `alpha`, `direction_file`: a regular file, relative to the
JSON, of exactly 2560 little-endian float32, unit norm) and serves NAME from
the same weights, sessions and batches: each
output y of a projected writer becomes y − α·(r·y)·r for adapter requests.
v1 projects the main-layer block outputs before they join the residual
streams (attention output, routed + shared expert MoE output); the MTP draft
layer, embeddings and head are unchanged. v2 lists its writers in `writers`
(`blocks`, required, plus any of `embedding`: the token embedding rows of the
trunk and of the MTP draft input, before image rows replace them; `ple`: the
PLE value projection; `mtp`: the MTP draft block's attention and MoE outputs),
so draft proposals follow the adapter's own target and MTP verification
stays lossless against it. The head, router, hyper-connection mixers and
input projections are never projected; DSpark drafts are unchanged.
Optional `routed_expert_layers` limits the MoE ablation to the listed layers'
whole output and ablates only the gated shared expert elsewhere; it exists to
reproduce artifacts that abliterated only some routed experts (the huihui GGUF,
see Quality) and is omitted for the published weights.
`tools/qwen-flash/abliteration_check.py` measures a published variant's
direction, strength and rank-1 structure per writer class from range reads of
its weights.
`/v1/models` lists the base id, then each adapter. Chat Completions, Responses
and Anthropic Messages pick the variant by `model`; raw completions serve the
base only. Base requests run the unmodified kernels, and adapter and base rows
share every batch with their standalone arithmetic. The adapter (its name and
a digest of α, r and, beyond v1's block outputs, its writers) is part of every
cache key: live continuations, RAM and disk snapshots, shared-prefix pins and
shared KV pages never cross variants.
Equivalence with the full abliterated weights is in [Quality](QUALITY.md#steering-adapters).

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
`moe_routing_eval.py` reports expert-union size and routing distortion per
`--moe-routing` policy on synthetic logits or on router logits dumped with
`GUFO_FLASH_NEXT_ROUTER_DUMP=<file>` (debug: synchronizes every layer).
`tools/qwen-flash/qa` holds the HTTP QA suite, `agent_sim.py` (randomly arriving
multi-turn coding agents with tool-output appends) and `throughput.py` (per-second
prefill/decode rates from server logs). No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.
