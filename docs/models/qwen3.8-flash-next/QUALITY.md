# Qwen3.8 Flash-Next quality

**All 63 measured tg128 requests match fresh AR completions:** 21 AR,
21 mixed MTP and 21 repetitive MTP at C1/2/4/6/8. Unsloth UD-Q4_K_XL target,
shared Q8_0 MTP; [identities](artifacts/model-identities.json).
These are consistency checks, not original unquantized-model or GGUF-conversion
qualification. Latest measurements: September 20–23, 2026.

| Check | Result |
| --- | --- |
| MTP versus scalar CPU formulas, eight text/image states | Fusion/attention relative RMS <0.0008 (limit 0.002); full-width normalization, split projections, recursive carry and full Q8 head checked |
| Batched MTP/AR, C2/C4/C6/C8 | Logits, tokens, acceptance, RNG and every 1–8-token rollback prefix match isolated execution (measured before wide batches used tiled projections) |
| Wide batched rows | Above a per-projection row count (16–160 rows) shared projections use tiled GEMMs: Q8 routes keep the per-32 int8 activation quantization (same FP64 error as the GEMVs, 2–7% of RMS on outlier-heavy inputs), F16 routes are 30–50× closer to FP64. Rows are no longer bit-identical to single-session decoding; the session test checks greedy MTP tokens and frontier logits within 5% of their range, and narrow AR batches stay exact |
| Sampling | 23 AR/MTP configurations; shared FP64 target filtering/CDF, p/q acceptance and residual correction pass |
| Prefill and cache | Full logits match across tested chunk boundaries, short tails and restored state through 4096 tokens |
| Scalar versus bulk prefill, 2176 tokens | Same top-1; logit RMSE 0.18, not bit-identical |
| Serving | Cancellation, three-turn continuation, reasoning/tool history, concurrent image/text isolation and disk restart pass |

Sampled MTP can consume different RNG draws from AR. Seeded replay requires
the same build, request budget, capacity and sampling configuration; live cost
timings only steer greedy decoding. Draft sampling uses the full Q8 head's
top 64 logits; upstream draft-sampler equivalence is not claimed.

## Serving options and lossy flags

September 26–27, 2026, `tools/qwen-flash/qa` against 2 × 131072 AR servers:
teacher-forced top-20 logprobs on two public-domain books at 2K/16K/64K/120K
depth (KL, top-1 agreement, forced-text perplexity), 12 needles in 8K–120K
haystacks, and 54 tasks (20 code with executed asserts, 20 exact-answer math,
8 tool calls, 6 JSON). Gates: mean KL ≤0.005 pass / ≤0.02 warn; top-1 ≥98% / ≥95%.

| Configuration | Mean KL | Top-1 | Forced-text PPL | Needles | Tasks |
| --- | --- | --- | --- | --- | --- |
| This branch, default (f16 KV, exact routing) vs `main` | 0.003–0.005 | 97.7–99.1% | within ±1.2% | 12/12 | 54/54 |
| `--kv-cache tq4` vs f16 | 0.007–0.015 | 95.4–99.6% | −1.6 to +2.4% | 12/12 | 54/54 |
| `--kv-cache tq3` vs f16 | 0.011–0.029 | 96.2–98.8% | −0.4 to +2.1% | 12/12 | 54/54 |
| UD-IQ4_XS vs UD-Q4_K_XL | 0.020–0.057 | 94.1–98.8% | −0.4 to +5.7% | 12/12 | 54/54 |

The default path is not bit-identical to `main`: the device-planned expert
kernels and long-context decode attention change rounding (operator checks put
both closer to FP64 than before), and this model's routing amplifies last-bit
changes. `tq4`/`tq3` and UD-IQ4_XS are opt-in. `--moe-routing` policies are
unqualified and gave no speed gain.

Batched decoding: narrow AR batches (C2–C8) are bit-exact against serial
decoding, including the MTP draft state. Wider batches and mixed prefill+decode
steps use tiled or shared routes and are checked by greedy-token agreement plus
KL(serial ‖ batched) of the next-token distribution: ≤0.1 per step, ≤0.25 for
32-session MTP cycles (up to 256 verified rows, eight replayed tokens). The
earlier bound (max |Δlogit| within 5% of the largest logit) sat inside the
noise between the two routes. Against the f32 CPU reference on the same weights
(September 30, 2026), 32-row AR frontier rows were 0.017–0.048 KL (serial) and
0.015–0.049 (batched), 0.008–0.015 from each other. Of the six most divergent
wide rows, the batched row was closer to the reference in five; the worst MTP
cycle row (0.21 between routes) was the exception at 0.35 versus 0.08.

## Speed-campaign options

October 25–26, 2026, UD-Q4_K_XL, `tq4` KV, in five integrated measurement
windows (`w1`–`w5`); retained outputs are in
[artifacts/campaign](artifacts/campaign). [Speeds](BENCHMARKS.md#speed-campaign-stack-october-2026) ·
[rejected ideas](EXPERIMENTS.md#speed-campaign-october-2026).

### The default path is exact

The campaign's exact wins are on by default; every lossy option is off and
enters the disk-cache identity. Evidence for the default, final binary
(`20261026T203743Z-w1-exact`):

| Check | Result |
| --- | --- |
| Op tests (attention incl. the select-window regression, GDN, HC mix, MoE routing/epilogue, MTP, projections, ...) | all 20 binaries pass |
| `snapshot_test` | passes |
| `--validate-prefill`, base vs integrated default | identical (rmse 0.29, cosine 1.00, max error 1.79) |
| 28 requests (short, 700/3000/9000-token prompts, a 6-way batch), token ids and logprobs, base vs integrated default | 28/28 identical |
| MTP `--mtp-policy adaptive` + 32K draft vocabulary vs AR | 22/22 token ids identical |
| `--prefill-capacity 8192 --prefill-phase-rows 2048` vs base | 23/28: single sessions identical, the packed 6-way prefill batch differs in 5/5 (the tile configuration follows the packed row count) |

The macro batch is therefore exact per session but not bit-exact for packed
batches, so it stays an option.

`session_test` (full run, final tree, `20261027T003131Z-w1j-session`): passes twice,
rc 0. Its MTP statistics checks failed on the base tree as well: a greedy
request's draft widths come from the model-wide timed controller (an online
step-cost fit on measured wall time plus a cycle counter), not from per-session
state, so a replay's widths, drafted and accepted counts legitimately differ
while the tokens stay exact. The test now asserts widths and counts only for
sampled requests, whose plan depends on committed history alone (seeded replay
and the fresh-load bitwise replays pass in all four sampler configurations at
depth 16 and 4096); a batched sampled MTP request is checked for progress and
budget because the base tree's shared batch routes are not bit-identical to
serial for it either, and the mixed-step MTP decoder KL uses twice the
documented wide-cycle bound (see [Experiments](EXPERIMENTS.md#integration-findings)).

### Lossy options: measured quality

Two teacher-forced ladders per arm against a same-window base run of the same
binary family: public-domain books (`tools/qwen-flash/ladder.py`) and
coding text (`tools/qwen-flash/code_ladder.py`, 24 documents,
2048-token prompts, 128 forced tokens). "Decode" rows teacher-force tokens
through decode; "prefill rows" are the logits prefill itself emits at the
prompt's positions. The accept envelope for a lossy option is decode KL ≤ 0.057,
top-1 ≥ 94%, PPL delta ≤ +2% against UD-Q4_K_XL. The integration added a
prefill-row budget: KL ≤ 0.1 and top-1 ≥ 90%, because the first token after
a long prompt and every MTP draft start from prefill rows.

| Arm | Books decode KL / top-1 / PPL | Code decode | Books prefill rows | Code prefill rows |
| --- | --- | --- | --- | --- |
| `--dense-prefill w4a4` | 0.0113 / 98.0% / −0.10% | 0.0054 / 98.4% / +0.40% | 0.060 / 92.6% | 0.043 / 95.8% |
| `--routed-w4 64` + macro | 0.0359 / 95.3% / −0.49% | 0.0202 / 95.3% / −0.30% | **0.41 / 74.1%** | **0.30 / 79.2%** |
| `--gdn-state f16` | 0.0068 / 97.5% / −0.23% | 0.0033 / 97.9% / −0.24% | 0.022 / 100% | 0.010 / 100% |
| `--prefill-residual f16` | 0.0091 / 97.7% / −0.78% | 0.0059 / 97.2% / +0.03% | 0.035 / 88.9% | 0.048 / 87.5% |
| `--gguf-decode-override` twins | 0.0266 / 96.6% / +0.15% | 0.0061 / 96.5% / +0.27% | 0.0013 / 100% | 0.0006 / 95.8% |
| `--experts-used 8 --router-norm-k 10` | 0.0380 / 95.3% / −0.65% | 0.0201 / 95.5% / −1.44% | 0.152 / 92.6% | 0.164 / 83.3% |
| **dense + macro + gdn f16 + twins (recommended)** | 0.0314 / 96.2% / +1.14% | 0.0104 / 95.9% / +0.42% | 0.050 / 92.6% | 0.066 / 95.8% |
| dense + routed-w4 + macro + gdn + F16 residual + twins | 0.0518 / 94.8% / +1.11% | 0.0246 / 94.3% / +0.86% | 0.24 / 96.3% | 0.30 / 79.2% |
| the same + top-8 | 0.0744 / 94.6% / +2.04% **fail** | 0.0313 / 93.5% **fail** | 0.42 / 77.8% | 0.47 / 70.8% |
| dense + macro + gdn + F16 residual + twins | 0.0329 / 95.8% / +1.40% | 0.0114 / 95.6% / +0.59% | 0.068 / **85.2%** | 0.093 / 91.7% |
| macro + gdn + F16 residual + twins (no int4 dense) | 0.0296 / 95.9% / +0.97% | 0.0088 / 96.2% / −0.05% | 0.036 / 88.9% | 0.037 / 91.7% |

What drives the prefill-row error: `--routed-w4` (KL 0.30–0.41), then top-8
(0.15), int4 dense prefill (0.04–0.06) and the F16 residual (0.035–0.048, but
top-1 87–89%). Decode-only options (twins, `--gdn-state`) leave prefill rows
nearly untouched. Single-seed ladders carry about ±0.01 KL of build-to-build noise.

The recommended stack is the arm that passes both gates:
`--dense-prefill w4a4 --dense-prefill-weights DIR --gdn-state f16
--gguf-decode-override q4k-gptq2-h5.gguf` (+ `--prefill-capacity 8192
--prefill-phase-rows 2048`, exact per session). `--routed-w4` and
`--prefill-residual f16` each fail the prefill-row budget; top-k stays out
(below).

Per-option scope notes:

- `--dense-prefill w4a4` applies to prompt batches of ≥ 2048 rows only; a
  256-row threshold failed (PPL +2.12%). With DIR it uses offline GPTQ S4G128
  weights (Hadamard-128 rotation); without, RTN of the Q8_0 weights at load.
- `--gguf-decode-override` adds decode twins (GPTQ Q4_K dense projections, Q5_K
  head, +1.9 GB) read by batches of ≤ `--decode-twin-rows` (default 4) rows;
  prefill, wider decode batches and the adapters' arithmetic are untouched.
  Quality is measured at C1; a batch crossing the row limit changes arithmetic
  between steps.
- `--gdn-state f16` also halves the recurrent state in every KV-pool slot; 64K
  prefill KL 0.0093 and a 32K+2048 drift check KL 0.0039. `bf16`/`q8` were
  worse (`q8` fails 64K prefill, KL 0.21).

### Agent-task quality: top-k and adapters

Quality ladders cannot see task behavior, so the same 96 pinned coding tasks
(32 agents, one 47-minute window per arm, three plain-UD replicates as the noise
floor) and GSM8K-250 ran against reduced-k routing and both adapters
(TaskQuality lane; raw JSON in `expert-prune-data/taskquality/`, reports
`docs-tq/tq-report-full.md` and `tq-pooled.md`).

| Arm | Finished / started | Accuracy on finished | Turns / completion tok per task |
| --- | --- | --- | --- |
| UD baseline (3 runs pooled) | 87% | 91.6% | 9.9 / 821 |
| top-8 (2 runs) | 70% (p < 0.0001) | 91.0% | 10.4 / 778 |
| top-8 + f16 GDN + affinity:6:1 | 75% | 88.9% | 11.7 / 980 |
| top-7 | 65% | 77.6% (p = 0.004) | 12.8 / 1036 |
| healed top-5 | 66% | 89.8% | 11.4 / 1001 |
| top-5 | 62% | 84.6% | 13.2 / 1341 |

The swarm is prefill- and tool-bound. Top-5 decodes faster (server ITL p50
294 vs 375–479 ms) but needs about three more turns and 370 more tokens per
task, which cancels the gain; correct tasks per hour fell 105–109 (top-8) and
78–93 (top-5 variants) against 127–145 for plain UD. GSM8K stayed within noise in
every arm (94.4–96.8%). Verdict: top-5, healed top-5 and top-7 do not ship;
top-8 gives no end-to-end gain on this workload and stays opt-in for
decode-bound use. Accuracy differences of 5% could not be resolved (each paired
comparison has a ±8 point interval); finish rate and turns per task are the
measurable effects.

| Adapter on the same swarm | Accuracy | Correct/h (steady) | GSM8K |
| --- | --- | --- | --- |
| plain UD (3 runs) | 88.5–94.8% | 127–145 (158–210) | 95.6–96.8% |
| huihui | 62.5% (25/40; McNemar p 0.012–0.039) | 48 (75) | 95.2% |
| Orca | 88.4% (76/86) | 135 (174) | 96.8% |

The huihui adapter loses real capability for coding agents: 6 of its 15
failures stop on turn 1 with a text-only reply and no tool call, and three more
hit the 40-turn limit; plain UD has no such failures. Orca is
indistinguishable from plain UD on both tasks and GSM8K. Agents should be served
with Orca, not huihui.

## Steering adapters

`--adapter` rows run the base kernels plus one ablation per block output, so
adapter-off serving keeps the base arithmetic bit for bit. The session test
loads a synthetic adapter and checks that mixed base/adapter prompt batches and
narrow AR batches equal standalone runs bit for bit, MTP cycles agree by greedy
tokens within the batch KL bound, and snapshots and shared KV pages restore
only into their own adapter; `moe_epilogue_ops_test` checks the kernels against
an FP64 reference. Equivalence with full abliterated weights is measured with
`tools/qwen-flash/qa/adapter_eval.py` and the `logits` suite of `qa.py`
(teacher-forced top-20 KL, reference first).

October 3, 2026, UD-Q4_K_XL, AR, 24 sessions, tq4 KV, against
huihui-ai's abliterated UD-Q4_K_XL GGUF (revision 73e9eb7c) served by the same
build. Forced texts are the adapter's greedy answers to 16 prompts (8
refusal-prone); books are the `logits` suite's 2K/16K Moby Dick continuations
and eight concurrent 2K passages.

| Candidate vs huihui GGUF | Forced texts KL (first token) | Books KL mean |
| --- | --- | --- |
| Base model, no adapter | 0.102 (1.33) | 0.017–0.029 |
| Adapter, GGUF scope (`routed_expert_layers` 2, 4, 30, 46, 47) | 0.0035 (0.009) | 0.006–0.010 |
| Adapter, full scope (published BF16 weights) | 0.178 (0.74) | 0.125–0.190 |

The GGUF is not a faithful conversion: its Q5_1 routed down projections in 43
of 48 layers are byte-identical to the base GGUF, while the published BF16
weights ablate every routed expert (α = 1.955 measured on BF16 and on the GGUF's
Q8_0 expert layers 2, 4, 30, 46 and 47). Scoped to what the GGUF carries, the
adapter matches it well inside the quantization noise between two artifacts;
the full-scope adapter follows the BF16 model and costs 9–16% forced-text
perplexity on prose. Base requests on an adapter build stay bit-identical to
the base build (teacher-forced KL 0, identical single-request greedy outputs).
Only 24-request decode cohorts vary (5 of 24 outputs; two adapter-off runs
differ from each other just as much) and one run's eight concurrent passages
(KL 0.001): fused steps depend on request arrival.

### OrcaRouter abliteration (v2 adapter, all writers)

October 26, 2026, UD-Q4_K_XL, evidence in
[artifacts/orca-adapter-proofs.json](artifacts/orca-adapter-proofs.json).
`qwen38-flash-next-orca-abliteration.json` removes r at α = 1 from block
outputs, embedding rows, the PLE value and the MTP block. Orca's BF16 repository
is gated, so the direction and strength were checked against the vmlinux Q8_0
GGUF and the jcbtc MTP sidecar (`artifacts/orca-direction-check.json`).

**GPU logits against float32 oracles.** `gpu_probe` prefilled a fixed 41-token
prompt on the GPU (base session, then adapter session); `reference_probe` ran
the same prompt in float32 on the dequantized weights with the adapter folded
into W. KL is the full-vocabulary KL(oracle ‖ GPU) over the 41 positions.

| Oracle → GPU run | KL mean | KL max | top-1 |
| --- | --- | --- | --- |
| base → GPU base (quantization floor) | 0.0051 | 0.029 | 40/41 |
| Orca fold → GPU Orca | 0.0097 | 0.088 | 37/41 |
| Orca blocks-only fold → GPU blocks-only (v1 scope) | 0.0057 | 0.062 | 41/41 |
| Orca fold → GPU base (no adapter) | 0.303 | 3.49 | 30/41 |
| base → GPU Orca | 0.316 | 3.42 | 33/41 |

The adapter's effect on the logits is 30–60 times the GPU-versus-float32 error.
Orca's mean sits 1.9 times above the floor, almost entirely at positions 15
and 16 (KL 0.084 and 0.088 against 0.003 for base); without them Orca is
0.0058 and the floor 0.0051. The paired per-position difference to the floor
(0.0046) has a bootstrap 95% interval of −0.0008 to 0.0112, which includes zero,
and the four top-1 misses are near ties (oracle top-1/top-2 logit gaps 0.00, 0.06,
0.24, 0.28; the base run misses one position too). The GPU base rows of the two probe
processes are bit-identical. Both adapter variants reproduce the folded model
to quantization noise; this is a 41-position check, not a perplexity measurement.

**Cache isolation, never-used prompts, RAM and disk** (`cache_fresh.py`,
production flags with `--cache-disk`, ids base, huihui, orca, orca-blocks). Every prompt is
time- and urandom-seeded and was asserted uncached on its first request. Seed
server: for each id, the other three ids asked its prompt (12 requests, all
`cached_tokens` 0), then every id repeated it (16 own-entry hits, all full-prompt).
The server was then restarted on the same disk directory (empty RAM): 21 cross-id
requests on disk-seeded prompts and on system blocks seeded by one id (three conversations each) all missed; the seven own-id
controls all restored from disk (`cache=disk`, 1405–1437 tokens, 811–828 for the
system block). Zero violations.

**MTP acceptance** (`mtp_accept.py`, 24 prompts × 192 greedy tokens, the same prompts
for every id): base 0.658, huihui 0.647, Orca 0.657, Orca without its MTP
writers 0.656; the paired per-prompt difference Orca − Orca without MTP
writers is −0.005 (bootstrap 95% −0.028 to +0.020). The 0.43/0.46/0.50 of the
first window compared different synthetic prompts per id and is not evidence.

## Vision

**One Gufo encoder comparison fails:** relative L2 **6.47%** versus official
Transformers BF16, above the **5%** limit, on a 1024×1024 synthetic texture.
Both use the same converted GGUF weights. Against FP32, Gufo and Transformers
BF16 differ by **7.83% / 8.10%**, respectively. This is numerical drift, not an
image-answer score; **llama.cpp was not tested**. Optimizations retain native
embedding bytes but do not resolve this gap. [Evidence](artifacts/vision-parity.json).

## Reproduce

Tests live in [`tests/models/qwen38_flash_next`](../../../tests/models/qwen38_flash_next).
Use `--batch-only`, `--sampling-only` or `--prefill-only` on the session test;
the snapshot test covers persistent image/text state. For independent MTP checks:

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$MODEL" --mtp-model "$MTP" --mtp-audit
```

The [vLLM](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py)
formulas supply independent predictor checks; pinned Transformers ignores MTP
weights. [Vision reproduction](../qwen3.8-27b/QUALITY.md#vision).

## Benchmark method

September 22–23, 2026; one warmed sample per point, greedy, thinking off.
Single-user uses pp2048/tg128; MTP pp is the maximum across mixed/repetitive
workloads. C1/2/4/6/8 use the same d0 prompts; every session prefills before
measured tg128, with at most four prompt-tail tokens reevaluated. Rates sum
individual decode rates. Gufo d0/C1 agree within 0.4% with matching drafts/output.
AR reference is llama.cpp b11069; MTP uses pinned `6fcaa16f`.
Loading: cold files, C1/MTP/capacity 262144. Memory: C1/AR/capacity 133121,
peak global HIP allocation including idle memory. Full commands, counts and
identities remain in [artifacts](artifacts/bench.json) and the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).
