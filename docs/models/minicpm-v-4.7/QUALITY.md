# MiniCPM-V 4.7 quality

October 13–14, 2026, on `MiniCPM-V-4_7-35B-A3B-Q4K-Q51down-Q8.gguf` unless
noted. These check the port against llama.cpp on the same weights and the
served model on public tasks; they do not qualify the quantization against the
original checkpoint.

## Parity with llama.cpp

`tools/minicpm/parity.py`: llama.cpp (`86ea01d` text, `1b70350` images, f16 KV)
generates greedy answers with top-20 logprobs; Gufo is teacher-forced on the
same text (`forced_completion`) and every position is compared. Prompt token
counts match exactly in every case. Gates follow the Flash-Next QA: mean KL
≤0.005 pass / ≤0.02 warn; top-1 ≥98% / ≥95%.

| Gufo | Positions | Mean KL | p99 KL | Top-1 | Perplexity ref → Gufo | Greedy identical |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| f16 KV | 1,160 | 0.0017 | 0.039 | 98.5% | 1.051 → 1.054 | 6/12 |
| f16 KV, 1.5K/3K-token prompts | 1,416 | 0.0029 | 0.047 | 98.5% | 1.059 → 1.062 | 6/14 |
| tq4 KV (before the prefill fix) | 1,160 | 0.0062 | 0.124 | 98.8% | 1.051 → 1.059 | 6/12 |
| tq3 KV (before the prefill fix) | 1,160 | 0.0154 | 0.311 | 97.6% | 1.051 → 1.068 | 3/12 |
| f16 KV, image prompts | 423 | 0.0029 | 0.060 | 99.8% | – | 6/8 |
| tq4 KV, image prompts | 423 | 0.0158 | 0.273 | 97.4% | – | 2/8 |

Before the dense WMMA prefill dispatch fix the f16 rows measured 0.0020 and
0.0031 (top-1 99.0% and 98.8%); the fix moves prefill attention from the
per-token kernel to the WMMA kernel and is gated on these two rows.

Batched decoding (`qa.py batch`, tq4, 8–64 identical cohorts against single
sessions) is not bit-identical: KL 0.008, top-1 95.5–97.1%, perplexity within
1.5%, the QA warn band. Flash-Next's wide batches measure 0.015–0.049 against
its reference. `qa.py determinism` passes.

## DSpark speculative decoding

October 17, 2026, one session, `--speculative dspark` against the same binary
without it: 20 prompts (7 prose, 7 code, 6 agent transcripts, 256 tokens)
greedy 20/20 identical (tool-call IDs aside), and with temperature 0.8, top-p
0.95 and a fixed seed per prompt 20/20 identical. Six two-turn conversations
whose first answers stop at the length limit and whose second turns restore
the cached prefix: 12/12 identical. At four concurrent sessions batched
verification is not bit-identical, like batched decoding without
speculation (7/20 identical to one session).

`qwen38_flash_next_session_test --model MiniCPM.gguf --mtp-model none
--dspark-model dspark.gguf` passes: frontier logits and tokens bitwise equal
to plain decoding for greedy, sampled, filtered and penalized configurations
at 16 and 4,096 tokens of context and after a fresh load, partially accepted
blocks rolled back, a snapshot taken between draft cycles restoring the
drafter's K/V (identical continuation and acceptance), and serving replay.
On `qwen35moe` prompt kernels are not chunk-width invariant (KL up to 0.0001
between chunkings, with or without a drafter), so the test holds this family
to its batch KL bound where qwen4exp is exact.

## Tasks

Greedy, thinking off unless noted. HumanEval runs the completions against the
tests in a sandbox; DS4 is the 75-item GPQA Diamond / SuperGPQA / AIME 2025 set
pinned in `tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json`, thinking
on with a 16K-token cap. Runner: `tools/minicpm/evals.py`.

| | MiniCPM (Q4_K experts) | MiniCPM (Q8_0) | Flash-Next `UD-Q4_K_XL` |
| --- | ---: | ---: | ---: |
| HumanEval pass@1 | 154/164 (93.9%) | 154/164 | 157/164 (95.7%) |
| GSM8K | 1,262/1,319 (95.7%) | – | 1,267/1,319 (96.1%) |
| `qa.py tasks` (code, math, tool calls, JSON) | 54/54 | 54/54 | 54/54 |
| DS4, thinking | 60/75 (AIME 20, GPQA 18, SuperGPQA 22 of 25) | 60/75 | 55/75 (17, 18, 20) |

MiniCPM finished all 75 DS4 items in 22 minutes (5 at the token cap, 316K output
tokens). Flash-Next at its default effort finished 54 within the 30-minute
budget (50 correct); a rerun of the other 21 without a time limit added 5
correct, 1 wrong and 15 stopped at the 16K cap (441K output tokens overall).

## Quantization against the Q8 original

October 17, 2026: `qa.py run --suites logits --logit-depths 2048,4096,8192,16384`
(teacher-forced Moby-Dick continuations after 2K–16K-token prompts plus eight
concurrent Pride and Prejudice passages; top-20 logprobs, so the KL is the
suite's truncated estimate) on each file against Gufo serving OpenBMB's
`Q8_0`, both with tq4 KV and 8 sessions; 2,006 positions.

| Weights | Mean KL | p99 KL | Top-1 | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| `Q8_0` (reference) | – | – | – | 2.703 |
| Production requant | 0.092 | 1.16 | 91.9% | 2.766 |
| + `--output-tensor-type q6_k` | 0.092 | 1.21 | 92.0% | 2.767 |
| + dense projections and head Q6_K | 0.102 | 1.47 | 91.5% | 2.819 |
| OpenBMB `Q4_K_M` | 0.212 | 3.16 | 88.6% | 3.085 |

The production file through the K-quant branch is bit-identical to the
previous binary when prompt chunking is fixed (`--max-itl-ms 0`); with the
default adaptive chunking the concurrent cases can split differently.

## Images

48 rendered documents (invoices, bar charts, tables, order screens) with exact
answers (`tools/minicpm/vision_eval.py`): 48/48 on Gufo with f16 or tq4 KV, on
the `-4x` projector and on llama.cpp.
