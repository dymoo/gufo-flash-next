# MiMo-V2.6-Flash (spike)

A spike runtime (`mimo2`) for MiMo-V2.6-Flash-RL (309B total, 15B active MoE)
in its 2.76 bpw IQ2_M-class GGUF, run in September 2026 to decide whether it
should replace Qwen3.8 Flash-Next as the serving model on one 128 GiB Strix
Halo. **Verdict: it should not**; Flash-Next stays the serving model. The
runtime is kept for reference, hard-wired to that one artifact, and has no
tests. `GUFO_MIMO_KV_POOL_GIB` sizes its KV pool and `GUFO_MIMO_PROFILE` prints
per-stage GPU time.

## Correctness

`tools/mimo/compare.py` records llama.cpp's greedy top-20 logprobs for fixed
prompts (8 short chat/code prompts plus 2K, 8K and 14K-token passages) and
compares Gufo's per-position KL while the histories agree, top-1 agreement and
the shared greedy prefix. Against llama.cpp HIP `86ea01d0`: mean KL 0.0072,
worst position 0.073, top-1 agreement 98.6%. Short prompts give KL
0.0001–0.006; long passages 0.012–0.024 (Gufo keeps q in F16 inside attention
and sums in a different order). A later build measured forced KL 0.0057 at
96.6% top-1, with determinism and passkey retrieval to 115K passing; a
prefill chunk-boundary check still failed.

## Results

SWE-bench Verified with the same agent harness, step limit and reasoning
settings for every run:

| Run | 50 everyday tasks | 45 hard tasks (1–4 h and > 4 h) |
| --- | ---: | ---: |
| Qwen3.8 Flash-Next on Gufo (tq4 KV, 24 sessions, effort medium) | 44/50 (88%) | – |
| GLM-5.3-Flash, publisher API (fp8, effort max) | 46/50 (92%) | 40/45 (89%) |
| MiMo-V2.6-Flash, publisher API (fp8, unquantised) | 49/50 (98%) | 44/45 (98%) |
| MiMo-V2.6-Flash IQ2_M on this Gufo spike | – | 6/8 completed before it was stopped (API on the same 8: 7/8) |

The 50-task set is close to saturated (every model resolves 88–98%, one task
is 2 points), so it measures cost per solved task rather than quality.

Serving speed on the Strix Halo (thinking off, greedy, cache bypassed), tok/s:

| | llama.cpp HIP (MiMo) | Gufo spike, first build | Gufo spike, best build |
| --- | ---: | ---: | ---: |
| Cold prefill 1K / 4K / 32K | 343 / 405 / – | 290 / 335 / 202 | 675 / 773–836 / 666–687 |
| Decode c1 / c2 / c4 | 22.9 / 33.7 / 43.0 | 19.0 / 24.5 / 31.4 | 23.4 / 39.7 / 48.1 |

## Why Flash-Next stays

- **Concurrency.** On the same box Flash-Next serves about 90 tok/s aggregate
  at c4 and about 241 tok/s at c24 over 24 sessions of 131K. MiMo IQ2_M
  reaches 48 tok/s at c4 over only four 128K sessions.
- **Quality per byte.** Unquantised MiMo beat GLM-5.3-Flash on both task sets
  with fewer steps, but what the IQ2_M quant costs stayed open (6/8 vs 7/8 is
  too few tasks), and the newer MOPD checkpoint in EXL3 would be the build to
  finish if a second local MoE is ever wanted beside Flash-Next.
