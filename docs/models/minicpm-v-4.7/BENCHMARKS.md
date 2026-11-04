# MiniCPM-V 4.7 benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory, October 13–14, 2026.
`MiniCPM-V-4_7-35B-A3B-Q4K-Q51down-Q8.gguf` (see [Weights](README.md#weights)),
`tq4` KV, HTTP streaming, greedy, thinking off, unique prompts per request.
"Final" is this branch with 128-session decode and the dense WMMA prefill
dispatch fix; "64-session" is the same port before both. Flash-Next is the
production `UD-Q4_K_XL` (AR, Gufo `52ce2264`); llama.cpp is `86ea01d` on the
same GGUF with 64 slots, `-fa on -kvu`. Load generator: `tools/minicpm/load.py`.

## Closed loop

Each user sends its next request when the last finishes. Output tok/s summed
over users; TTFT and inter-token latency (ITL) are medians.

**Decode** (200-token prompt, 512-token answer)

| Users | Final | 64-session | Flash-Next | llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 55.1 | 55.8 | 25.1 | 49.7 |
| 8 | 201.9 | 199.1 | 92.9 | 146.5 |
| 32 | 369.6 | 352.3 | 168.0 | 179.0 |
| 64 | 480.2 | 467.2 | 217.4 | 191.5 |
| 128 | **583.1** (ITL 207 ms) | 475.5 (queued) | 216.4 (queued) | – |

**Chat** (1,000-token prompt, 300-token answer)

| Users | Final | 64-session | Flash-Next | llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 52.7 (TTFT 0.30 s) | 51.4 (0.37 s) | 23.7 (0.79 s) | 44.1 (0.89 s) |
| 8 | 156.6 | 148.7 | 68.4 | 103.0 |
| 32 | 247.5 | 234.5 | 105.8 | 104.3 |
| 64 | 295.6 (TTFT 3.6 s) | 276.9 (3.4 s) | 127.4 (9.7 s) | 96.4 (49.6 s) |
| 128 | **336.4** | 282.1 | 128.9 | – |

**RAG** (6,000-token prompt, 300-token answer)

| Users | Final | 64-session | Flash-Next | llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 41.6 (TTFT 1.67 s) | 33.2 (3.39 s) | 19.0 (4.04 s) | 28.5 (4.59 s) |
| 8 | 78.5 | 54.3 | 34.7 | 35.7 |
| 32 | **90.5** | 60.6 | 39.3 | 20.1 |

## DSpark speculative decoding

October 17, 2026, `--speculative dspark` against the same binary without it,
`MiniCPM-V-4_7-35B-A3B-Q4K-Q51down-Q8-outQ6K.gguf`, greedy, thinking off.

Production shape (32 sessions, 131,072 context, `tq4`, Kev in-process, disk
cache; `tools/minicpm/load.py`, output tok/s summed over users, off → on):

| Users | Decode | Chat | RAG |
| ---: | ---: | ---: | ---: |
| 1 | 57.2 → **65.5** | 54.6 → **72.4** | 42.1 → **54.7** |
| 8 | 195.1 → 191.0 | 155.5 → 151.1 | 78.0 → 74.9 |
| 16 | 265.9 → 261.3 | 199.3 → 195.1 | 84.6 → 83.0 |
| 32 | 378.4 → 372.5 | 254.9 → 249.0 | 93.1 → 91.6 |

TTFT is unchanged. A 20-minute mixed soak (8 agents with tools, 8 sampled
chats, 2 RAG users, 4 Kev clients) completed 1,306 requests and 2,481 Kev
requests without an error; GTT peaked at 66.3 GiB beside production.

By text kind (8 sessions, 400-token answers; unique prompts: a novel passage to
continue, coding tasks, and a Flask agent transcript whose next turn is a
`write_file` call; agent turns stream their tool call at the end):

| Kind | C1 | C4 | C8 |
| --- | ---: | ---: | ---: |
| Prose | 55.8 → 58.0 | 128.1 → 127.0 | 181.1 → 180.0 |
| Code | 57.5 → **112.4** (ITL 17.2 → 8.9 ms) | 134.2 → 143.6 | 188.2 → 187.4 |
| Agent | 48.8 → **87.3** | 105.6 → 161.4 | 119.3 → 167.5 |

Beside other sessions a drafting pass costs the whole step about 8 ms and
pushes verification past the eight-row GEMV route, so sessions there draft
only while their accepted drafts per pass pay for it (agent transcripts do;
prose and most code do not). The remaining 1–3% at C8–C32 is the per-step
feature injection.

Acceptance per verification on 420-token greedy continuations
(`qwen38_flash_next_dspark_probe`, cycle starts as speculative decoding visits
them; upstream llama.cpp `620d222` on the same drafter in brackets):

| Block | Prose | Code | Agent |
| ---: | ---: | ---: | ---: |
| 4 | 0.92 (0.90) | 2.02 (2.10) | 2.86 (2.90) |
| 8 | 0.98 (0.96) | 2.39 (2.58) | 4.09 (3.81) |
| 15 | 1.03 (0.94) | 2.48 (2.61) | 4.43 (4.37) |

These are the BF16 drafter with a full-vocabulary Markov chain. The served
drafter (Q8_0 projections, top-256 Markov shortlist) accepts 0.90/1.89/2.86
at block 4 and 0.98/2.31/4.24 at block 15. Distinct routed experts per layer
over 1/4/8 consecutive tokens are 8/18.5–20.6/28.4–32.7, well below
independent routing, so verification rows are cheap.

## Open loop

Chat shape, Poisson arrivals for 60 s at 25–125% of each server's closed-loop
peak. Goodput counts requests with TTFT ≤ 3 s and ITL ≤ 100 ms.

| | Final | 64-session | Flash-Next | llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| Peak goodput (req/s) | **0.51** | 0.46 | 0.11 | 0.25 |

## Long context

Cold prefill, one request, no cache (`qa.py prefill`):

| Prompt tokens | Final | 64-session | Flash-Next |
| ---: | ---: | ---: | ---: |
| 1,901 | **3,616** | 2,748 | 1,433 |
| 15,731 | **3,019** | 900 | 1,379 |
| 63,718 | **1,776** | 261 | 1,328 |
| 118,538 | 1,190 | 128 | **1,302** |

Decode at depth, prepared cohorts of identical sessions (`qa.py decode`, sum of
server-timed decode rates, 64-session build):

| Sessions | MiniCPM, empty | MiniCPM, 32K each | Flash-Next, empty | Flash-Next, 32K each |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 56.0 | 45.9 | 26.5 | 25.6 |
| 8 | 297.7 | 145.6 | 145.4 | 137.5 |
| 32 | 536.5 | 188.2 | 269.3 | 251.8 |
| 64 | 641.2 | 202.4 | 331.8 | **298.1** |

Attention is dense for `qwen35moe`, and decode attention launches one block per
query head, so each of the 8 query heads sharing a KV head re-reads its cache.
Above roughly 10K tokens per session at high concurrency, Flash-Next's indexed
attention decodes faster.

## Capacity

One session prefilled to 122,880 tokens, its snapshot restored into every other
session, then all decode together (`qa.py capacity`). Restored sessions map the
snapshot's KV pages, so these runs measure sessions sharing one document.

| | Sessions | Decode tok/s (sum) | ITL p50 | GTT peak |
| --- | ---: | ---: | ---: | ---: |
| MiniCPM final, tq4 | 64 | 64.4 | 1,036 ms | 30.6 GiB |
| MiniCPM final, tq3 | 128 | 61.6 | 2,166 ms | 38.5 GiB |
| Flash-Next, tq4 | 24 | 218.6 | 116 ms | 84.5 GiB |
| Flash-Next, tq4 | 48 | 281.2 | 178 ms | 89.2 GiB |

Independent 131K contexts need about 0.75 GB per session at tq4 (0.57 GB at tq3)
on top of 23 GiB of weights.

## K-quant weights

October 17, 2026, branch `perf/kquant-dense`: lab server with 8 sessions,
32K context, tq4 KV, greedy, thinking off (`tools/minicpm/load.py`, decode
shape; RAG is the 6,000-token prompt). C32 queues behind the 8 sessions. GTT is
the server's own resident memory. "Requant" files come from the `Q8_0`
original with the README command plus the noted overrides.

| Weights (binary) | C1 tok/s (ITL) | C8 | C32 | RAG TTFT | GTT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production requant (before) | 55.4 (17.8 ms) | 195.5 | 194.3 | 1.66 s | 22.6 GiB |
| Production requant | 55.6 (17.8 ms) | 193.1 | 188.2 | 1.68 s | 22.6 GiB |
| + `--output-tensor-type q6_k` (before: re-rounded to Q8_0) | 54.1 (18.3 ms) | 188.2 | 195.2 | 1.67 s | 22.6 GiB |
| + `--output-tensor-type q6_k` | 57.3 (17.2 ms) | 195.9 | 197.0 | 1.66 s | 22.5 GiB |
| + dense projections and head Q6_K | 61.6 (16.0 ms) | 193.8 | 193.8 | 2.31 s | 22.0 GiB |
| OpenBMB `Q4_K_M` | 64.8 (15.2 ms) | 167.6 | 167.9 | 2.07 s | 23.1 GiB |

Single-stream decode follows the bytes. Two kernel limits keep the K-quant
files from winning elsewhere (rotated weights beyond the MALL, MiniCPM shapes):

- Decode GEMV, GB/s at 1 / 8 rows: Q8_0 232 / 226, Q6_K 226 / 177, Q5_K
  218 / 91, Q4_K 209 / 77. The K-quant dot products are instruction-bound
  beyond four rows, so C8 gains nothing from Q6_K and loses with Q4_K/Q5_K.
- Prefill, 2,048 tokens, 12288x2048: Q8_0 F16 WMMA 3.0 ms; Q6_K MMQ 7.0 ms,
  Q4_K MMQ 4.7 ms. A dequantize-on-stage K-quant variant of the dense F16 WMMA
  GEMM is the follow-up; F16 weights through hipBLASLt took 18 ms.

## Images

48 rendered documents with exact answers, 16 concurrent, default projector
(about 228 tokens per image; `tools/minicpm/vision_eval.py`).

| | Correct | Images/s | Latency p50 |
| --- | ---: | ---: | ---: |
| Gufo + mtmd, f16 KV | 48/48 | **3.41** | 4.23 s |
| Gufo + mtmd, tq4 KV | 48/48 | 3.34 | 4.36 s |
| llama.cpp mtmd | 48/48 | 2.59 | 6.03 s |

## Energy

APU package power (`power1_average`, 1 Hz) over each closed-loop run, divided by
output rate; prompt tokens are not charged.

| Decode, 64 users | W | Wh per 1M output tokens |
| --- | ---: | ---: |
| MiniCPM final | 114 | 66 |
| Flash-Next | 118 | 151 |
| llama.cpp (MiniCPM) | 105 | 153 |
