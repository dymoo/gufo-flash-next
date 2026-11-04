# LFM2.5-24B-A2B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory, October 14, 2026.
`LFM2.5-24B-A2B-Q4K-Q51down-Q8.gguf` (see [Weights](README.md#weights)),
f16 KV, 128 sessions, HTTP streaming, greedy, unique prompts per request. The
model always reasons, so output tokens include the reasoning and TTFT is the
first generated token. MiniCPM-V 4.7 (`tq4`, 128 sessions) and Flash-Next
(production `UD-Q4_K_XL`, 24 sessions) are the Gufo rows in
[MiniCPM's benchmarks](../minicpm-v-4.7/BENCHMARKS.md). llama.cpp is `86ea01d`
on Liquid's `QAD-Q4_0`, 64 slots, `-fa on -kvu`. Load generator:
`tools/minicpm/load.py`.

## Final runtime, both models

The same final build (everything on `spike/lfm25`) and harness for both
models, October 14 evening. Output tok/s; TTFT and ITL medians.

| | LFM2.5 | MiniCPM-V 4.7 (`tq4`) |
| --- | ---: | ---: |
| Decode, 1 user | 62.0 | 55.1 |
| Decode, 128 users | **1,021** (ITL 120 ms) | 605 (206 ms) |
| Decode, 256 users | **1,289** (190 ms) | 671 (370 ms) |
| Decode sampled (T 0.6, top-p 0.95), 128 users | **980** | 586 |
| Decode sampled, 256 users | **1,219** | 652 |
| Chat, 128 users | **561** (TTFT 9.1 s) | 339 (8.2 s) |
| Chat, 256 users | **627** | 366 |
| RAG, 32 users | **139** | 90 |
| Energy, decode at 256 users | **25.7 Wh / 1M** | 49.5 Wh / 1M |

The sections below are the day's progression on earlier builds.

## Closed loop

Each user sends its next request when the last finishes. Output tok/s summed
over users; TTFT and inter-token latency (ITL) are medians.

**Decode** (200-token prompt, 512-token answer)

| Users | LFM2.5 | MiniCPM | Flash-Next | LFM2.5, llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 68.1 (ITL 14 ms) | 55.1 | 25.1 | **97.1** (10 ms) |
| 8 | 186.4 | 201.9 | 92.9 | **206.1** |
| 16 | **257.5** | – | 126.2 | 255.5 |
| 32 | **393.2** | 369.6 | 168.0 | 305.8 |
| 64 | **615.5** (ITL 96 ms) | 480.2 | 217.4 | 394.2 (141 ms) |
| 128 | **907.1** (ITL 133 ms) | 583.1 | 216.4 (queued) | – |

**Chat** (1,000-token prompt, 300-token answer)

| Users | LFM2.5 | MiniCPM | Flash-Next | LFM2.5, llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 65.4 (TTFT 0.26 s) | 52.7 (0.30 s) | 23.7 (0.79 s) | **84.7** (0.53 s) |
| 8 | **163.1** | 156.6 | 68.4 | 153.7 |
| 32 | **312.2** | 247.5 | 105.8 | 199.8 |
| 64 | **436.7** (TTFT 2.7 s) | 295.6 (3.6 s) | 127.4 (9.7 s) | 202.5 (26.1 s) |
| 128 | **552.2** (TTFT 9.2 s) | 336.4 (7.1 s) | 128.9 | – |

**RAG** (6,000-token prompt, 300-token answer)

| Users | LFM2.5 | MiniCPM | Flash-Next | LFM2.5, llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 47.9 (TTFT 1.82 s) | 41.6 (1.67 s) | 19.0 (4.04 s) | **51.0** (2.78 s) |
| 8 | **87.7** | 78.5 | 34.7 | 60.3 |
| 32 | **110.2** | 90.5 | 39.3 | 40.1 |

Single-user decode trails llama.cpp because the serving GGUF keeps every dense
matrix in Q8_0 (about 2.0 GB read per token against 1.4 GB for `QAD-Q4_0`).
Q4_0 dense and expert kernels would let Gufo serve the QAD file directly.

**Wide batches** (decode and chat shapes, 256 sessions per step, fused router)

| Users | Decode | ITL p50 | Chat | Chat TTFT p50 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 944.7 | 126 ms | 558.5 | 9.0 s |
| 192 | 1,164.3 | 159 ms | 604.8 | 33.9 s |
| 256 | **1,286.4** | 191 ms | 634.8 | 53.4 s |

At 128 sessions a step already reads nearly every expert of every layer
(14.3 GB) at the memory ceiling, so more sessions per step cost little more
time. Chat at 192+ users is prefill-bound: 1,000-token prompts queue.

Beyond 256 the gain flattens (sampled decode, final build): 1,023 tok/s at
256, 1,142 at 384 and 1,157 at 512 users, with ITL 415 ms (2.4 tok/s per
user) at 512; useful only for offline batch work.

**Sampled decoding** (temperature 0.6, top-p 0.95; the rows above are greedy)

| Users | Greedy | Sampled, serial draws | Sampled, parallel draws |
| ---: | ---: | ---: | ---: |
| 32 | 389.6 | 282.0 | 354.5 |
| 128 | 984.6 | 485.3 | **924.5** |

Drawing a token over 128,000 logits costs milliseconds of CPU; drawn
serially on the scheduler thread that halved 128-user throughput. Runners
whose token selection touches only host state (`concurrent_select`) now draw
a batch in parallel on every core, and a step's logits download once.

The draw itself is cheaper too. Top-p needs the softmax normalizer only to
place its cutoff, so it is first bounded (float exponentials near the
maximum, a bound for the rest) and computed exactly only when the cutoff is
ambiguous within the bounds; penalties are applied in one merged walk. The
same tokens come out (identical over 14,400 seeded draws across top-p,
top-k, min-p, min-keep and penalty settings). CPU per token, 128,000 logits:

| Settings | Before | After |
| --- | ---: | ---: |
| temperature 0.6, top-p 0.95 | 2.73 ms | 0.71 ms |
| the same with repeat penalty 1.05 | 4.91 ms | 0.75 ms |
| Liquid's defaults (0.1, top-k 50, penalty 1.05) | 1.41 ms | 0.40 ms |
| top-p 1, no top-k (full distribution) | 2.09 ms | 2.10 ms |

## Open loop

Chat shape, Poisson arrivals for 60 s at 25–125% of each server's closed-loop
peak. Goodput counts requests with TTFT ≤ 3 s and ITL ≤ 100 ms.

| | LFM2.5 | MiniCPM | Flash-Next | LFM2.5, llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| Peak goodput (req/s) | **0.53** | 0.51 | 0.11 | 0.43 |

## Long context

Cold prefill, one request, no cache (`qa.py prefill`). "FP32" is the first
port's scalar prompt attention; the WMMA kernel replaced it:

| Prompt tokens | LFM2.5 | LFM2.5, FP32 attention | MiniCPM | Flash-Next |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | **4,942** | 4,186 | 3,702 | 1,433 |
| 16,384 | **4,478** | 1,904 | 3,035 | 1,379 |
| 65,536 | **2,804** | 642 | 1,779 | 1,328 |
| 122,880 | **1,989** | 358 | 1,191 | 1,302 |

The prompt kernel gives each wave 16 queries of one head, so scores, softmax
and output stay in registers, and stages each 64-key tile once for the four
heads that share it.

Decode at depth, prepared cohorts of identical sessions (`qa.py decode`, sum of
server-timed decode rates):

| Sessions | LFM2.5, empty | LFM2.5, 32K each | MiniCPM, 32K each | Flash-Next, 32K each |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 69.6 | **56.5** | 45.9 | 25.6 |
| 8 | 453.5 | **236.1** | 145.6 | 137.5 |
| 32 | 964.7 | **298.8** | 188.2 | 251.8 |
| 64 | 1,199.1 | **302.7** | 202.4 | 298.1 |

Ten attention layers of 8 × 64 KV heads make 20 KB of f16 cache per token, so a
32K session reads 640 MiB per decode step; decode attention reads each KV tile
once for the four query heads that share it.

The 8-bit `tq8` cache halves those bytes (current build, same cohorts):

| Sessions | f16, empty | tq8, empty | f16, 32K each | tq8, 32K each |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 63.1 | 63.0 | 52.0 | 52.7 |
| 8 | 418.5 | 427.2 | 225.2 | 231.4 |
| 32 | 915.1 | 934.6 | 292.5 | **344.5** |

## Energy

APU package power (`power1_average`, 1 Hz) over each closed-loop run, divided by
output rate; prompt tokens are not charged.

| Decode, 64 users | W | Wh per 1M output tokens |
| --- | ---: | ---: |
| LFM2.5 | 105 | **47** |
| LFM2.5, 128 users | 122 | 37 |
| MiniCPM | 113 | 66 |
| Flash-Next | 118 | 151 |
| LFM2.5, llama.cpp | 119 | 84 |
