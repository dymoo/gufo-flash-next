# LFM2.5-24B-A2B quality

October 14, 2026. These check the port against llama.cpp on the same weights
and the served model on public tasks; they do not qualify the quantization
against the original checkpoint.

## Parity with llama.cpp

`tools/minicpm/parity.py`: llama.cpp (`86ea01d`, f16 KV) generates answers to
14 prompts with top-20 logprobs; Gufo is teacher-forced on the same text
(`forced_completion`) and every position is compared. Prompt token counts match
exactly in every case. Gates follow the Flash-Next QA: mean KL ≤0.005 pass /
≤0.02 warn; top-1 ≥98% / ≥95%.

| Gufo | Weights | Positions | Mean KL | p99 KL | Top-1 | Perplexity ref → Gufo |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| f16 KV | Q8_0 | 1,792 | 0.0035 | 0.039 | 97.8% | 1.342 → 1.344 |
| tq4 KV | Q8_0 | 1,792 | 0.0235 | 0.271 | 94.4% | 1.342 → 1.381 |
| tq3 KV | Q8_0 | 1,792 | 0.0610 | 0.573 | 91.1% | 1.342 → 1.440 |
| f16 KV | serving (Q4_K experts) | 1,777 | 0.0092 | 0.135 | 97.7% | 1.293 → 1.306 |
| tq8 KV | serving (Q4_K experts) | 1,777 | 0.0093 | 0.125 | 97.7% | 1.293 → 1.305 |

The reference used the GGUF's default repetition penalty (1.05), so greedy
texts are not comparable; the teacher-forced distributions are. Keeping the
router in F32 (as llama.cpp does) moved the Q8_0 row from KL 0.0043 to 0.0035.

Batched decoding (`qa.py batch`, f16, 8–64 identical cohorts against single
sessions): KL 0.0019–0.0024, top-1 98.2–98.7%, perplexity within 0.3%.

## KV precision

The model is unusually sensitive to quantized KV. TurboQuant on 64-wide heads
reaches its theoretical signal-to-noise (20.8 dB at tq4, 15.1 dB at tq3,
measured on captured keys), yet costs far more than on Flash-Next or MiniCPM
(tq4 KL 0.006 on MiniCPM against 0.024 here). llama.cpp shows the same on its
own: `llama-perplexity` on 40K characters of *Moby-Dick*, 2,048-token chunks:

| KV | Perplexity |
| --- | ---: |
| f16 | 24.35 |
| q8_0 | 24.11 |
| q4_0 | 25.67 (+5.4%) |

The 8-bit `tq8` format (same rotation, one int8 code per element and an
F16 scale per row: 66 bytes against 128) measures the same as f16 against
llama.cpp (table above) and halves KV bytes for long contexts; the 4- and
3-bit formats are not worth their error on this model.

## Which 4-bit weights

KL divergence from Q8_0 logits (`llama-perplexity --kl-divergence`, the
same 40K characters of *Moby-Dick*, 2,048-token chunks):

| Weights | Size | Mean KL | 99th pct | Perplexity vs Q8_0 | Same top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serving requant (Q4_K gate/up, Q5_1 down, Q8_0 dense) | 15.5 GB | **0.090** | 0.82 | +4.6% | **87.0%** |
| Liquid Q4_K_M | 14.5 GB | 0.151 | 1.23 | +4.0% | 82.2% |
| Liquid QAD-Q4_0 | 13.6 GB | 0.218 | 2.06 | +11.3% | 78.3% |

QAD (quantization-aware distillation) is trained for Q4_0 everywhere; it is
the smallest file but the furthest from the full model on this text.

Where the error comes from (same measurement, requants of the BF16 file in
formats Gufo serves; an importance matrix from 400 chunks of other text
changed nothing: KL 0.088 against 0.090):

| Routed experts (gate/up, down) | Dense, attention, convolution | Mean KL | Same top-1 |
| --- | --- | ---: | ---: |
| Q4_K, Q5_1 (serving) | Q8_0 | 0.090 | 87.0% |
| Q5_K, Q5_1 | Q8_0 | **0.069** | **88.7%** |
| Q4_K, Q8_0 | Q8_0 | 0.079 | 87.9% |
| Q8_0, Q8_0 | Q4_K | 0.144 | 82.6% |

The always-active dense path is the most sensitive part, so it stays 8-bit;
the next bits are best spent on the routed gate/up matrices (Q5_K: about 12%
more expert bytes per decode step at high concurrency).

## Hard evals

`hardevals.py` (spike harness), thinking on, temperature 0.6, top-p 0.95,
32K-token cap, one sample per item; both models on Gufo. MiniCPM-V 4.7
completed all 1,119 items; LFM2.5 1,091 (11 LiveCodeBench and 17 MMLU-Pro
items were cut by the GPU window after a host crash, likely among its
hardest).

| | LFM2.5 | MiniCPM-V 4.7 | Tokens per answer (LFM2.5 / MiniCPM) |
| --- | ---: | ---: | --- |
| GPQA Diamond (198) | 68.7% | **76.3%** | 12,111 / 4,098 |
| AIME 2024 (30) | 80.0% | **86.7%** | 12,191 / 8,217 |
| AIME 2025 (30) | 66.7% | **73.3%** | 15,307 / 10,773 |
| MATH-500 (100) | 90.0% | 90.0% | 1,930 / 1,425 |
| LiveCodeBench v6 (80) | **73.9%** (69) | 60.0% | 13,203 / 9,697 |
| IFEval, strict prompt (541) | **95.9%** | 89.8% | 2,048 / 1,138 |
| MMLU-Pro (140) | 81.3% (123) | **84.3%** | 3,638 / 1,859 |

LFM2.5 reasons much longer on hard items: 53 of its answers ran into the 32K
cap (MiniCPM: 2), 32 of them on GPQA. Per box-hour of correct answers at
each model's sampled 128-user throughput (980 and 586 tok/s), MiniCPM leads
GPQA (2.0x) and MMLU-Pro (1.2x), LFM2.5 leads LiveCodeBench (1.5x) and
MATH-500 (1.2x), and AIME and IFEval are even. Liquid's recommended
sampling (temperature 0.1, top-k 50, repeat penalty 1.05) was not tested
and may shorten the runaway traces.

### Against Qwen3.8 Flash-Next

Flash-Next (production, `tq4` KV) ran a nested 518-item subset with the same
prompts and settings. Paired over items all three models finished; gaps are
percentage points against Flash-Next with a 95% interval (McNemar):

| Suite (shared items) | Flash-Next | MiniCPM-V 4.7 | LFM2.5 |
| --- | ---: | ---: | ---: |
| GPQA Diamond (198) | 83.3% | 76.3% (−7.1 ± 4.3) | 68.7% (−14.6 ± 5.7) |
| AIME 2024 + 2025 (60) | 86.7% | 80.0% (−6.7 ± 11.2) | 73.3% (−13.3 ± 11.7) |
| MATH-500 (50) | 96.0% | 94.0% (−2.0 ± 6.8) | 94.0% (−2.0 ± 6.8) |
| LiveCodeBench v6 (35) | 65.7% | 65.7% (0.0 ± 15.8) | 71.4% (+5.7 ± 15.7) |
| IFEval, strict prompt (100) | 91.0% | 89.0% (−2.0 ± 6.2) | 97.0% (+6.0 ± 4.7) |
| MMLU-Pro (64) | 85.9% | 87.5% (+1.6 ± 5.3) | 81.2% (−4.7 ± 6.8) |

The seven-suite mean is 85.1% / 81.8% / 79.9%. MiniCPM's shortfall is hard
science (GPQA chemistry 74% / 63% / 54%; physics 94% / 92% / 87%); elsewhere it
is within two points. Flash-Next thinks about as long as LFM2.5 (10.8K tokens
per GPQA answer, 45 capped answers on the shared items against LFM2.5's 51 and
MiniCPM's 1) and decodes at about 217 tok/s on this box, so it yields 2.9-6.5x
fewer correct answers per box-hour than the better of the two smaller models.
