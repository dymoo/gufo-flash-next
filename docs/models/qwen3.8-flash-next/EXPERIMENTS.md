# Qwen3.8 Flash-Next experiments

| Experiment | Decision / evidence |
| --- | --- |
| Full-width MTP RMSNorm and split projection | Retained after independent CPU stage audit; one 10240-wide normalization, embedding projection shared across HC branches. |
| Full Q8 vocabulary head | Retained; private Q4 shortlist removed. Sampled top-64 proposals use exact target verification. |
| Batched MTP transformer and heads | Retained; independent body/head comparisons, private KV/recurrent/rollback/RNG state. |
| Batched decode mixers, residual epilogues and MTP norms | Retained; each request keeps its scalar reduction and private state; exact C2/C4/C6/C8 logits, acceptance and RNG. |
| HC Q8 weight prefetch | Retained for 1–8 rows of the 320×10240 projection; exact original products/FMA order, no extra allocation. |
| Q4 shared-expert weight reuse | Retained with a separate compact kernel for single-request experts; faster repetitive C1/C4/C8, no mixed-work regression, exact projection/model replay. |
| Q5 high-bit expansion | Retained; exact integer multiply/mask replaces repeated shifts, with unchanged dot products and faster serving. |
| Short convolution/history fusion | Retained for 1–8 tokens; exact output, rolling state and rollback snapshots, fewer launches and no additional allocation. |
| Batched GDN recurrence | Retained; private ragged state/rollback rows, cancellation isolation and exact session replay. Helps shallow batches most; end-to-end gains are modest. |
| Batched small projections and MoE preparation | Retained; group independent rows, quantize activations once in existing scratch, and batch router/shared-expert work. Exact scalar/batch outputs and sampled state; C1 unchanged. |
| Q4 expert grouping across the full batch | Retained; bounded groups share weights across request boundaries, with original scalar arithmetic and exact session replay. Q5 down grouping/reordering was slower on mixed routing and was rejected. |
| Expert-ordered prefill intermediates | Rejected: exact gate/up → down results with captured routing at 1024/2048 tokens, but no useful projection-chain gain for Q4_K/Q5_1 or Q5_K/Q8_0. |
| Wave64 batched Q4 gate/up | Retained above eight rows; independent 32-lane reductions preserve exact outputs and sampled state. Repetitive serving improves, C1 remains stable. Four output rows, 16-input groups and replacing the slot scan did not improve performance. |
| Register-cached vision softmax | Retained for bounded row sizes; unchanged reduction order, byte-identical Flash-Next/Qwen27B embeddings, no additional allocation. |
| 4096-patch vision attention tiles | Retained; byte-identical GEMMs/embeddings, lower latency and 24 MiB less attention scratch. Other shapes keep their original tiles. |
| Partitioned BF16 WMMA vision value projection | Rejected: failed the full-encoder reference gate despite lower isolated FP64 error. |
| Integer WMMA Q8 verification | Retained through 48 input rows for wide target projections/heads; preserves K8 partials/FMA order and reduces each result once, with exact session replay. Small batches retain vector kernels. |
| Q8 activation reuse across output rows | Rejected: no repeatable gain on the real projection shapes. |
| Wider Q8 decode and Q5 expert tiles | Rejected: 56–64 dense rows, phased/packed token tiles, Q5 wave64 and 16/32 Q5 expert rows were slower despite exact output. |
| Compact expert launch groups | Rejected: improved shared routing but negligible mixed-routing gain. |
| Sparse attention register/cache retuning | Rejected: exact d128K output, but register scheduling/occupancy gave no material gain and reloading queries was slower. |
| Shared attention selection lists | Rejected: exact outputs, but roughly 1% isolated gain did not justify another buffer and setup kernel. |
| Skip unused attention Q8 staging | Rejected: exact chunk/full-logit checks passed, but no clear end-to-end prefill gain justified the extra dispatch logic. |
| FP32 selector load scheduling | Retained; bounded scheduling removes scalar-register spills, preserves every score bit and lowers d32K selection time to 22.4 ms per pp2048. Matched d128K AR prefill improves about 1.5%. |
| Integer WMMA value transpose and paired FP32 selector lanes | Rejected: bit-preserving transpose and exact selector scores, but both were slower on deep-context inputs. |
| Packed Q8 prefill staging | Rejected: exact output, but extra decode/register/transpose costs outweighed reduced LDS use. |
| Persistent packed Q8 SSM weights | Rejected: small prefill/1–4-row gains would cost roughly 14% on eight-row decoding; two-step prefetch did not recover it. No second weight copy or private format retained. |
| Transient F16 SSM weights and parallel HC branches | Rejected: F16 staging was exact but slower overall; parallel HC branches changed quantization ties. |
| Smaller-LDS SSM projection, unrolled HC expert sum and wave64 HC combine | Rejected: exact output but no useful prefill speed gain. |
| HC quantized-output store layouts | Rejected: row-major staging plus transpose is slower; cooperative stores within one kernel save only about 2% in isolation. Residuals, normalized activations and Q8 bytes remain exact, including ragged rows. |
| Transient key transpose and mixed expert tiles | Rejected: exact outputs; key transpose slows deep selection, mixed tile sizes provide no useful prefill gain. |
| Query sharing, paired-key FP32 scoring, query LDS caching, MoE prefetch barriers/unrolling | Rejected: exact outputs, but no useful speed gain. Four-wave Q8 matrix reduction also lost to eight waves. |
| Approximate selector screening followed by exact rescoring | Rejected: GPU thresholding and compaction erased the isolated gain, before accounting for runtime error bounds. Remains a synthetic experiment; no approximate production selector added. |
| Ratio-four predictor QSA, FP32 ranking queries | Retained with sparse state, rewind and deep selector checks. |
| Greedy batch cost controller | Retained only for all-greedy C>1; separate occupancy/context bins, stable plain controls, no transition timings. Sampled replay uses fixed curves calibrated from median warmed cycles on 2026-09-20. |
| One-row MTP prefill lag | Retained; 320 KiB kept hidden state/session, avoids replaying a final prefill chunk. |
| Final-tile MTP catch-up | Retained; preserve every KV/indexer row, rank only the final attention tile, then restrict output projection, mixers and shared experts to the final aligned tile(s), with one routed Q8 expert row. Full predictor/catch-up stages, candidates and recursive carry match exactly at 224/257/2047/2048 rows. |
| Batched MTP catch-up tail | Retained; after writing every attention cache row, compact final request rows into existing scratch and skip unused output projections/FFNs. Preserve the scalar arithmetic of one-row tails; full-head candidates and recursive carry match independent execution. |
| Routed Q8 accumulation order | Retained; explicit rounded product/FMA prevents identical rows changing with column placement. Independent FP64 dot and predictor carry checks pass. |
| Isolated MMQ quantizer rounding change | Deferred: it changes half-integer tie behavior shared with W8A8 and fails their existing agreement gate. No quantizer change retained. |
| SSM tile, wave and compiler scheduling variants | Rejected: four/sixteen-wave groups, wave64, wider token tiles and iterative ILP scheduling retained exact output but did not beat the existing eight-wave projection. |
| SSM fragment lifetime and prefetch pipeline | Rejected: shorter-lived K16 fragments reduce register use without a useful gain; delayed prefetch and double-buffered LDS are slower. Projection/convolution outputs remain exact at 2048/2049 tokens. |
| Captured shared stages in batched target decoding | Rejected: exact C2/C4/C6/C8 logits, sampled state and cancellation checks, but no serving or warmed-cycle gain justified graph metadata/capture overhead. |
| Compact recurrent rollback | Retained; one full state plus exact FP32 update operands, with the original product/FMA order. All rollback prefixes match fresh execution. Depth grows on demand; seven-draft cap about 147 MiB/session, released on reset. |
| Live final frontier and async prompt snapshots | Retained; immutable branch snapshot, worker capture, bounded persistence outside the lookup lock. |
| Thread-local decode graph capture | Retained; independent snapshot workers no longer invalidate a peer's capture. Snapshot bytes and captured/replayed logits are exact; no request serialization added. |
| Chunk-equivalent projections/attention | Retained with exact continued-image/cache/full-logit gates; one-token tails keep prefill arithmetic. |
| More Q8 vocabulary rows/block | Rejected: no C1 improvement. |
| Private shared-expert F16 rows | Retained; the shared expert's SwiGLU rows for its F16 down projection get their own buffer instead of overwriting the routed experts' narrowed token rows, which removes one full-batch narrowing pass per layer. Every GEMM reads identical bytes; greedy output, chunk-boundary logits and the C2–C8 batch checks are exact. Interleaved Nix A/B at d0: 1559.6 → 1568.1 tok/s (+0.6%); the profiled kernel time fell 1362.8 → 1333.1 ms. |
| 256-row routed down-projection blocks | Rejected: exact output, Q5_1 unchanged and Q8_0 slower. The routed kernels already stream expert weights at roughly two thirds of DRAM bandwidth, so per-block prologue/epilogue and activation restaging were not the bound. |
| 64-token HC down-projection tiles | Rejected: exact output, 23% slower than the 128-token tile despite twice the resident blocks. |
| One-tile-ahead LDS fragment prefetch (dense and routed WMMA) | Rejected: exact output, but the explicit double buffer raised register use (dense 221 → 253 VGPRs; the 128-token pair reached 256 with spills) and every GEMM slowed 3–14%. |
| hipBLASLt F16 dense projections | Rejected: the pinned library reaches 19–26 TFLOPS on the 2048-token dense shapes against 30–34 TFLOPS for the Q8→F16 WMMA kernels; `tools/qwen-flash/dense_blaslt_sweep.hip` reproduces the sweep. |
| Batched decode to 64 sessions with tiled shared projections | Retained; up to 64 requests (512 verification rows) per step. From a per-shape crossover each shared projection reads its weights once through the W8A8 or F16 WMMA GEMM instead of per-chunk GEMVs: at 64 rows 583 → 296 µs for 16384×2560, 350 → 107 µs for 2560×6144, 9.3 → 3.6 ms for the vocabulary head (`qwen38_flash_next_batch_dense_bench`, real weights). The 320×10240 HC down GEMV keeps its route below 160 rows. PLE projections are shared, MoE beyond 64 rows takes the tiled expert route, logits reuse idle MoE scratch and greedy verification reduces every row on the device. Relaxed contract, see [quality](QUALITY.md). |
| Side-stream inject/shared-expert overlap | Rejected: exact output and real kernel overlap in the trace, but the co-running kernels slowed each other and interleaved wall-clock runs were 0.5–0.9% slower. |
| Device-planned routed experts for decode/verification | Retained for 1–512 rows: on-device bucketing (no host sync, graph-capturable), each expert streamed once, exact int8 32-value dots with one explicitly rounded FP32 order, so rows are bitwise independent of batch and position. FP64 error on Q5_1-down layers halves (relative RMS 2.2% → 1.1%; the offset term uses the code sum). Layer-0 microbenchmark: 8 rows parity, 64 rows 6.5 → 4.9 ms, 512 rows 8.5 → 8.1 ms against the prefill route. Replaced the routed GEMV kernels. |
| Lane-interleaved routed expert layout | Retained: Q4_K/Q5_K gate/up and Q5_1/Q8_0 down experts are permuted in place at load (32-row groups, 16-byte pieces contiguous across a wave; same bytes, no second copy, about 24 ms per layer). Decode and prefill routed kernels read it; outputs are bit-identical. Layer-0 routed MoE: 8 rows 1077 → 1022 µs, 64 rows 4894 → 4615 µs; prefill route 8 rows −17%, 512 rows −14%, 2048 rows −4% (`qwen38_flash_next_moe_bench`). |
| Two-stage weight prefetch in the prefill Q5_1/Q8_0 routed GEMM | Rejected: exact output, down projection within ±1%. |
| Prompt tiles for 256–1023 tokens | Retained, bit-identical: large F16 projections take the eight-row-group tile below 1024 tokens (256 tokens: 16384×2560 802 → 714 µs, 2560×6144 292 → 252 µs), the fused HC down its 64-token tile (145 → 93 µs), 640-row W8A8 projections 64×64 tiles (78 → 41 µs), and every prompt batch pairs routed gate/up (routed MoE −1.3 to −3.1% at 256–512 tokens). No change from 1024 tokens (`qwen38_flash_next_batch_dense_bench <shard> prefill`). |
| Double-buffered LDS, 4×4 wave tiles, wave64 W8A8 below 1024 tokens (dense prefill) | Rejected: exact output; the double buffer (BK = 1, one barrier per stage) ran 3× slower, 4×4 wave tiles 3–6% slower, wave64 W8A8 neutral. gfx1151 int8 and F16 WMMA peak at the same rate (57.4 TOPS / 55.5 TFLOPS), and the per-block FP32 scaling of W8A8 costs 25%, so F16 remains the dense prefill route. |
| MALL-aware block order for large prompt batches | Retained, bit-identical: the 2560×6144 F16 projection (SSM output) dispatches the ten row blocks of each token block together once its F16 activation panel exceeds 32 MiB. At 4096 tokens 5.7 → 3.6 ms (22 → 35 TFLOPS); 2048 tokens keep token-major order (1.73 ms; row-major was 8–10% slower there). Token-sliced launches instead were slower at 2048 and only partly recovered 4096 (4.9 ms). |
| gfx1151 WMMA instruction rates incl. int4 (`tools/bench/wmma_issue_probe.hip`, 2026-10-22) | Measured, 40 CUs at 2900 MHz, results host-verified: fp16, bf16 and fp16-accumulate 55.5 TFLOPS, iu8 57.2 TOPS, **iu4 110.9 TOPS (2.00x fp16)**; every type sits at 93% of its theoretical rate and one dependent chain saturates at 8 waves/SIMD. An iu4 16x16x16 WMMA issues in half the time of fp16/iu8. WMMA and VALU do not overlap on a SIMD: time = WMMA clocks + 1 clock per VALU op, which fits every epilogue row (iu4, per-scale-group int32->fp32 epilogue: K=16 group 60.8 TOPS, K=32 77.5, K=64 88.9, K=128 96.6; none 103.7). int8 therefore has no instruction-rate headroom over F16 (the earlier W8A8 finding, explained); int4 does. |
| W4A4 iu4 dense prefill GEMM probe (`tools/bench/wmma_gemm_probe.hip`, 2026-10-22) | Same tiling for fp16/iu8/iu4 (128x128 tile, 64-byte rows per K slab, double-buffered LDS, activations as the A operand so a group scale is one scalar per lane). Per-token activation scale, Q4_K-style 6-bit integer weight group scales folded with `v_mad_i32_i24`, T = 2048: ssm_in 16384x2560 62.3 TOPS (g32) / 69.2 (g64) / 69.2 (g128) against Gufo's best route 35.3 TFLOPS; attn_qkv 61.7 / 68.7 / 70.6 vs 35.3; ssm_out 59.5 / 67.8 / 70.2 vs 36.4; shared gate+up 54.7 / 59.5 / 62.3 vs about 32; shared down 52.7 / 59.7 / 62.3 vs 29.8. Row-major fp32 output (what the consumers read today) costs 15-25% and made the kernel erratic (30-58 TOPS across T); tile-major output is stable. iu8 variants reach only 21-32 TOPS, below the F16 kernel. Kernel only: random operands, no activation quantizer. |
| Integer W4A4 prefill path in the engine (`GUFO_SPIKE_REAL`, spike branch, 2026-10-22) | Wrapper-level, real Q8_0 weights, includes the int4 activation quantizer with a block-Hadamard (128) and row-major fp32 output: 2048 rows ssm_in 4.65 -> 3.90 ms (1.2x), attn_qkv 3.83 -> 2.78 (1.4x), ssm_out 1.77 -> 1.18 (1.5x), attn_out 1.79 -> 1.26 (1.4x), shared gate/down about 1.4x; about 1.33x over the dense projections as a set (1500 rows: 1.3-1.5x; 300 rows 1.4-1.6x). Dense projections are 25% of a fused agent step (44% routed experts, which stay on the shared decode-route pass), so this is about -6% step time before any quality cost. Numerics: relative RMS error against the Q8_0 GEMM on outlier-heavy synthetic activations 0.12-0.23 for W4 g32 + A4 per-token + Hadamard128, 0.65-0.78 for A4 without rotation, 0.13-0.15 for A4 per-32-group, 0.06-0.07 for A8; the integer and emulated paths agree to 0.5%. |
| Direct-to-register weight fragments, 256-row routed tiles | Rejected: exact output. Loading each wave's Q8_0 rows straight into WMMA fragments (activations alone in LDS) reached 256 VGPRs with spills and ran 4× slower; 256-row routed Q4_K gate/up and Q5_1/Q8_0 down tiles were 5–50% slower, so the routed GEMMs are not bound by activation LDS reads. |
| Sparse prefill attention masking from LDS | Retained; the union list records which of its four queries selected each block, so the softmax no longer reads global mask words, and separate K/V LDS tiles drop the loop-head barrier; TurboQuant tiles are prefetched two ahead. Bit-identical. F16 per layer (`qwen38_flash_next_prefill_bench`, random 512-block selections) d100K 512/2048 rows 7.2/26.4 → 5.8/20.9 ms, d32K 5.6 → 5.1 ms; correlated selections −5% to −14%; TQ4 −5% to −11%. A second F16 stage exceeded 240 VGPRs (five waves per SIMD) and was slower. |
| Eight-query dual-issue prefill selector | Retained above eight rows: each thread converts its pooled key to F32 once and scores eight queries with the unchanged FMA chains and reduction tree (VOPD FMAs instead of single-issue mixed-precision FMAs); selection runs 256 rows per launch so the scores stay in the MALL for marking. Bit-identical scores and masks; selection per layer at 512 rows d32K/d65K/d100K 0.62/1.11/1.72 → 0.44/0.83/1.30 ms. Unrolled marking passes and an eight-token GDN window were neutral. |
| 48 sessions and `--max-itl-ms 3000` (agent swarm, production flags) | Not adopted. 40 agents, 20K-token shared prompt, tq4, the live workhorse flags (Kev, `--admission refuse`, 22 GiB KV pool cap), same seed, 240 s of arrivals, one run each, 2026-10-21, host memory PSI (full avg10) mean 1.2 / 2.0 / 2.6 (healthy): 32 slots (production) / 48 / 48 with `--max-itl-ms 3000`: turns 91 / 88 / 88; decode 53.1 / 48.5 / 47.3 tok/s; TTFT p50 2.95 / 3.24 / 3.65 s, p90 16.5 / 26.7 / 18.8 s; ITL p99 2.20 / 2.55 / 3.29 s; agents refused with 503: 8 / 0 / 0; peak GTT 89.7 / 91.4 / 91.2 GiB; peak acpitz / Tctl / edge 90/92/88, 89/92/87, 89/90/88 C, never at 95 C. The GPU is already saturated at 32 slots: 48 admits the 8 agents the production setting refuses (so they would not fall back to the cloud) at about the same total work, 9% lower decode rate and clearly worse latency; a 3 s ITL target only worsens the tail. Keep 32 and the default 1500 ms unless avoiding refusals matters more than latency. |
| Pinned or double-buffered upload for snapshot restores | Rejected before building it into a PR. A pageable `hipMemcpy` of a 360 MiB image takes about 5 ms on gfx1151 (73 GB/s; pinned source 4.9 ms), and a phase trace of 36 restores at the production flags puts the copy at a median 4 ms of a median 148 ms restore. The upload was never the cost. |
| Restore phase breakdown and a pre-committed reserve pool | Not needed. `GUFO_RESTORE_TRACE` laps of 34-40 session restores per run: under host memory stall (PSI full avg10 15-32%) a restore took a median 148 ms, 97 ms of it in the page-in commit/reserve of GPU memory and 40 ms in attaching KV pages; on a healthy host (PSI mean 1.2-2.6%) the same restores take a median 40-44 ms: 18-19 ms commit/reserve (p90 50-75 ms), 10-12 ms attach/reserve, 4 ms copy. The 97 ms was direct reclaim stalling the page mapping, so a pre-committed reserve pool would buy about 20 ms per restore, 0.24-0.28% of wall time at 34-40 restores per 287 s. |
| Two GDN state rows per lane in prefill | Retained; each q/k LDS fragment serves two rows, with the original per-lane FMA chains, reduction and update order written out explicitly (implicit contraction produced different bits). Bit-identical output and state; recurrence 512/2048 tokens 0.72/2.93 → 0.61/2.44 ms per layer. Four rows per lane (one block per head) was slower. |
| TurboQuant-style KV cache (`--kv-cache tq4\|tq3`) | Retained, opt-in: fixed randomized Hadamard rotation plus Lloyd–Max codes and one FP16 scale per 256-value head row (tq4 4.06 bits, tq3 3.06 bits); quantize on store in every writer, dequantize in decode, prefill and MTP readers. 3.9×/5.2× smaller KV. QJL residuals, unrotated per-32 V and a uniform 4-bit codebook measured worse. Quality in [QUALITY](QUALITY.md#serving-options-and-lossy-flags). |
| Balanced split-K decode attention | Retained: long-context decode was DRAM-latency-bound; spreading each row's selected blocks evenly over splits and loading eight keys per wave cut d131K decode attention from 581 to 49 µs per layer (1 row). F16 keeps the original key loop and constant splits so dense positions stay bit-identical to `main`. |
| One attention launch per layer for all batched sessions | Retained; descriptor-driven, bit-identical to per-session calls. |
| Batch-aware MoE routing (`--moe-routing affinity/mass/budget`) | Rejected as a speed lever: shrinks the expert union on synthetic logits, but on device-planned experts measured −8% to +6% at 32–64 sessions with higher batched-row KL. Kept flagged and off. |
| SplitFuse, first version (prompt chunk inside MTP decode steps) | Reverted: decoders drifted past tolerance and a mixed agent load showed no gain. |
| Multi-session prompt packing (second SplitFuse) | Retained: several requests' prompt chunks share one step; decode rows keep decode-only arithmetic, prompt-only steps equal standalone prefill. A halving rule across prompts regressed agent load (decode 25.6→10.9 tok/s); shortest-first to completion fixed it. |
| Fused-step ITL target | Retained at 1500 ms: 750 ms never grew past the fixed budget (steps already ~700 ms); 1500 ms finished 30% more turns and halved first-turn TTFT under a 96-agent load. |
| Lazy KV commit (HIP VMM piece pool) | Retained: VMM works on gfx1151 (4 KiB granularity; graphs replay across later mappings). Slots commit KV as they grow; idle sessions page to RAM then NVMe; 32 × 131072 sessions load. Unmap does not return memory until the reservation is freed, so pieces are recycled in-process. |
| Host memory budget | Retained: every host-RAM consumer sized itself from the container's view of memory; on a shared host that drove host MemAvailable to 6.6 GiB. One explicit `--host-memory-budget` now bounds snapshots, paging and staging. |
| Shared system-prompt prefix store | Retained: checkpoints at the end of the system block, pinned after three conversations, persisted on disk. Agent load: misses 18→2, TTFT p50 33→6.9 s, turns 214→341. |
| Disk checkpoint read-ahead | Retained: a disk-tier restore read, SHA-256-checked and uploaded the whole image on the scheduler thread, about 260 ms for a 360 MB conversation, GPU idle. The scheduler now offers the first request of its next three queued clients to the store, whose reader thread verifies the image into the staging budget (at most half of it, skipped when RAM serves the prompt about as well); the restore takes the verified image or falls back to the inline read, so the bytes restored are unchanged. 40-agent fast-ramp load, 24 slots, tq4, same seed, 240 s of arrivals, one run each: disk restore median 257 → 111 ms (mean 260 → 144), stalls of 150 ms or more 5.9 → 2.3 s (2.5% → 1.0% of wall time). The remaining ~110 ms is not the upload: a pageable `hipMemcpy` of 360 MiB takes about 5 ms (73 GB/s, `tools/bench`-style microbenchmark), and a phase trace of 36 session restores at the production flags (32 slots, tq4) put the copy at a median 4 ms. The time is in making the target session resident again, before the restore overwrites it: median 97 ms in the page-in commit/reserve of GPU memory (no bytes copied) and 48 ms attaching and reserving KV pages, 148 ms per restore in total (2.3% of that run). That trace ran while the host's memory PSI (full) sat at 15-32% throughout; on a healthy host (PSI 1-3%, three later runs) the same restores take a median 40-44 ms (commit/reserve 18-19 ms, attach/reserve 10-12 ms, copy 4 ms), so the 97 ms was mostly direct reclaim stalling the page mapping. The 257 and 111 ms disk-restore medians above were measured in stalled periods too: the relative gain (read, hash and parse off the scheduler thread) should hold, the absolute numbers are inflated. |
| Online step-cost MTP controller and prompt-lookup drafts | Retained: per-session draft counts from a learned step-time model, never below AR under the model; lossless prompt-lookup drafts. MTP still gives no gain on agent load at high concurrency. |
| UD-IQ4_XS and UD-Q3_K_XL experts | Retained for IQ4_XS (packed IQ layouts, WMMA prefill): prefill 10–16% faster per layer at 256–1024 rows, ~18 GiB less memory, measured quality cost. Q3_K_XL loads but is unqualified. |
| GPU DPM level `high` | Rejected for sustained load: GPU edge and CPU Tctl reached 108 °C and the host powered off during long-context prefill; `auto` holds ~2475 MHz and ≤92 °C. |
| NPU (XDNA2) prefill offload | Not pursued: dispatch overhead and shared DRAM/power bound the gain near that of disabling the IOMMU, which also removes the NPU. |
| Flex tier admitted on free sessions beyond a reserve of 4 | Rejected: with a 48-agent flex swarm beside the standard 40-agent load, default turns fell 243 → 164 and TTFT p50 rose 3.1 → 9.6 s. Flex took up to 20 sessions between agent turns and held them to completion, so default turns queued behind it. |
| Flex capped at 2 sessions, admitted only with no default prompt queued or prefilling | Partly: standard-load default turns 194 (−20%), but light-load ITL p90 rose 97 → 316 ms. Beside a few default decoders a flex prompt chunk turns their cheap routed-expert decode step into a full expert stream, and the ITL target then lets it grow toward 1.5 s. |
| Flex prompts only in steps with a default prompt or no default decoder; flex prefill always chunked | Retained (default `--flex-sessions 2`): light-load ITL p50/p90 73/92 → 84/113 ms, turns within run-to-run noise (two identical baselines: 89 and 76); standard load 199 turns. Flex itself completed 7–8 turns in 5–10 min: agent traffic leaves little idle compute, so flex is for genuinely idle periods such as overnight batch work. |
| NVIDIA FP4-QAT checkpoint → INT4, baseline-identical tensor types | Rejected (2026-10-24): decode KL 0.078 and top-1 92.8% vs UD-Q4_K_XL, ppl +2.2%; prefill at parity; coding swarm 176/140 vs 124/164 correct/h, overlapping CIs, 4/83 failures vs 0/72. Adapter not run. [Details](#nvidia-fp4-qat-checkpoint-2026-10-24). |
| Speed campaign (2026-10-25/26): retained and rejected ideas, A/B numbers, evidence paths | [Section below](#speed-campaign-october-2026). Retained: exact decode kernels, int4 dense prefill, F16 GDN state, GPTQ decode twins, MoE macro batch, combined MTP policy, block-selection fix. Rejected: top-k routing for agents, routed int4 experts and the F16 residual (prefill-row budget), chunked GDN prefill, AutoRound experts, power and clock knobs. |
| Plane-aware LDS chunk swizzle in the dense F16 GEMM | Retained, bit-identical: the K-block planes of a BK = 2 stage are a multiple of 128 bytes apart and the commit assigns neighbouring lanes to the two planes of a row, so every 16-byte commit store had a two-way bank conflict (12% of LDS-active cycles, `SQC_LDS_BANK_CONFLICT` 2048 cycles per wave). Folding the plane index into the chunk permutation (`(row >> 1) ^ (kk << 1)`) leaves 0.8%. Real Q8_0 weights, 2048 rows, ten alternating runs counting `GRBM_COUNT` cycles per dispatch (clock independent): ssm_in 16384×2560 −4.9% cycles, attn_qkv −4.5%, ssm_out −4.1%; time +3.5%, +3.3%, +3.8% (the package is power limited, so the clock sags 2% when the pipe is busier). Wall-clock ABBA process pairs in four further experiments: ssm_in +3.4% to +3.6% (+0.4% in one cool-start run whose baseline alternated between two clock states, 4430 and 4700 µs), attn_qkv +3.0% to +4.5%, ssm_out +1.7% to +4.1%, attn_out +2.8% to +3.5%, ssm_in +2.9% to +5.6% from 256 to 4096 rows; a sustained 2.3 s run +3.4% (4717 → 4564 µs). The BK = 4 tile that serves fewer than 96 rows (batched decode and verification) had four-way conflicts and gains more: decode-mode ABBA against origin/main (six rounds, 429 shape × row × route cases), F16 route at 8–64 rows median +11.2% (hc_down +24% to +27%, shexp_gate +21% to +25%, attn_out +11% to +18%, ssm_out +6% to +15%, shexp_down +9% to +15%, ssm_in +8% to +11%, attn_qkv +5% to +10%), 96–192 rows +1.6%, 256–512 rows +2.5%; the unchanged W8A8 and GEMV routes move −0.1% to −0.7% in the same runs (noise floor). All 85 whole-output hashes of the bench (and 120 at ragged rows 1 to 4095), the FP64-error column of its decode mode (429/429 cases), the `projection_ops_test` and `hc_mix_ops_test` logs and the `blaslt_ops_test` output hashes (F16-weight router and alpha/beta projections) are identical to the base build. BK = 1 tiles are unchanged (hc_up within ±1%) and narrow prefill tiles stay within ±2%. The prefill gain is about 0.7% of a pp2048 pass by the d0 profile's kernel shares (30 ssm_in, 10 attn_qkv, 30 ssm_out and 10 attn_out calls); the small-batch gain is not converted to decode tok/s. Neither is measured end to end. [Counter pass](#dense-f16-prefill-counter-pass-2026-11-05) |
| Dense F16 GEMM ablations: LDS fragment prefetch, deeper global prefetch, coalesced activation loads, 128-byte-row LDS layout, branch-free fetch, fetch spread over the K loop, LDS-only barriers, `s_setprio`, block phase offset, other tile shapes, swapped WMMA operands | Rejected or neutral, all exact where the output was compared (hash equal): none gives more than the swizzle. Numbers in the [counter pass](#dense-f16-prefill-counter-pass-2026-11-05). |

Separate d32K pp2048 profiling attributes 29.1% of kernel time to MoE, 34.9%
to dense projections and 12.6% to attention/indexing. Final-tile catch-up
takes 20.9 ms of MTP kernel time; the target takes 1450.2 ms.

A d0 pp2048 profile (2026-09-21) is GPU-bound: 1362.8 ms of kernel time in a
1380 ms span. Routed expert GEMMs take 35%, dense F16 projections 29%,
hyper-connection combines 10.5%, the GDN recurrence 9.5%, HC down/inject/shared
projections 7% and attention 5.5%. The dense projections run at 30–34 TFLOPS
against the measured 59 TFLOPS matrix ceiling (a 2.9 GHz burst number; the
kernels sustain 2.25–2.5 GHz, see the [counter pass](#dense-f16-prefill-counter-pass-2026-11-05))
and the combines at about 200 GB/s; the routed gate/up and down kernels stream 0.9–1.1 GB of expert
weights per layer at 160–190 GB/s.

A C4 mixed MTP trace with 32 output tokens per request is 84.9% GPU-busy
during inference, excluding model loading. Batched GDN updates take 40.9 ms,
rollback replay 14.5 ms and lazy allocations 31.8 ms. The scheduling thread
spends 5.4% of the window outside HIP API calls, including model-side CPU
work; this is not pure scheduler overhead. These are profile observations,
not unprofiled throughput measurements.

Next: improve prefill at depth and target/draft batch projection reuse while
preserving [quality](QUALITY.md). The 1700 tok/s PP and flat d0–d128K
objectives remain unmet; see [current benchmarks](BENCHMARKS.md).

## Dense F16 prefill counter pass (2026-11-05)

First hardware-counter pass over this code (`rocprofv3` 1.1.0, ROCm 7.2.3,
`tools/prof/pmc.py`; counter set and pitfalls in
[PERFORMANCE](../../PERFORMANCE.md#hardware-counters-pmc)). Question: why do the
dense F16 projections run at 30–34 TFLOPS at pp2048 against a 59 TFLOPS matrix
ceiling? Case: `qwen38_flash_next_batch_dense_bench <shard> prefill` with
`DENSE_BENCH_SHAPE=ssm_in DENSE_BENCH_ROWS=2048 DENSE_BENCH_ROUTE=f16`, real
Q8_0 weights, four weight copies rotating past the MALL. It ran beside the
production server on the test host, so absolute times are noisy; every
comparison is interleaved. The timeline and ablations used a standalone copy of
`DenseF16GEMMKernel<256, 128, 2, 8, 1>` with the library kernel's resource usage
(221 VGPRs, 48 KiB LDS, 4 waves per SIMD); it is not retained.

**Answer.** Part of the gap is clock and part is matrix-pipe idle time; neither
memory nor occupancy is involved.

| Measure (ssm_in 16384×2560, 2048 rows) | Value | Reading |
| --- | --- | --- |
| Clock (`GRBM_COUNT` / dispatch time) | 2.44–2.51 GHz; sustained 3–6 s runs 2.25–2.48 GHz at 139–156 W | The 59 TFLOPS ceiling is a 2.9 GHz number. The SoC is power limited and WMMA power depends on operand toggling: the calibrator's constant operands hold 2.9 GHz, the same kernel shape fed constants ran at 2.6–2.9 GHz and 42.9 TFLOPS, fed pseudo-random words 2.34 GHz and 38.7. At 2.45 GHz the measured F16 WMMA ceiling is about 47 TFLOPS |
| Occupancy | 7.8 waves per CU of 8; WGPs busy 98.5% of active cycles | full: 1024 blocks × 8 waves, `SQ_WAVES` exactly 8192 |
| Matrix pipe | about 78% of WMMA issue slots in the K loop: 11 230 cycles per wave-stage with four waves per SIMD, against 4 × 2200 | the part that is left to close |
| Instructions per wave | 8274 VALU (2560 of them WMMA), 3776 LDS, 1066 SALU | dequantization is not the cost: removing it changed nothing |
| LDS | `SQC_LDS_IDX_ACTIVE` (modelled, 4 cycles per `ds_*_b128`); bank conflicts 12.0%, 2048 cycles per wave, all in the stage commit stores | now 0.8% |
| LDS bandwidth | about 70 B/clk per WGP in the K loop, 55% of 128 B/clk (calibrator: 6.8 TB/s, 117 B/clk per WGP at 2.9 GHz); 67% at the full matrix rate | a co-limiter, not a wall |
| L2 and fabric | L2 hit 70.8%; 360 MB read from the fabric per dispatch for 55 MB of unique data, about 75 GB/s. A 64 MiB write between launches (cold MALL): hit 71.3%, +2.2% time | d32K behaves like d0 for these GEMMs. Making the weights, the activations or both cache resident changed time by −0.3% to +1.1% |
| Wave time, K loop (cycles) | commit (load wait, dequant, LDS stores) 11.1%, barrier 1 3.8%, fetch issue 4.7%, compute 69.2%, barrier 2 9.4%; the output stage adds about 6% of the loop | removing the output stage entirely is +8.7%; direct stores without the LDS transpose were not faster, so the stores themselves are the cost |

**Ablations** (`ssm_in`, 2048 rows, output hash equal unless noted, interleaved
in one process; this loop's schedule is fragile, so each variant's ISA and VGPR
count were checked):

| Variant | Result |
| --- | --- |
| Plane-aware swizzle (retained) | +3.0% to +3.4% |
| 128-byte-row LDS layout; the same with 8-lane coalesced activation loads | +3.1%; +2.4% (the swizzle's gain, coalescing adds nothing) |
| Global prefetch two stages ahead | +3.2% (swizzle alone +3.0%) |
| One to three token tiles of LDS fragment prefetch | ±0.2%: the compiler sinks the loads back to their use. The 2026 campaign's forced double buffer (221 → 253 VGPRs) was 3–14% slower |
| Aligned weight blocks without the scale load | +1.1% |
| Branch-free or unguarded fetch | −13%: the scheduler hoists the stage commit next to the loads and exposes their latency |
| Fetch spread across the K loop | 256 VGPRs with spills, −63% |
| LDS-only barriers (no `vmcnt(0)` or L0 invalidate); `s_setprio` | 0%; −1% |
| Half a stage of start delay for every second block of a WGP | +0.1% to +0.5% (noise) |
| Other tiles against the swizzled 8×1 BK 2: 4×2 BK 2, 4×2 BK 1, 128×128 BK 4, 128×256 | −2%, −16%, −15%, −13% |
| Swapped WMMA operands with direct coalesced output stores | bit-identical (the dot product is symmetric in A and B) but −1% |
| Zeroing the upper 16 lanes of an operand | changes the result: both lane halves are read |
| Constants instead of loaded data | +12% to +20%, an artifact of the clock; +4% with pseudo-random words |

Reproduction: `tools/prof/pmc.py run --kernel DenseF16GEMMKernel --skip 3 --out
/tmp/pmc -- <bench>` with the environment above plus `DENSE_BENCH_REPS=1
DENSE_BENCH_LAUNCHES=4`. The routed GEMMs also show LDS bank conflicts
(`moe_bench --rows 2048 --paths wmma`, layer 3: 4.5% of LDS cycles for the Q4_K
gate/up, 12.1% for the Q5_1 down projection); their layout has four LDS arrays
and was not changed.

## NVIDIA FP4-QAT checkpoint (2026-10-24)

Rejected for now. NVIDIA's quantization-aware checkpoint
`nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a` was converted to INT4 using the
baseline's exact tensor types, so only the weights differed from UD-Q4_K_XL.
It finished outside the accepted quality envelope and gave no coding-goodput
win. Because quality failed, the abliteration adapter was not run against it.

**Conversion.** `tools/qwen-flash/convert_nvidia_fp4_local.py` produced
the artifact; it is now pinned to the revision's index SHA-256 and the git
blob of its `config.json`. Three defects were found and fixed before any
result counted:

- **PLE scale.** The global scale for the per-layer embedding (PLE) is one
  module-level `ngram_embedding.weight_scale_2`, not one scale per shard. The
  old silent fallback to 1.0 made the table about 30,000× too large.
  Decoded rows now have RMS 0.0075, matching NVIDIA's dequantization, and a
  missing global scale is an error.
- **Compression ratios.** transformers 5.19 relabels `full_attention` layers
  as `indexed_attention`. The pinned `qwen4exp.py` then writes all-zero
  `attention.compress_ratios`, and Gufo refuses to load the model. The
  converter now reads `config.json` directly and verifies the emitted ratios.
- **Tensor types.** `Q4_K_M` with regex overrides gave many non-expert
  tensors other types than the loadable baseline. Examples: Q4_K SSM α/β and
  HC inject, and Q8_0/Q5_0 routed down-projections. The converter now mirrors
  a reference GGUF's per-tensor types (`--type-reference`) and verifies exact
  equality.

The artifact under test was re-quantized on the CPU from the BF16 file with
its ratios patched in place. All 1224 tensor types match UD-Q4_K_XL, and the
file is 111.33 GB. Its INT4 SHA-256 is `e9d1d57b…0512`; the manifest records
the provenance of the patch and the requantization.

**Quality.** The ladder compares against a fresh UD-Q4_K_XL run in the same
window, scoring forced public-domain text:

| Metric | NVIDIA INT4 | Accept threshold |
| --- | ---: | ---: |
| Decode KL | 0.078 | ≤ 0.057 |
| Decode top-1 agreement | 92.8% | ≥ 94% |
| Perplexity delta | +2.21% (2.124 vs 2.078) | ≤ +2% |

Prefill KL was 0.196 with 88.9% top-1 agreement over 27 cases. The result
clears the gross-failure gate (KL ≤ 0.25) but misses all three accept limits.

**Throughput.** Median prefill is at parity with identical tensor types:

| Prompt tokens | NVIDIA INT4 | Fresh baseline |
| ---: | ---: | ---: |
| 2048 | 1383 tok/s | 1362 tok/s |
| 8192 | 1333 tok/s | 1332 tok/s |

The archived 2026-10-23 baseline ran about 5% faster than today's fresh
baseline (1448 and 1400 tok/s), so compare only runs from the same window.

**Coding swarm.** 32 agents, 900 s budget, 300 s warmup, the pinned code
queue:

| Run | Correct/h (CI95) | Success | Unfinished | Invalid tool calls |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA INT4 | 176 (124–228) | 93.6% (44/47) | 29 | 0.29% |
| NVIDIA INT4, replicate | 140 (94–186) | 97.2% (35/36) | 29 | 0.35% |
| Fresh baseline | 124 (80–168) | 100% (31/31) | 31 | 0% |
| Archived baseline | 164 (114–214) | 100% (41/41) | 28 | 1.05% |

Both candidate runs beat the fresh baseline. The replicate falls below the
archived baseline, and the intervals overlap throughout, so this is not a
win. The candidate also failed 4 of 83 graded tasks, while the baselines
failed none of 72.

**Adapter.** The huihui steering adapter (α 1.9553, full scope) is
architecturally compatible: the hidden size is checked and it acts on
activations. `stages-nvidia-adapter.sh` holds the gated validation, which
covers refusal rate, forced KL, prose perplexity and cache isolation. It
stays unrun until a checkpoint passes the quality gate.

**Artifacts.** The ladder, gate, prefill and swarm results, both manifests
and the three window logs stayed on the measurement host with the tested
GGUF and are not part of this tree; the intermediate BF16 and first `Q4_K_M`
files were deleted after the run.

Next, if this is revisited: the swarm needs more replicates before a
goodput claim, and the measured KL/top-1 gap points to real weight
divergence, not tensor types.

## How much of the model can run on INT4 hardware (2026-10-24/25)

The question was how much of Qwen3.8 Flash-Next can move onto gfx1151's
4-bit integer WMMA (iu4, 110.9 TOPS, 2× fp16) to offset its limited
memory bandwidth while keeping quality inside the accept envelope: decode
KL ≤ 0.057, top-1 ≥ 94%, perplexity delta ≤ +2%.

**Answer.** Prefill can put every dense and routed projection on 4-bit
activations at acceptable quality, but it gains only a few percent. Most of
that work is limited by weight streaming, not arithmetic.

Single-session decode is bandwidth-bound, so the lever there is fewer
weight bits. 4-bit dense weights cost too much quality. A per-family mixed
layout (`nv-dense-mixed`) is inside the envelope and gives +16% single-user
decode, but it costs 13% prefill and brings no gain at 8–32 sessions.

No layout here is a clear win for Gufo's multi-session agent workload.

### Method

- **Quality.** Every result is a teacher-forced ladder on public-domain
  text (`tools/qwen-flash/ladder.py`), compared with the same
  model's run without activation or weight emulation.
- **Emulation knobs.** The spike emulation (`spike.hip.cpp`) gained
  `MIN_ROWS`/`MIN_ROWS_MODE` (prefill-only quantization), a separate routed
  recipe, clip search, outlier retention, an FP4 (E2M1) grid, WBITS 5/6 and
  a decode-GEMV hook. It also gained `JITTER`, a seeded perturbation of the
  quantization scales by a few ulps.
- **Error bars.** A one-ulp scale change gives a different but equally
  valid 4-bit result: decode KL 0.012 between the two runs and a PPL swing
  of 1.2 points. So the decision recipes were each run with three jitter
  seeds, and results below show mean [min, max].
- **Corrections to earlier rungs.** Before this work, `attn_out` and the
  shared expert's gate/up never reached the spike (a routing test in
  `Executor::Dense`), and decode never used emulated weights. The
  10-22/23 "all dense", "W4-only", "mix" and "minrows" rungs therefore
  measured less than their labels say. The deployed binary also silently
  ignored `MIN_ROWS` and `ROUTED_*`.

### Prefill: 4-bit activations (W4A4)

Each family alone stays inside the envelope: decode KL 0.011–0.021 and
top-1 ≥ 95.8% on NVIDIA QAT weights. Stacked results are three-seed
ensembles on NVIDIA QAT weights:

| Recipe | Decode KL | Top-1 | PPL delta |
| --- | ---: | ---: | ---: |
| Every family W4A4, prefill batches only (≥ 2048 rows) | 0.0171 [0.0166, 0.0180] | 96.4% | +0.88% [+0.52, +1.39] |
| The same without `attn_out` | 0.0152 [0.0145, 0.0161] | 96.8% | +1.40% |
| Routed experts only, activations in groups of 16 | 0.0132 [0.0132, 0.0133] | 96.8% | +0.13% |
| UD-Q4_K_XL, every family, prefill only (one seed) | 0.0181 | 96.7% | +0.14% |

- **INT4 vs FP4 grid.** An integer grid loses almost nothing against the
  FP4 (E2M1) grid NVIDIA trained for: routed decode KL 0.0128 for int4 g16
  against 0.0119 for NVFP4.
- **QAT.** NVIDIA's QAT lowers the damage only modestly compared with
  UD-Q4_K_XL.
- **Most sensitive family.** `attn_out` is the most fragile.

**Speed.** The real integer kernels pass correctness: relative RMS about
5e-4 against the emulation, and 0 failures in 192 in-engine checked calls.
They give little speed:

- **Routed experts** are 44% of a step. At 2048-token chunks each expert
  sees about 40 rows, so the FP16 route already runs at 24–33 TFLOPS and
  streams weights.
  - The probe measured iu4 at 1.08× at 2048 rows, 1.22× at 8192 and 1.27×
    at 16384 rows and above. Per-32-group scales lose most of that.
  - End to end, prefill rose +3.6% (g0), +2.1% (g32) and −0.4% (g64),
    and quality fell (g0 decode KL 0.039).
  - Larger prefill chunks don't help: 2048 stays the fastest chunk, and
    iu4 gains at most 0–2% at any chunk size.
- **Rotated experts.** Experts stored in a Hadamard-128 basis
  (`nv-exp-had128`) cut per-row int4 activation damage by 4×: prefill KL
  0.428 → 0.106 and decode KL 0.039 → 0.021. Speed stays at +2.9%.
- **Dense projections** (25% of a step) remain the only clear iu4 win:
  about 1.33× on their GEMMs, or roughly 6% of step time (window
  2026-10-22).

### Decode: weight bits

At one session a decode token reads 6417 MB. Dense Q8_0 tensors are about
70% of that: GDN projections 34%, HC 11%, the head 10.5%, attention 10.5%
and the shared expert 4%. Routed experts are 23%. Measured tg128 is
25.6 tok/s.

Emulated decode-weight bits per family (Hadamard-128, group 32):

| Family | MB/token saved at 4-bit | W4 decode KL / PPL | W5 | W6 |
| --- | ---: | ---: | ---: | ---: |
| ssm_in | 755 | 0.024 / +0.83% | 0.013 / +1.09% | 0.010 / +0.63% |
| ssm_out | 283 | 0.026 / +3.74% | 0.013 / +0.85% | 0.009 / +0.18% |
| attn_qkv | 204 | 0.013 / +1.46% | 0.010 / +0.08% | 0.009 / +0.65% |
| attn_out | 94 | 0.036 / +2.18% | 0.017 / +2.40% | 0.010 / +0.80% |
| shared | 79 | 0.013 / +0.90% | 0.009 / +0.64% | 0.009 / +0.64% |
| all five | 1415 / 1061 / 686 | 0.073 / +6.53% | 0.023 / +2.61% | 0.014 / +2.36% |
| all five, no rotation | 1415 | 0.110 / +18.3% | | |

Real K-quant GGUFs, unrotated, with the head at Q6_K:

| Variant | tg128 | pp512 | C8 / C32 decode | Decode KL | Top-1 | PPL delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA basetypes (Q8_0 dense) | 25.4 | 1150 | 140 / 263 | — | — | — |
| Dense Q6_K | 28.0 (+10%) | 881 (−23%) | 143 / 268 | 0.0143 | 97.4% | +1.41% |
| **Dense mixed** | **29.5 (+16%)** | 999 (−13%) | 131 / 266 | **0.0167** | **97.1%** | **+1.68%** |
| Dense Q5_K | 29.4 (+16%) | 1017 | 128 / 265 | 0.0292 | 95.4% | +5.02% |
| Dense Q4_K | 31.3 (+22%) | 1032 | — | 0.078 | 92.6% | +6.63% |
| Dense Q4_K + IQ4 experts | 32.3 (+26%) | 1057 | — | 0.087 | 93.0% | +7.48% |
| Dense Q4_K + IQ3_S experts | 31.7 (+25%) | 1039 | 126 / 262 | 0.112 | 90.5% | +13.1% |

The mixed layout follows the per-family matrix: ssm_in Q5_K, attn_qkv and
shared gate/up Q4_K, ssm_out Q6_K, attn_out kept at Q8_0, and the head at
Q6_K (5277 MB/token).

- **Quality.** Only Q6_K and mixed are inside the envelope.
- **Prefill.** Every K-quant dense layout costs 12–23% prefill against
  the tuned Q8_0 routes.
- **Concurrency.** At C8 and C32, weight reads are shared across sessions,
  so fewer bits stop helping. C8 even regresses for Q5_K, mixed and
  IQ3_S.
- **Experts.** Lowering expert bits barely moves decode: IQ3_S experts add
  +1% over Q4_K dense alone, at a large quality cost.

### Open leads

- **Dual-format dense weights.** Keep Q8_0 for prefill and a mixed
  K-quant copy for decode. Built since as the decode twins
  (`--gguf-decode-override`, +1.9 GB; see the speed campaign below).
- **Rotated dense K-quants.** Store dense weights in a Hadamard basis and
  rotate decode activations to match. In emulation this halves 4-bit
  damage (all-five W4: decode KL 0.110 → 0.073). It needs a real-kernel
  activation rotation and a rotated-weight writer; the writer exists for
  experts.
- **Multi-session decode** needs its own profile. Neither lever moves C32.
  Half-precision GDN state (7.2 GB/step at C32 in F32) was built since as
  `--gdn-state f16` (C32 +13%, C64 +19%; see the speed campaign below).

### Artifacts

Raw results and the window scripts that produced them stayed on the
measurement host and are not part of this tree; the tables above carry the
retained numbers. Only `nv-dense-mixed` and `nv-dense-q6k` variant GGUFs were
kept.

## Speed campaign (October 2026)

One research lane per phase (decode C1, multi-session decode, dense prefill,
routed prefill, prefill mixers, speculative policy, dense-weight calibration,
expert pruning, adapters, power, fault hunt), each measured against the same
baseline (UD-Q4_K_XL, tq4 KV), then merged into one tree and re-measured as a
whole. The lane handoffs and window scripts are not part of this tree; the
retained per-window outputs are in [artifacts/campaign](artifacts/campaign).

Rules the lanes worked by: an exact (bit-identical) change becomes the default;
a lossy one must pass the quality envelope (decode KL ≤ 0.057, top-1 ≥ 94%, PPL
≤ +2%) and, in the integration, the prefill-row budget (KL ≤ 0.1, top-1 ≥ 90%)
and stays behind one explicit option. Speed is interleaved A/B with a
sustained second pass (the 140 W burst lasts about 110 s; the sustained
package limit is 120 W).

### Retained, exact (default path)

| Change | Evidence |
| --- | --- |
| Prefetching small decode GEMMs, register-resident HC combine/mix, GDN epilogue that writes Q8_1 for the next projection | Logit hash and `DecodeC1b` identical; lane C1 tg 26.32 → 28.33. Integrated default measured +6.7% tg128, +4% pp2048. |
| The prefetch kernel only at one or two rows | The first integration measured C8 distinct-prompt decode 84.5 against base 96.8 tok/s (−11%); toggling isolated the prefetch kernel at more than one row (C2 43.6, C4 64.4, C8 83.0 with it; 42.4 / 66.0 / 93.2 without). Both kernels are bit-exact, so limiting it changed no output. After the fix C8 is +2.3% over base ([cohorts](BENCHMARKS.md#concurrent-decode-distinct-prompts)). |
| tq4 KV-store epilogue in the fused attention projection | Compile-time tile switch avoids scratch spills; kernel hash, `session_test` and `--validate-prefill` identical; fused projection 7323 → 3907 µs, pp2048 +1.4%, pp8192 +3.1%, d32K +3.6%. |
| Block-selection windows in `SelectBlocksBatch` (FaultHunt) | Production bug, below. |
| Combined greedy C1 MTP policy (`--mtp-policy adaptive`, 32K compact draft head, calibrated gate, longest-match lookup) | Exact: greedy tokens equal AR, sampled requests untouched. [Speeds](BENCHMARKS.md#single-user-mtp-adaptive-policy). |

### Retained, lossy opt-in

| Change | Quality | Speed (A/B) |
| --- | --- | --- |
| `--dense-prefill w4a4` + offline GPTQ S4G128 weights, ≥ 2048 rows | KL 0.011 / 0.005, top-1 98%; a 256-row threshold failed (PPL +2.12%) | pp2048 +15% sustained, pp8192 +15%, d32K +14%; dense set 402 → 205 ms (kernel sum) |
| `--gdn-state f16` | KL 0.007 / 0.003; 64K prefill KL 0.009 | distinct C8 +8%, C32 +13%, C64 +19%; identical-prompt C32 +20%; 32K-prefix C32 +11% |
| `--gguf-decode-override` (GPTQ Q4_K dense + Q5_K head twins, ≤ 4 rows) | KL 0.027 / 0.006; prefill rows untouched | C1 tg +22% alone (32.42 vs 26.53), C2 +12%, C4 +2%, C8/C32 equal; +1.9 GB. The same GPTQ tensors through `--gguf-override` cost pp512 −13% (MMQ route), which is why they are decode twins. |
| `--prefill-capacity 8192 --prefill-phase-rows 2048` | exact per session (KL 0 at 2K/16K/65K) | pp8192 +5%, pp32768 +6%; ~4 GB more scratch [INFERENCE]; prompt steps ~4× longer (ITL policy). Not bit-exact for a packed multi-request prompt batch (23/28 identical). |
| Stack of the four above | Passes decode envelope and prefill-row budget (KL 0.031 / 0.010, prefill rows 0.050 / 0.066) | pp2048 +17%, pp8192 +24%, tg128 +37%, d32K pp +13.5%, d32K tg +36%; C32 +13%; see [Benchmarks](BENCHMARKS.md#speed-campaign-stack-october-2026) |

### Rejected or not shipped

| Idea | Outcome and why |
| --- | --- |
| Top-k routing for agents (`--experts-used 5/7/8`, healed top-5, `--router-norm-k 10`, affinity) | Task-level A/B (96 pinned coding tasks × 32 agents, three UD replicates as the noise floor): UD finished 87% of tasks, top-8 70%, top-8 + f16 + affinity 75%, top-7 65% (accuracy 77.6%, p = 0.004), healed top-5 66%, top-5 62%; turns per task 9.9 → 13.2 for top-5. Decode is faster (UD top-5 C32 +22%) but extra turns cancel it on a prefill/tool-bound swarm. Top-8 matches accuracy but gives no end-to-end gain. Does not ship; top-8 stays an opt-in for decode-bound use. Healing could not close top-5 (books KL 0.138). [Quality](QUALITY.md#agent-task-quality-top-k-and-adapters). |
| `--routed-w4` (symmetric int4 routed experts, decode too) | Decode KL 0.036 / 0.020 (in envelope) but prefill rows KL 0.41 / 0.30, top-1 74% / 79%: fails the prefill-row budget. With macro batching it was the fastest single lever (pp8192 +16.6% at g128) and the stack including it reached pp2048 +38%, pp8192 +38%, d32K +32%, tg128 35.2; not recommended. |
| `--prefill-residual f16` (F16 HC residual streams between wide combines) | Decode KL 0.009 / 0.006 but prefill rows top-1 87.5–88.9% (KL 0.035 / 0.048): below the 90% budget. pp2048 +4.3%, HC bytes −30.5%. The recommended stack leaves it out (q1 with it: books prefill top-1 85.2%). |
| Chunked GDN prefill (`GUFO_GDN_PREFILL`) | Kernel 1.19× but +1.4% end to end, no ladder; not integrated. |
| AutoRound int4 g128 routed experts (Intel W4A16-AutoRound) | Offline error against the BF16 experts: gate/up 2.1–3.9× and down ~3× UD's (weight RMS 0.107–0.119 vs 0.072–0.078; equal to RTN g128). The GPU window's quality arms ran UD only (GSM harness broke). Closed on offline evidence. |
| Power and clock knobs | Package power bound at 120 W: raising fclk to its maximum or capping the CPU measured nothing (pp2048 1572 and 1575 against 1569–1574 for the base segments); an `sclk` cap of 2400 MHz cost −3.6% and an `sclk` minimum of 2700 MHz −9%. Raising the limit needs a signed `ryzen_smu` and ryzenadj on a secure-boot host (+3–5% [INFERENCE], 4–5 °C headroom); an undervolt is +1–3% with numerics risk. Owner decision, nothing applied. |
| decode-c1 follow-ups (`HcDownSplit`, graph fork/join, top-k folded into the plan, MoE in the combine) | `HcDownSplit` not bit-exact at T=1; MoE-in-combine −1.6%; graph branches and top-k-in-plan +1% each, not separable. Left out. |
| `--gdn-state q8` | Fails prefill at 64K (KL 0.21); not shipped. `bf16` is accepted but was not qualified. |
| `--moe-routing affinity:6:1` stacked with top-8 and f16 state | +31% C32 decode in a same-window A/B, but the swarm arm with it finished 75% of tasks against UD's 87% (accuracy 88.9%, no extra loss from co-batching). Not integrated. |
| 32K compact draft head alone | Exact, but agent file edit −17% (A/B); superseded by the combined policy. |

### Production bug found by the fault hunt

"Expert counts: an illegal memory access" is only the first host sync after
queued kernels, so any earlier kernel can be the culprit. The hunt found two
unrelated faults. `SelectBlocksBatch` (the multi-session attention path used by
fused and packed prefill) indexed its block-score scratch by the packed batch
row. The scratch holds 512 rows at the model's 262144-token context and 1024 at a
131072-token session, but a fused prompt step packs up to 2048 rows, so a
sparse (position > 2048) item at packed row ≥ 1024 wrote up to ~128 MiB past
the allocation. It is reachable with two or more concurrent prefills or a long
uncached tail fused with decoders once `--max-itl-ms` adapts the budget up, and
the production build (`52ce2264`) has it; no fault or output divergence was
observed there (the overrun lands in whatever scratch follows), so the risk is
silent corruption or a rare fault. The fix scores the packed rows in windows of
the scratch's capacity (one window is the old launch sequence, so the exact path
is unchanged); `attention_ops_test` `CheckSelectBatchRowCapacity` goes
red → green (red: "packed block selection wrote 891184 score words past its
1024-row scratch"). It is in this tree (`21e68c8`) and should go to production
independent of the rest. The second fault, in `gpu_probe` (a forward on a cold,
non-resident session), was not a production path; `Executor::Forward` now
rejects non-resident sessions with an error instead of a GPU fault.

### Integration findings

| Finding | Evidence and resolution |
| --- | --- |
| MTP drafts almost never accepted on the stack (prose 3/109, code 11/128) | Bisected by arm (`w3f`: macro, int4 dense and F16 state keep 40–50% acceptance, the decode twins collapse it) and by `--decode-twin-rows` (`w3g`: 0 → 125/295, 1, 2, 8 → 2–3 of 119–135). `Executor::LoadDraftVocabulary` copied the head descriptor, including its pointer to the token-id-indexed Q5_K twin head, so one-row draft rounds scored ids 0–32767 instead of the ranked vocabulary. Fixed (`draft_head_.alt = nullptr`); stack MTP prose 27 → 42 tok/s ([Benchmarks](BENCHMARKS.md#single-user-mtp-adaptive-policy)). A plain full-width draft head never had the problem. |
| MTP output on the stack is not bit-identical to the stack's AR | `w3i`: `--decode-twin-rows 8` (every verify chain reads the twins) did not restore equality or run-to-run determinism and cost 2–8% on four workloads; `w3f`: `--gdn-state f16` alone already differs between MTP and AR (prose and code hashes), because the recurrent state is rounded per verified chain instead of per token. With twins alone and rows 8, MTP equals AR (`w3g`). Documented; default `--decode-twin-rows 4` kept. |
| First-turn TTFT in `agent_sim` varies 8–19 s between runs | Five stack/default runs against two base runs, no flag-correlated pattern (`w5c`–`w5e`); noise at this sample size, not shown to be a regression. |
| `session_test` fails "interleaving changed MTP proposals or acceptance" | **Not a regression.** Greedy draft widths come from `MtpBatchController`: one model-wide object whose step-cost model is a recursive least squares fit of measured wall time, plus a cycle counter (a plain step every 32nd cycle, exploration every 32nd+16th). Another request's cycles, earlier steps and the clock all move it, so an isolated replay of a greedy request, or a serving request against a direct session, legitimately picks other widths while the tokens stay exact. The per-session plan of a *sampled* request depends on committed history only, and the corrected test asserts exactly that. Evidence: the unmodified test fails the same assertion on the **base tree** (`w1h` base-1/-2, base HEAD `d26627d`), the replay widths are identical run to run while the first run's widths vary (`w1e` strict-1 vs strict-2), and with the greedy assertions relaxed the sampled configurations (depth 16 and 4096, four sampler configs) and the fresh-load bitwise replays pass. The speculative merge did not introduce it. |
| `session_test` batched sampled MTP pair; mixed-step decoder KL | Also failing on base (`w1i`: top-p / min-p, batched-w2, same draft counts but different tokens): a sampled MTP request in a batched step verifies on the shared tiled routes, which are not bit-identical to serial (QUALITY, "Wide batched rows"). The check now requires progress and budget for that case only. The mixed-step decoder frontier KL depends on whether any decoder drafted: 0 in both base runs (no drafts); on the integrated tree 0 in two runs, 0.016–0.022 in five and 0.37 in one (`w1g` fixed-2); 0.18 on the pre-merge tree and 0.23 with every draft width forced to 3. The bound is twice the documented wide-cycle bound. |
