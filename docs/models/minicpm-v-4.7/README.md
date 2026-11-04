# MiniCPM-V 4.7 35B-A3B (text)

OpenBMB's MiniCPM-V 4.7 35B-A3B (Apache-2.0, released 2026-10-12) pairs a
SigLIP2 vision encoder with a `qwen35moe` language model: the Qwen3.5-35B-A3B
geometry of 40 layers (Gated DeltaNet with gated attention every fourth layer),
256 experts with 8 routed plus one shared, hidden 2048, 262,144-token native
context. Gufo runs the language model on the Flash-Next engine (same
batching, lazy KV pool, `tq4`/`tq3` KV, paging, disk cache and flex
scheduling). Images go through llama.cpp's mtmd, linked from a pinned upstream
build: it decodes and slices the image, runs the SigLIP2 encoder, merger and
projector, and plans the special tokens and canvas M-RoPE positions; Gufo
injects the embeddings and applies those positions. OpenBMB's DSpark drafter
runs speculative decoding (see [DSpark](#dspark)).

Status: experimental, built in this tree on the Flash-Next engine. Teacher-forced parity with llama.cpp on
the same GGUF is in [Quality](QUALITY.md); speeds in [Benchmarks](BENCHMARKS.md).
It decodes about 2.2× faster than Flash-Next on short contexts, but attention is
dense and decode attention re-reads each KV head once per query head, so above
roughly 10K tokens per session at high concurrency Flash-Next is faster. A
grouped decode kernel is the next step.

## Differences from Flash-Next the engine handles

| | Flash-Next (`qwen4exp`) | MiniCPM-V 4.7 (`qwen35moe`) |
| --- | --- | --- |
| Residual | 4 hyper-connection streams | one stream, `attn_norm` / `post_attention_norm` / `output_norm` |
| Attention | QSA indexer above 2,048 tokens | always dense (no indexer, no mask) |
| DeltaNet output gate | `sigmoid(z)` | `silu(z)` (Qwen3.5) |
| Experts | 512 × 640, top-10 | 256 × 512, top-8 |
| Heads | 24 q / 2 kv; 48 DeltaNet value heads | 16 q / 2 kv; 32 DeltaNet value heads |
| Chat template | Qwen3.8 (effort instruction) | Qwen3.5 (no effort instruction) |

`Config` sets `hc_count` 1 and no indexer for `qwen35moe`; mixers become the
block pre-norm and combines a residual add. The device-planned routed-expert
kernels and the dense WMMA prefill attention are compiled for both geometries.

## Weights

The pinned repository is `openbmb/MiniCPM-V-4.7-35B-A3B-gguf@3c3f7e84`. Its
`Q4_K_M` and `Q8_0` load directly. Dense Q4_K/Q5_K/Q6_K projections (the
attention, DeltaNet and shared-expert matrices, the LM head and the embedding)
run the vendored MMVQ dot products in decode and the MMQ tiles in prefill;
Q4_K and Q6_K routed down experts, which only llama.cpp's K-quant mixes use,
are re-encoded at load into the Q5_1/Q8_0 the routed kernels read. The
serving build requantizes `Q8_0` into the Flash-Next layout (Q4_K routed
gate/up, Q5_1 routed down, everything else Q8_0, 22.8 GB) with llama.cpp:

```sh
hf download openbmb/MiniCPM-V-4.7-35B-A3B-gguf --revision 3c3f7e84f14a760ff26d17ab5932d4aa996227c0 \
  --include MiniCPM-V-4_7-35B-A3B-Q8_0.gguf --local-dir models/minicpm-v-4.7
llama-quantize --allow-requantize --tensor-type ffn_gate_exps=q4_k --tensor-type ffn_up_exps=q4_k \
  --tensor-type ffn_down_exps=q5_1 models/minicpm-v-4.7/MiniCPM-V-4_7-35B-A3B-Q8_0.gguf \
  models/minicpm-v-4.7/MiniCPM-V-4_7-35B-A3B-Q4K-Q51down-Q8.gguf Q8_0
```

Adding `--output-tensor-type q6_k` stores the LM head as Q6_K: single-stream
decode is 3% faster at unchanged quality; Q6_K for every dense projection
gains 11% but costs prefill time until the K-quant WMMA prefill route lands
([benchmarks](BENCHMARKS.md#k-quant-weights), [quality](QUALITY.md#quantization-against-the-q8-original)).

## Images

Build with `-DGUFO_LLAMA_CPP_DIR=/path/to/llama.cpp` (a tree at or after
`69f201a2`, *model : support MiniCPM-V 4.7*, with a HIP build in
`build-hip`), then add the projector:

```sh
./result/bin/gufo serve --sessions 64 llm --model ...Q4K-Q51down-Q8.gguf \
  --mmproj models/minicpm-v-4.7/mmproj-model-f16.gguf --served-model-name minicpm-v-4.7 --context 32768 --kv-cache tq4
```

Requests send OpenAI `image_url` parts (data URLs). Each image becomes mtmd's
media marker at its position in the message, exactly as llama-server joins
content parts, so prompts tokenize identically. Image tiles share one KV
position each and use `zyxt` M-RoPE sections; Gufo keeps them as explicit
per-token positions, which snapshots persist, so image conversations continue
from cache.

## Serve

```sh
./result/bin/gufo serve --sessions 64 llm --model models/minicpm-v-4.7/MiniCPM-V-4_7-35B-A3B-Q4K-Q51down-Q8.gguf \
  --served-model-name minicpm-v-4.7 --context 32768 --kv-cache tq4 --think off
```

Add `--speculative dspark --dspark-model
models/minicpm-v-4.7/dspark-MiniCPM-V-4_7-35B-A3B-F16.gguf` for speculative
decoding.

Thinking defaults to on (the template opens `<think>`); send
`chat_template_kwargs.enable_thinking=false` or start with `--think off` for
direct answers. `reasoning_effort` has no effect on this template.

## DSpark

OpenBMB ships a DSpark drafter (`dspark-MiniCPM-V-4_7-35B-A3B-F16.gguf`,
architecture `dflash`): five non-causal transformer blocks (hidden 2048, 8 query
and 2 KV heads of 256) that read the target's token embedding and LM head, a
rank-256 Markov head and a confidence head. Gufo follows llama.cpp's DSpark
semantics (`src/models/dflash.cpp`, `common/speculative.cpp`):

- The inputs of target layers 4, 12, 20, 28 and 36 are concatenated, fused by
  `fc` and RMS-normalized, and become each draft layer's K/V at the token's
  position (full-width NEOX rotary, theta 1e7). Every prompt chunk, decode
  step and verified prefix is injected; rejected rows never are.
- One pass over `[anchor, MASK, ...]` at the next positions attends to the
  whole context and the whole block. Row *i* proposes the token after it
  (`sample_from_anchor`): base logits from the target head plus
  `W2 · W1[previous]`, chained greedily from the anchor. The Markov bias is
  scored on each row's top-256 base logits.
- `conf_i = sigmoid(conf · [h_i; W1[previous]] + b)` is calibrated as the
  conditional acceptance of proposal *i*; the verified length maximizes
  expected tokens per millisecond under a gfx1151 cost table
  (`dspark_policy.hpp`), and two unproductive passes in a row back off to
  plain decoding for 2–32 tokens. Beside other sessions a session drafts
  only while its running mean of accepted drafts per pass pays for the
  pass (probing every 128 tokens), so unpredictable prose stops drafting.
- Drafts are deterministic, so verification draws the target exactly
  (point-mass speculative sampling): greedy and seeded sampled output equal
  plain decoding at one session. Verification uses the MTP path's at most
  eight rows (seven drafts).
- Drafter projections are transcoded to Q8_0 at load; sessions decoding
  together draft in one pass and inject their rows in one stacked K/V
  GEMV plus one store launch. Draft K/V (F16, 10 KB per token) is part of
  session snapshots (format 17).
- A request's last token is selected but not evaluated, as plain decoding
  leaves it, so speculative and plain sessions cache the same prefix and
  later turns replay identically.

Image prompts (planned M-RoPE positions) decode without drafts; without
`--mmproj` the server refuses image input either way.
`qwen38_flash_next_dspark_probe` drafts every position of a greedy
continuation for acceptance studies; the session test covers DSpark with
`--mtp-model none --dspark-model PATH`.
