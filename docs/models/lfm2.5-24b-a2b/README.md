# LFM2.5-24B-A2B

Liquid AI's LFM2.5-24B-A2B (`lfm1.0` licence) is a 24B-parameter mixture of
experts with 2.45B active per token, 131,072-token context and a 128,000-token
vocabulary. It always reasons: the template opens `<think>` for every
assistant turn, so responses carry `reasoning_content` even when a request
turns thinking off. Its Pythonic tool calls
(`<|tool_call_start|>[name(arg='v'), other(n=1)]<|tool_call_end|>`, parallel
calls in one list) return as OpenAI `tool_calls`. Gufo runs it on the
Flash-Next engine (same batching, lazy KV pool, paging, snapshots and prefix
cache).

Status: experimental, built in this tree on the Flash-Next engine. Parity with llama.cpp is in
[Quality](QUALITY.md); speeds in [Benchmarks](BENCHMARKS.md).

## Architecture the engine handles

| | Flash-Next (`qwen4exp`) | LFM2.5 (`lfm2moe`) |
| --- | --- | --- |
| Residual | 4 hyper-connection streams | one stream, `attn_norm` / `ffn_norm` / `token_embd_norm` |
| Linear layers | gated DeltaNet | double-gated short convolution: `in_proj` → B, C, x; y = C · conv3(B · x); `out_proj` |
| Attention | gated GQA, 24 q / 2 kv heads of 256, QSA indexer | plain GQA, 32 q / 8 kv heads of 64, QK RMS-norm, NEOX RoPE, no gate, dense |
| Layer pattern | attention every fourth layer | attention on layers 2, 6, ..., 38 (from the per-layer KV head count) |
| FFN | 512 experts × 640, top-10, shared expert | layers 0–1 dense SwiGLU (11,776); then 64 experts × 1,536, top-4, no shared expert |
| Router | softmax top-k | sigmoid scores, top-4 chosen on score + bias, weights renormalized |
| Embeddings | separate head | tied |
| Chat template | Qwen3.8 | ChatML with `<\|startoftext\|>`, Pythonic tool calls, `<think>` opened |

The short convolution keeps two rows of B · x per layer (2 × 2,048 floats) as
its whole recurrent state. Attention is dense: decode reads every cached key,
so per-token cost grows with context; the new attention kernel reads each
KV tile once for the four query heads that share it.

New kernels (`kernels/rocm/lfm.hip.cpp`): token-parallel short convolution
with a carried state, the attention prepare step (QK norm, RoPE, KV store),
GQA-4 flash attention for 64-wide heads (WMMA matrix cores for prompts,
eight key splits for decode, multi-session items), TurboQuant tq4/tq3 for
64-wide rows, and the sigmoid-plus-bias router. The routed-expert kernels
gained the 2,048 × 1,536 geometry.

## Weights

The pinned repository is `LiquidAI/LFM2.5-24B-A2B-GGUF@0912d331`. Its `Q8_0`
loads directly. The serving build quantizes the `BF16` file into the
Flash-Next layout (Q4_K routed gate/up, Q5_1 routed down, everything else
Q8_0, 15.5 GB):

```sh
llama-quantize --tensor-type ffn_gate_exps=q4_k --tensor-type ffn_up_exps=q4_k \
  --tensor-type ffn_down_exps=q5_1 LFM2.5-24B-A2B-BF16.gguf \
  LFM2.5-24B-A2B-Q4K-Q51down-Q8.gguf Q8_0
```


## Serve

```sh
./result/bin/gufo serve --sessions 256 llm \
  --model models/lfm2.5-24b-a2b/LFM2.5-24B-A2B-Q4K-Q51down-Q8.gguf \
  --served-model-name lfm2.5-24b-a2b --context 32768 --kv-cache f16
```

Up to 256 sessions decode in one step: at that width every step reads all
the routed experts anyway, so throughput grows with sessions (see
[Benchmarks](BENCHMARKS.md)). Use f16 or `tq8` KV: `tq8` halves the cache at
no measurable cost, while the model is unusually sensitive to 4-bit KV (see
[Quality](QUALITY.md)).

Liquid's `QAD-Q4_0` file also loads (dense matrices and down experts are
re-encoded exactly as Q8_0 and Q5_1, gate/up experts stay Q4_0), but it is
further from the full model than the requant above.
