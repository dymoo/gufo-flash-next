# Fork status

This repository is a **hard fork** of [gufo-org/gufo](https://github.com/gufo-org/gufo).
It is not a git fork: the tree was re-rooted from a working snapshot of the
fork's campaign tree on 25 October 2026 (`campaign base: gufo-int8 working tree`),
so there is no merge base with upstream. Upstream changes are ported by hand
and recorded below; the fork will keep diverging.

## Why it diverged

The fork tunes Qwen3.8 Flash-Next for many concurrent coding agents on one
Strix Halo: steering adapters (abliterated variants served from the base
weights), the flex service tier and router contract, quantized and paged KV,
and a set of measured speed options. Upstream tunes the exact single-request
path and many more models. Both sides rewrote the same Flash-Next kernels
(`executor.cpp`, the routed and prefill kernels), so nearly every upstream
change conflicts and each port is a reviewed hand port with its own GPU
quality and speed check.

## Porting rules

1. One upstream change per commit, message ending `(ported from gufo-org/gufo#N)`.
2. Host-side changes (HTTP, scheduling, caches) need the hosted CPU checks
   (`nix build .#checks.x86_64-linux.pr`).
3. Kernel or numerics changes also need the affected model's quality gates and
   an interleaved speed A/B on gfx1151 before they land; the fork's own lossy
   options are re-qualified on top of them.
4. Where both sides changed the same thing in different ways, keep the faster
   exact one, measured, and record the other here.

## Upstream changes

Upstream state reviewed: `main` at `6905be28` (4 November 2026).

| Upstream | Status | Notes |
| --- | --- | --- |
| #29 (fork `main`) homelab CI runners | ported, then reverted | Self-hosted runners would run public fork pull requests inside a private network; the public repository uses GitHub-hosted runners. |
| #16 (fork) sampler randomness keyed by seed and committed position | ported | Clean cherry-pick. |
| #17 (fork) grammar-constrained structured outputs | ported | Two added-line conflicts, both sides kept; one duplicated test assert removed. |
| #21 (fork) disk-checkpoint read-ahead for queued requests | ported | Clean cherry-pick. |
| #23 (fork `main`) unfused TurboQuant projection | not ported, needs a GPU A/B | Removes the fused tq4 QKV epilogue because it spilled to scratch. The campaign tree instead rewrote that epilogue to stay in registers (part of its exact +4% pp2048), so the two conflict on one kernel; keep the faster after measuring. |
| #875 prefill rows independent of the pass route | candidate, needs GPU | Likely the same family as the fork's open fused-step KL gap (prompt rows ≤ 256 differ, KL up to 0.0076; [Quality](models/qwen3.8-flash-next/QUALITY.md)). Highest-value port. |
| #953, #954 serving concurrency, cached requests without snapshot headroom | candidate, needs GPU | Serving correctness. |
| #894, #895, #933, #890 decode/prefill fusions, 16K-row prefill passes | candidate, needs GPU | Upstream's exact prefill now reaches ~1,930 tok/s, about the fork's lossy W4A4 stack; re-measure W4A4 after porting. |
| #957 drop API-key auth | **rejected** | The fork's router contract relies on `--api-key-file` ([Router](ROUTER.md#authentication)). |
| EmbeddingGemma 2 native GPU path (#770, #778, #823, #765, #813) | candidate | The fork's EmbeddingGemma work is the XDNA2 NPU experiment, not a GPU path. |
