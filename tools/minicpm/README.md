# MiniCPM-V 4.7 spike tools

Stdlib-only Python (the test host has no numpy), run against any OpenAI-compatible
server; `gen_vision.py` alone needs Pillow and runs on the editing host.

| Script | Measures |
| --- | --- |
| `load.py` | Streaming load: closed-loop C1…C128 sweeps per prompt shape (chat, decode, rag, code, extract) and open-loop Poisson arrivals at fractions of measured peak (`--rates-from`); TTFT, inter-token latency, per-user and aggregate tok/s, SLO goodput, wall-clock run windows for energy joins. |
| `evals.py` | HumanEval (pass@1, sandboxed), GSM8K, and the pinned DS4 set (GPQA Diamond, SuperGPQA, AIME 2025) with thinking. |
| `parity.py` | Text parity on one GGUF: llama.cpp greedy top-20 logprobs as reference, Gufo teacher-forced (`forced_completion`): KL, top-1 agreement, greedy identity. `--corpus` adds long prompts. |
| `vparity.py` | The same for image prompts (llama.cpp mtmd as reference). |
| `gen_vision.py` / `vision_eval.py` | 48 synthetic documents (invoices, bar charts, tables, order screens) with exact answers, and their accuracy/throughput eval. |
