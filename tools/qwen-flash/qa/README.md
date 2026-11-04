# Flash-Next serving QA

Judges every serving change or opt-in flag by measured quality delta next to its
speed delta, over HTTP against a running `gufo serve`. Stdlib-only Python
(the test host has no numpy); greedy, thinking off, pinned public-domain texts
(Gutenberg *Moby Dick* and *Pride and Prejudice*, SHA-256 checked, cached under
`artifacts/qa/corpus`).

| Group | Suite | What it measures |
| --- | --- | --- |
| quality | `logits` | Teacher-forced book continuations at 2K/16K/64K/120K tokens plus 8 concurrent 2K passages: per-position top-20 logprobs → KL, top-1 agreement, forced-token Δlogprob, perplexity. Needs `logprobs`/`forced_completion` (this branch). |
| | `needle` | Three needles at 10/50/90% of 8K/32K/64K/120K haystacks, one multi-key question per haystack. |
| | `tasks` | 20 code tasks run against asserts, 20 exact-answer math, 8 tool calls, 6 JSON outputs (C8, ~2–3 min). |
| | `multiturn` | Three turns: restored-snapshot repeats must match; cold-prefill equality and logprob drift are reported. |
| correctness | `determinism` | Same greedy request twice. |
| | `batch` | Single-session references vs C=8/32/64 cohorts: identical outputs, plus teacher-forced KL of the batched rows. |
| | `cancel` | Abort mid-decode and mid-prefill; health and identical output afterwards. |
| | `stops` | Stop sequence, streamed/unstreamed equality, streamed tool-call arguments. |
| | `overload` | `sessions + 2` concurrent requests all complete; context overflow is a 400; still healthy. |
| performance | `decode` | Prepared cohorts at C=1…64 (≤ sessions), prose/repetition/code, depth 0 and 32K shared prefix: summed decode tok/s, wall tok/s, TTFT, ITL, draft acceptance, output hashes. |
| | `prefill` | Cold prefill tok/s at 2K…120K and the warm restore of the same prefix. |
| capacity | `capacity` | Fill every session to `context − 8K` (shared prefix), decode together; memory peaks. |

Every run samples GTT, container `MemAvailable`, cgroup memory and server RSS
every 0.5 s per suite. Results: `artifacts/qa/<label>.json` (full positions and
outputs; never commit them).

## Maintenance window (production stopped)

```sh
cd /path/to/gufo          # any checkout with a built gufo
Q=tools/qwen-flash/qa
BIN=build/release/gufo                       # the build under test (needs this branch for `logits`)
$Q/window.sh --gufo $BIN --label base                      # exact baseline, AR, 8 sessions, port 8011
$Q/window.sh --gufo $BIN --label kv-tq4 -- --kv-cache tq4  # candidate flag (after `--`)
$Q/window.sh --gufo $BIN --label base-mtp --mtp --suites performance,tasks,determinism
python3 $Q/qa.py report artifacts/qa/base.json artifacts/qa/kv-tq4.json --out artifacts/qa/kv-tq4.md
python3 $Q/qa.py report artifacts/qa/base.json artifacts/qa/base-mtp.json   # MTP greedy identity
# Capacity sweep: one server per configuration
for cfg in 8:131072 16:65536 32:32768 64:16384; do
  $Q/window.sh --gufo $BIN --label cap-${cfg/:/x} --sessions ${cfg%:*} --context ${cfg#*:} --suites capacity
done
```

`window.sh` refuses to start while production or another model holds the GPU,
waits for `/ready`, runs `qa.py run`, and always stops its server. Default
`--suites all` takes roughly 25 minutes at 8 sessions; `--suites quality` about
10. Run teacher-forced suites on an AR server: probing requests bypass
speculation. `--continuation` adds `tools/serving/check-continuation.py`.

Smoke-testing the harness against production is limited to C1 and short inputs:

```sh
python3 $Q/qa.py run --base-url http://127.0.0.1:8000 --api-key-file /etc/strix-models/api-key \
  --label prod-smoke --quick --sessions 1 --suites needle,tasks,determinism,multiturn,cancel,stops,decode,prefill
```

## Load tools

`agent_sim.py` replays coding-agent traffic: Poisson arrivals sharing one or
more long system prompts (`--agent-types`), multi-turn tool loops with random
tool times and output sizes, streamed replies. It reports TTFT (all and
first-turn), inter-token latency, per-turn cache hit and totals.
`throughput.py SERVER.log` rebuilds per-second prefill/decode rates (average,
60 s and 10 s peaks) from the server's request logs; client totals alone
cannot separate prefill from decode. `mixed_load.py` runs fixed streaming
decoders plus periodic prompt appends to measure decoder latency under prefill.

```sh
python3 $Q/qa.py corpus                                  # pinned texts for prompts
python3 $Q/agent_sim.py --base-url http://127.0.0.1:8011 --corpus artifacts/qa/corpus/*.txt \
  --duration 900 --arrival-rate 0.05 --max-agents 40 --out agents.json
python3 $Q/throughput.py server.log
```

## Gates (proposed, candidate vs exact baseline)

| Metric | Pass | Warn | Basis |
| --- | --- | --- | --- |
| mean KL (nats) | ≤ 0.005 | ≤ 0.02 | Q6_K-vs-FP16 drift is ~0.005; Q4-class ~0.02–0.05 |
| p99 KL | ≤ 0.05 | ≤ 0.2 | tail positions |
| top-1 agreement | ≥ 98% | ≥ 95% | |
| forced-text PPL increase | ≤ 0.5% | ≤ 2% | |
| needles lost | 0 | 0 | any loss fails |
| task pass-rate drop | ≤ 2 pp | ≤ 5 pp | 54 tasks, ±1 task ≈ 1.9 pp |

Worse than warn fails. KL uses the reference top-20 plus a tail bucket (a
token outside the candidate's top-20 gets its 20th logprob), so it slightly
underestimates full-vocabulary KL. Lossless changes (same math, fusion,
batching) instead need KL 0, identical greedy outputs and unchanged
needle/task results; the report's "greedy outputs identical" row and batch KL
show it.
