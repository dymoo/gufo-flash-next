# Swarm goodput

Measures **correct answers per hour**: verified agent tasks completed per
wall-clock hour while a swarm of agents shares one OpenAI-compatible server.
Raw tok/s does not count; a task counts only when a mechanical checker accepts
its final state. Every reply, tool call and thinking token comes from the model
under test, so a model that solves tasks in fewer turns or fewer tokens gains
exactly as much as one that decodes faster. Stdlib-only Python (the test host has no
numpy); runs against Gufo, llama.cpp, exllamav3 or any server with streamed
Chat Completions and tool calls.

```sh
python3 tools/swarm/swarm.py run --base-url http://127.0.0.1:8011 --label flash-mid \
  --agents 32 --budget 1800 --thinking off --temperature 0.7 --top-p 0.8 --extra '{"top_k": 20}' \
  --runtime --gtt --out artifacts/swarm/flash-mid.json
python3 tools/swarm/swarm.py report artifacts/swarm/*.json --out artifacts/swarm/report.md
python3 tools/swarm/swarm.py tasks --count 16          # the pinned queue; --show code-00003 prints one task
```

Results of the October 2026 model selection, the v3 calibration history and the empty-reply
finding: [RESULTS.md](RESULTS.md).

## Load model

The load keeps [`agent_sim.py`](../qwen-flash/qa/agent_sim.py)'s arrival
pattern, shared system prompts and tool-output shapes, and replaces its fixed
reply lengths with real task trajectories.

- **Swarm size** `--agents N`: N agent slots. `--burst` slots start at t=0; the
  rest arrive as a Poisson process within `--ramp` seconds. Mid-size swarms are
  16–40 slots, large ones 64–128.
- **One ordered task queue**, the same for every run with the same seed: a
  slot takes the next task, runs it to the end, grades it, waits `--task-gap`
  and takes the next. A run therefore attempts a prefix of the queue, longer
  for faster or more efficient models. The queue is a sequence of stratified
  blocks of 16 (every family × variant once, in a seeded shuffle), so the
  family and tier mix is the same at every position. Slots are not tied to an
  agent type: each task brings its family's system prompt and tools.
- **Agent types.** Each task family is an agent type with its own shared
  system prompt (role, rules, a generated handbook of `--system-tokens`,
  default 20,000) and a fixed tool set, so concurrent agents of one type share
  a cacheable prefix, as in a real swarm. Every task starts a fresh
  conversation.
- **Tool time.** After each tool-calling turn the slot waits the simulated
  latency of its slowest call (exponential, per-tool means: 0.5 s for file and
  ticket edits, 1–1.5 s for searches and documents, 5 s for a test run;
  `--tool-time-scale` multiplies them).
- **Budget** `--budget`: the run ends at a fixed wall-clock time. Tasks still
  running then are recorded as `unfinished`: counted separately, never as
  failures, and excluded from the success rate and token costs.

## Tasks

Four families, each with four variants: one easy, one medium and two hard, so
a block of 16 holds 4 easy, 4 medium and 8 hard tasks. All worlds are generated
from `(family, index, seed)` with invented names, so no answer is in pretraining
data and the queue is identical everywhere for one seed and suite version.

| Family | Agent | Tools | Easy / medium / hard | Checker |
| --- | --- | --- | --- | --- |
| `code` | works in a four-module Python package (~70 tests) | list, read, grep, edit, write files; run tests | implement a simple stub / implement a function whose docstring specifies its edge cases (hidden tests check them) / fix a bug a failing test exposes plus one known only from a user report, or implement a specified function plus fix a reported bug | original visible tests plus hidden tests on the final files, in a sandbox (rlimits, unprivileged uid) |
| `extract` | turns business documents into records | search, get document; three strict-schema submit tools | labelled invoice / corrected invoice replacing an earlier one, net-terms due date, discount, decimal commas / incident review in a local time zone (UTC conversion across midnight, only customer-impacting services), or a purchase order against shipment notices with a cancelled notice, a goods return and other orders' notices | field-by-field comparison of the submitted record |
| `research` | answers questions from an encyclopedia of ~80 pages | search, open page, submit answer | two hops / three hops, or the latest product of the company in a city / a sum over every company whose current chief executive was born before a year, or who led a company when one of its products launched | normalized exact answer |
| `ops` | applies a policy to a 120-ticket tracker | list and get tickets, list users, update ticket, add comment | close duplicates with a reference comment / reassign an absent engineer's work to the on-call / escalate stale tickets except blocked ones, or an SLA sweep whose due dates and customer tiers only `get_ticket` shows | whole tracker state, so missed tickets, collateral edits and wrong comments fail |

Tool calls are validated against the declared JSON schemas; an invalid call
gets an error tool result (the agent may recover) and counts toward the
invalid-call rate. A server's tool-call parser can fail on markup the model
wrote correctly, so failures are attributed (`toolcalls.py`): tool markup left
in `content` that a reference parser (Hermes JSON, Qwen XML
`<function=..><parameter=..>`, LFM Pythonic, Mistral) turns into a
schema-valid call, or parsed arguments that are wrong only in scalar types, is
a `server_parse_failure`; anything else is `model_malformed`. Three
consecutive turns without a valid call end the task with the majority cause.
Only `model_malformed` counts against the model; a high
`server_parse_failure` count means fix the server and re-run. An agent that
stops without calling its family's submit tool gets one reminder.

### Hard tier (suite v3)

`--suite v3` (`families_v3/`, suite version `swarm-v3`) is a separate queue
of long-horizon tasks for when thinking models saturate v2. Its queue digest
covers its own sources, so v2 results stay pinned. Every v3 task is hard; a
block of 16 holds each family x variant once. The turn cap defaults to 60.
Tasks read 15-90K tokens of tool output, so serve them with a 128K context.
Every v3 checker also reports which of the task's requirements it met (a deal
field or table row, a brief's field, table row or citation, the share of an ops
rule's tickets or a ledger item kind's invoices handled right, a hidden test);
`report` prints the miss rate per requirement, so a comparison shows what
discriminates rather than one pass/fail per task. A task succeeds only when
every requirement is met; table rows are matched by key and extra rows count as
misses.

| Family | Agent | Variants | Checker |
| --- | --- | --- | --- |
| `code3` | works in `shopkit`, a ten-module order-management app (~2.1K lines with tests, plus seeded utility modules) | three-item tickets: a feature touching four or more modules (loyalty points, partial shipments, a ten-rule promotions engine), a storage-format migration that must still read the old format, or a pricing-API refactor through every call site, each plus two bugs known only from support tickets whose symptoms and causes sit in different modules; or three such bugs (39 combinations) | original visible tests, one scenario test per requirement, and a seeded differential test that drives the agent's package and the reference solution through the same random order script and compares quotes, stock, events, reports, CLI output and stored documents |
| `extract3` | a deal desk's briefs over document packs | three DealGate-style deal records - a software subscription (master agreement, order form, amendment, side letter: end date, notice deadline, compounding annual uplift, fee on a reference date, monthly contract value, ...), a supply agreement (tiered prices, forecasts in cases, indexation, volume rebate, penalties) and a services statement of work (milestones with relative due dates, two signed change requests, retention and its release, expenses cap) - each with documents that must not change the record (unsigned drafts, "subject to contract" emails, a rate card, an affiliate's NDA, other suppliers' contracts); or an accounts-payable open-items ledger (split and re-sent invoices, amendments, a cancellation, credit notes, multi-invoice and GBP remittances at the payment day's rate, a payment on account applied oldest due date first, a look-alike sister company) | every deal field (18-22 per record), or every invoice's outstanding amount, GBP value and aging bucket and the total |
| `research3` | Byline-style research briefs over a ~210-page encyclopedia (~90K tokens, 24 companies) | a company profile (17 fields), a person profile (10), a market brief for a country (10) or a group (M&A) brief (9), citing the page each single-source fact comes from; archived revisions, headquarters moves, renamed companies, near-duplicate names and three-level acquisitions on the way | every field and every citation |
| `ops3` | applies a precedence policy to two teams of a 400-ticket tracker (about 200 tickets read, 25-70 changes) | business-hours response SLA (deadlines from a `business_deadline` calendar tool; halved enterprise targets, paused tickets) / leave rebalancing to the least-loaded qualified engineer under a load cap with the lead as fallback / a five-rule first-match escalation with staleness in regional business days / duplicate chains merged into their root; each with a security override and comments an interrupted earlier sweep already left | whole tracker state; each rule reports the share of its tickets handled right |

`tests/tools/swarm_test.py` replays each family's oracle (its reference action
sequence) through the checkers and a fake streaming server: oracles must pass,
a do-nothing agent must fail, every code target's bug must be caught by its
visible and hidden tests, and refusals, overflows, invalid calls and the
budget cut-off must be classified as below.

## Fairness rules

For a comparison, every run uses the same suite version, seed (queue digest),
arrival schedule, system-prompt size, turn cap (`--max-turns`, default 40; 60 for v3) and
per-request output cap (`--max-tokens`, default 32,768), and the same budget.
Within those equal caps each model runs the way it works best: its model-card
sampling, thinking off and on, and each reasoning effort it supports, and
its best serving configuration next to the production one. The ranking uses
each model's best setting; all settings are reported. Earlier reasoning is
sent back with each request (`--history-reasoning keep`) and the model's own
chat template decides whether to keep it; a probe at the start of every run
renders one tool-loop history with and without it and records the template's
choice (`template_keeps_reasoning`).

## Metrics

Headline:

| Metric | Definition |
| --- | --- |
| Correct answers per hour | successful tasks completed within the budget × 3600 / budget (95% interval from a Poisson count); also steady state, counting completions after `--warmup` (default 300 s) |
| Success rate | successes / finished tasks (Wilson interval); unfinished tasks are a separate count |
| Completion tokens per correct answer | generated tokens of all finished tasks, including failed ones, / successes |
| Prefill tokens per correct answer | newly processed prompt tokens (prompt − cached) of finished tasks / successes |
| Turns per task | model requests per graded task |
| Thinking tokens per task | reasoning tokens per graded task (`completion_tokens_details` when the server reports it, else streamed reasoning chunks) |
| Seconds per solved task | mean wall time of successful tasks, including queueing and tool time |

Also reported: turns per hour, output and uncached-prefill tokens per second,
prefix-cache hit (cached / prompt tokens; per-turn median for turns ≥ 2), TTFT
p50/p90/p99, inter-token latency (ITL) p50/p99, capacity refusals, requests
longer than 600 s (the gateway's Gufo timeout), invalid-call rate, success by
family and tier, context size, and with `--runtime`, `--metrics-url` and
`--gtt` the peak resident and decoding sessions, queue, KV pool and GPU GTT.
Client TTFT is the time from sending a request to its first generated output
(text, reasoning or a tool call); Gufo emits a tool call whole when generation
ends, so for tool-only turns it approaches the request time. Server TTFT is
queue plus prefill to the first token as the server reports it
(`usage.gufo.ttft_ms`, or llama.cpp `timings`). Client ITL is the gap between
streamed text chunks; server ITL is each request's reported mean.
Correct answers per hour divide by the configured `--budget`. A run that finishes its
`--max-tasks` queue early (`budget_binding: false`, marked † in the report) therefore reports
a number that is not a rate; stall-proof comparisons use the token costs per correct answer
and the projection instead.
Record the serving configuration (sessions × context, KV type, flags) with
`--server` and `--note`. `--warmup` additionally reports correct answers per
hour counting only completions after the warm-up; latency percentiles use
the requests sent after it.

Outcomes: `success`, `wrong_answer` (finished, checker failed), `empty_reply` (a task without a
submit tool that ended on a reply with no content and no tool call, usually thinking only), `no_submission`,
`model_malformed` and `server_parse_failure` (see above), `turn_limit`, `output_cap` (final reply hit `--max-tokens`),
`context_overflow`, `server_error`, `unfinished` (budget ended mid-task). A
refused or reset connection is retried for `--outage-grace` seconds before the
task fails; when 3 × agents tasks in a row end in server errors the run stops
and records `aborted`, so a dead server cannot silently drain the queue.
429/503 capacity refusals are retried after `Retry-After`, backing off by 1.5×
per repeated refusal up to 8 s (the OpenAI SDK's cap); the wait counts as task
time.

## Empty replies

An agent loop that stops on a reply without a tool call ends the task on a thinking-only
reply (`finish_reason: stop`, no content, no tool call). The harness classifies it as
`empty_reply` and, with `--trace-dir DIR`, saves the raw stream of every one (request shape,
response headers, every SSE event, the reasoning text, no credentials) so it can be matched
with the server's log line by `body_bytes`. `swarm.py trace --task ID --repeat N --trace-dir DIR`
replays a task's first request with varied seeds to hunt for them; run it at light load only.
Over the first 3,659 requests recorded (7 settings, v2 and v3) one reply was empty; a second,
captured with `--trace-dir`, is analysed in [RESULTS.md](RESULTS.md).

## Results files

`run` writes one JSON per run (default `artifacts/swarm/<label>.json`, ignored
by Git): configuration, queue digest, server identity, every task record
(outcome, turns, tokens, timing; no generated text), every request (timing,
usage, finish) and server samples, plus the summary. `report` ranks runs by
correct answers per hour and warns when their queues or budgets differ.
`--max-tasks` bounds the queue; a run that exhausts it ends early, so its
correct answers per hour are not a rate (use it for the stall-proof
token-efficiency passes, which compare models on the same first N tasks).
`--first-task` starts at a later queue index, and `--skip-graded RESULT` skips
the tasks an earlier part already graded, so one pass can continue across GPU
windows; `merge` combines the parts by queue index (graded records win). A
partial result is written every `--checkpoint` seconds (default 300).

## Serving-independent cost and the Gufo projection

Correct answers per hour depend on the server. A model measured on a reference
server (llama.cpp, exllamav3) understates what Gufo could serve, so every run
also reports its server-independent cost: completion and newly-prefilled
tokens per correct answer, and turns per task. `report --rates rates.json`
adds a clearly labelled projection of correct answers per hour on Gufo:

```text
box seconds per correct answer = kappa * (P / R_p + C / R_d)
```

P and C are the run's prefill and completion tokens per correct answer, R_p
and R_d Gufo's measured prefill and decode rates for a model of that shape at
that swarm width, and kappa is calibrated so that a run measured on Gufo
reproduces itself. `rates.json` names the calibration run and each run's rates
with their source:

```json
{"calibrate": "flash-off-mid",
 "models": {"flash-off-mid": {"prefill_tok_s": 1500, "decode_tok_s": 275, "source": "Gufo BENCHMARKS.md"}}}
```
