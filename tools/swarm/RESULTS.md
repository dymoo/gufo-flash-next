# Swarm goodput: results (October 2026)

Decision: **Qwen3.8 Flash-Next on Gufo, thinking on, `reasoning_effort: medium`**, as the swarm
workhorse. Every number below comes from runs on one Strix Halo test host (production
workhorse `gufo-749a129b`, 32 × 131072 tq4 sessions), the pinned queue seed 1, the default
20,000-token system prompts and real model trajectories. Result JSONs stayed on that host
(not committed); `compare.py` and `report` regenerate the tables.

## How these were run, and what they do not show

The host was memory-stalled for most of this work (a hung network mount), so the comparison is
**stall-proof**: 4 agents (8 for Ornith in its own window) on a task-count budget, comparing
models on the queue indices every run graded. The runs measure accuracy and the
serving-independent token cost of a correct answer. They do not measure correct answers per
hour at 32-96 agents: wall-clock times include other agents' windows and host stalls, and the
`correct/h` column `report` prints for these runs is not a rate (marked †). Throughput at swarm
scale is a projection:

    box-seconds per correct = κ × (ideal prefill tokens / Rp + completion tokens / Rd)

with κ = 1.8 fitted on the Gufo agent-serving tables, Flash-Next Rp = 1500 and Rd = 245 tok/s
at 32 concurrent sessions, and Ornith on Gufo-EXL3 Rp ≈ 660 and Rd = 123 tok/s at C32.
"Ideal prefill" removes the shared system prompt and counts only what each turn adds, so it
does not depend on the server's prefix cache.

## v2 (the main suite): settings

Flash-Next, all graded tasks of the first 64 (xhigh was cut short at 16 started, 12 graded):

| Setting | Correct | Success (95% CI) | Compl. tok/correct | Think tok/correct | Ideal prefill tok/correct | Turns/task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| thinking off | 52/63 | 83% (71-90) | 1,066 | 0 | 7,026 | 9.7 |
| on, low | 64/64 | 100% (94-100) | 1,554 | 851 | 4,071 | 5.3 |
| **on, medium** | 63/64 | 98% (92-100) | **1,250** | 605 | **4,002** | 5.3 |
| on, xhigh (partial) | 12/12 | 100% (76-100) | 1,364 | 704 | 4,012 | 5.3 |

Same 42 tasks graded by every run (indices 0-47 minus 27, 35, 37, 41, 46, 47):

| Model / setting | Correct | Compl. tok/correct | Think tok/correct | Ideal prefill tok/correct | Projected correct/h (C32) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next off | 35/42 (83%) | 899 | 0 | 6,118 | ~258 (17% of answers wrong) |
| Flash-Next low | 42/42 | 1,525 | 831 | 4,038 | ~224 |
| **Flash-Next medium** | 42/42 | **1,166** | **530** | **3,823** | **~274** |
| Ornith-1.5-122B on (exllamav3) | 42/42 | 2,058 | 1,218 | 4,394 | ~86 on Gufo-EXL3 (~176 at Flash-Next's rates) |

- Thinking is worth it: off loses 17% of answers and needs 1.6× the turns.
- Medium beats low on cost at equal accuracy: fewer completion tokens on 37 of 48 paired tasks,
  in every family. Gufo's low effort adds "keep your thinking brief" to the prompt; in tool loops
  it buys more turns and longer reasoning. Medium adds nothing (the native default).
- Ornith matches the accuracy but spends 1.8× the completion tokens and decodes at half
  Flash-Next's rate on this box, so it projects to about a third of the goodput. Ornith off was
  86% on its first 22; Gufo-EXL3 on was too slow under the stall to finish (6/6 graded).
- Not measured: Qwen3.8-27B, DeepSeek-V4-Flash, LFM2.5, MiMo-V2.6-Flash (needs an infra
  change), MiniMax-H3 and MiniCPM (excluded). The swarm-scale runs (32/96 agents) waited for a
  healthy host and were not run.

## v3 (hard tier): calibration history and saturation

v3 exists because thinking settings saturate v2. Target: 40-60% for the recommended setting.
Five revisions, 32-task calibrations on the live workhorse at 4 agents:

| Rev | Setting | Queue | Result | What changed / what missed |
| --- | --- | --- | --- | --- |
| 1 | low | d0de5c07 | 19/20 graded (stopped) | longer tasks; 1 miss (a subtle spec rule) |
| 2 | low | 3c3245da | 12/12 graded (stopped) | more volume and hops: no effect |
| 3 | medium | 22182235 | 25/32 (78%) | requirement-dense realistic briefs; misses in 2 rules + an unfair citation check (fixed) |
| 4 | medium | ce2491a4 | 27/32 (84%) | denser briefs, 3-item code tickets; misses: company parent lookup 2/2, SLA arithmetic 2/2, 1 empty reply |
| 5 | medium | 88967cd4 | 21/25 graded (84%, CI 65-94), 4 unfinished when stopped | tables per row, paginated search, a calendar tool; misses: parent lookup 2/2, SLA 1, promotions 1, 1 empty reply |

The tier saturates at about 84% for Flash-Next medium. Volume, hops, longer contexts and more
fields did not move it; what failed was a small set of rules (finding a company's owner from the
owner's page, business-hours SLA application) and one model fault (the empty reply). Per-field
reliability on realistic extraction and research briefs is above 99%. v3 still separates
thinking off from on, and its per-requirement miss table shows which rules a model gets wrong;
it does not give the headroom of a 50% tier. No further revision was made (decision: rev 5 is
the last).

## Thinking-only empty replies

One reply in 3,659 recorded requests before tracing (Flash-Next medium), and a second one after
`--trace-dir` was added, on the **byte-identical request** (code3-00004 turn 1, body 82,992 bytes):

- Server: `request=r3208 ... prompt_tokens=19118 generated_tokens=97 finish=stop cache=memory
  cached_tokens=18545 resident_at_admission=4 batch_width=4 plan=batched-w4` (the first was
  r2142 with the same plan and token counts).
- Stream: 97 tokens of `reasoning_content` — "Let me start by exploring the repository
  structure." then a restatement of the three ticket items — then `finish_reason: stop`. No
  `</think>`, no tool-call markup, no content.
- The same request alone (`serial-fallback`) and 150 replays at 4 concurrent requests all called
  `list_files` after 24-61 tokens.
- Gufo's output parser treats everything before `</think>` as reasoning, and `FinishReason`
  returns `stop` for any reply without tool calls, so EOS inside thinking reaches the client as a
  clean, empty stop. An agent loop that stops on a reply without tool calls ends the task.
- The harness now records this as `empty_reply` and saves the raw stream (`--trace-dir`); the
  `trace` subcommand and `empty_scan.py` reproduce the analysis.
