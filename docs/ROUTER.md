# Router contract

Gufo is the data plane for one machine: it admits or rejects each request at
once from real session, queue and KV state, then schedules it. A router such
as [llm-router](https://github.com/dymoo/llm-router) is the control plane: keys,
priority, waiting, retries, failover, batch jobs and accounting. This page is
the interface between them. Contract version: **1**.

## Service tiers

Chat Completions and Responses accept OpenAI's `service_tier`, with the
semantics of OpenAI Flex processing:

| `service_tier` | Behaviour |
| --- | --- |
| absent, `null`, `auto`, `default`, `scale`, `priority` | Default tier: the bounded pending queue (`--max-pending`), then a session |
| `flex` | Admitted only when the machine has idle compute: a session is free, no default request is queued or prefilling, and fewer than `--flex-sessions` (default 2) flex requests hold sessions. Otherwise **HTTP 429 `resource_unavailable`** with `Retry-After`, before any work or SSE headers |

`X-Gufo-No-Queue: 1` makes a default request fail fast too: 429 `queue_full`
when every session is taken, instead of waiting in the pending queue. A
router that owns waiting sends it on every request.

Other values return 400 `invalid_service_tier`.

The model id `<id>-flex` (for example `workhorse-flex` beside `workhorse`)
selects the flex tier for callers that can choose a model but not body fields
or headers. Every served id has one, steering adapters included
(`workhorse-abliterated-flex`): it is that variant with exactly the flex rules
above, on Chat Completions and Responses; responses report
`"model": "workhorse-flex"` and `"service_tier": "flex"`. The tier changes
scheduling only, never the variant's weights, arithmetic or cache entries;
a lossy variant such as the `-fast` model (Flash-Next `--fast-experts`) is a
separate id, so the two combine (`workhorse-fast-flex`). A model id or
adapter name ending in `-flex` is rejected at startup. A body `service_tier` of `default`, `scale` or
`priority` with it returns 400 `invalid_service_tier`; `flex`, `auto` and
`null` are accepted. Anthropic Messages (no tiers) returns 400 and raw
completions 404 for it. Responses and every stream
chunk report the tier that served them (`"service_tier": "flex"` or
`"default"`).

An admitted flex request runs to completion (no preemption), so the session
cap bounds what default traffic can lose to it: queued default requests are
admitted first, and in each fused Flash-Next step flex prompt chunks take
only the rows default prompts leave. While default requests only decode,
flex prompts wait; after 128 such steps (4 × the prefill aging limit) one
chunk of at most `--prefill-chunk` rows rides along, so an admitted flex
request always finishes. `--flex-sessions 0` rejects every flex
request. Under sustained default load flex is mostly rejected; it uses the
gaps. Measured costs and the rejected designs are in the Flash-Next
[experiments](models/qwen3.8-flash-next/EXPERIMENTS.md).

Gufo never holds a flex request waiting for capacity. The caller retries with
backoff, falls back to the default tier, or gives up; that policy belongs to
the router.

## Admission mode

`gufo serve llm --admission refuse` makes refusal the default for every
default-tier Chat Completions and Responses request: when every session is
taken, Gufo answers **HTTP 503 `capacity_exhausted`** with `Retry-After: 1`
at once, instead of waiting in the pending queue. The default,
`--admission queue`, keeps the behaviour above.

It exists for gateways that fail over on an upstream error, such as
Cloudflare AI Gateway. Its docs name no status codes: "By default,
Cloudflare triggers your fallback if a model request returns an error"
([Fallbacks](https://developers.cloudflare.com/ai-gateway/configuration/fallbacks/)),
and a dynamic-route model node takes its `fallback` output "if model fails
after all retries or times out", its `success` output only "if model
successfully starts streaming a response"
([Dynamic routing JSON](https://developers.cloudflare.com/ai-gateway/features/dynamic-routing/json-configuration/)).
Any refusal before the 200 therefore fails over; 503 is HTTP's "temporarily
overloaded" status and stays distinct from the gateway's own 429 rate and
spend limits. Set the gateway's retry count to 0 or 1 for Gufo, or each retry
waits out its delay before failing over.

| Request | `--admission queue` | `--admission refuse` |
| --- | --- | --- |
| no header | waits (`--max-pending`) | 503 `capacity_exhausted` |
| `X-Gufo-Queue: 1` | waits | waits (interactive clients, e.g. Open WebUI) |
| `X-Gufo-No-Queue: 1` | 429 `queue_full` | 429 `queue_full` |
| `service_tier: "flex"` | flex rule | flex rule |

`X-Gufo-No-Queue` wins over `X-Gufo-Queue` and keeps contract version 1, so
llm-router is unaffected by the mode. Waiting requests stay bounded by
`--max-pending` and `--max-pending-per-client` (429 `queue_full` /
`client_queue_full`). Refusal happens before any session lease, cache lookup,
image encoding or prefill; streams are refused before SSE headers. Requests
are logged with `error_code=capacity_exhausted` and counted as
`gufo_requests_total{outcome="rejected_capacity"}`. `GET /v1/runtime`
reports `"admission": "queue"` or `"refuse"`; `accepting.default` predicts
whether a request would be admitted. Anthropic Messages and raw completions
follow the mode but read neither queue header, so under `refuse` they always
get 503 when full; System One ignores the mode (`/v1/embeddings` is not
implemented). The key file holds one key, so
there is no per-key mode.

A gateway route node chooses only provider, model, timeout and retries, so a
route sends background traffic to `workhorse-flex` (refused at once with 429
`resource_unavailable` when there is no idle compute) and interactive traffic
to `workhorse` (503 `capacity_exhausted` under `--admission refuse`); both
fail over.

## Model ids

`GET /v1/models` lists the text model's id first, then every steering adapter
it serves (`--adapter NAME=PATH`, Flash-Next) with `"parent"` naming the base
id, then each of those ids' flex alias (`<id>-flex`, `"parent"` the id it
suffixes, see [Service tiers](#service-tiers)), then Kev. For example, one
process serves `workhorse`, `workhorse-abliterated` (the Orca adapter),
`workhorse-flex` and `workhorse-abliterated-flex` from the same weights,
sessions and batches. Chat Completions, Responses and Anthropic Messages select the variant
by `model` and echo it; raw completions accept only the base id; an unknown id
is 404 `model_not_found`. Variants share capacity, queues, `/v1/runtime` and
metrics, and never share cache entries. Request log lines carry `model=`.
The router maps each public id to the same deployment with the id as its
upstream model.

## `GET /v1/runtime`

Live state, computed on each call (authenticated like every route):

```json
{
  "object": "gufo.runtime",
  "contract_version": 1,
  "model": "qwen3.8-flash-next-gufo",
  "revision": "3787aec",
  "kv_cache": "tq4",
  "moe_routing": "exact",
  "speculative": "off",
  "draining": false,
  "admission": "queue",
  "accepting": { "default": true, "flex": true },
  "sessions": { "capacity": 24, "resident": 9, "decoding": 7, "flex": 1,
                "flex_limit": 2, "prefilling_default": 0,
                "context_tokens": 131072 },
  "queued": { "default": 0, "flex": 0, "limit": 512 },
  "kv_pool_bytes": { "used": 8589934592, "limit": 16911433728 },
  "totals": { "prompt_tokens": 91830112, "generated_tokens": 402731 }
}
```

- `accepting.default`: a default request would start now, without queueing.
  False while draining or when every session is taken; a no-queue request
  would then get 503 `draining` or 429 `queue_full`. Use it as runtime
  evidence of saturation.
- `accepting.flex` predicts whether a flex request would be admitted now. It
  is a snapshot: the request can still get 429 `resource_unavailable`.
- `totals` are monotonic since start; rates are their differences over time.
- `resident` counts sessions leased by requests; idle conversations paged to
  RAM or disk are not resident.

## Drain

`POST /v1/runtime/drain` stops admission until restart. New requests get 503
`draining` with `Retry-After`, `/ready` returns 503 `draining`, and admitted
and queued requests finish. Drain before planned maintenance, wait for
`sessions.resident` and `queued` to reach 0, then stop the service.

Because one call takes the server out of service, drain is not open to every
key holder. By default it is accepted only from a loopback peer (127.0.0.0/8,
the TCP peer address, never a header), with the API key when one is set; other
clients get 403 `admin_required`. A router or operator on another host needs
`--admin-key-file PATH`: then the request must carry that key as its bearer
token from any address, loopback included, and the API key no longer drains.
The admin key authorizes nothing else and must differ from the API key. Behind
a reverse proxy on the same host every client is loopback, so set
`--admin-key-file` there.

## Request identity and accounting

A caller's `X-Request-ID` of up to 128 characters from `[A-Za-z0-9._:-]`
becomes the request's ID in logs and in the echoed `X-Request-ID` header;
otherwise Gufo generates one. Usage reports `prompt_tokens`,
`completion_tokens` and `prompt_tokens_details.cached_tokens`; `timings`
(llama.cpp format) reports queue, prompt and generation milliseconds.

Gufo queues and round-robins by client IP. Behind one router every request
shares an IP; with `X-Gufo-No-Queue` nothing queues, so the per-client limit
does not bind.

## Authentication

`--api-key-file PATH` reads the bearer key from a file, and
`--admin-key-file PATH` the separate key for [drain](#drain). Bind to a private
address; TLS belongs in a proxy. `--metrics-address` serves an
unauthenticated `/metrics` for Prometheus.

## Mapping in llm-router

| Router | Gufo |
| --- | --- |
| High and medium keys | default tier; high keys keep reserved router permits |
| Low keys, any client `service_tier: "flex"`, batch items | `service_tier: "flex"`; the router retries a refusal after `Retry-After` until the key's `maxWaitMs`, then returns 429 `resource_unavailable`. Flex never goes to cloud |
| Waiting, backoff, visible queued notice, cloud failover | router only |
| Verified saturation | `accepting.default == false`, 429 `queue_full`, or 503 `draining` |
| Capacity and latency figures | `/v1/runtime` sessions and `totals`, not static catalogue values |

Flex 429 is local overload for that request, not verified saturation: it only
means the reserve is binding.

## System One

With Kev loaded, `POST /v1/systemone` takes TypeSafe System One requests
(Kev-4B, [model docs](models/kev-4b/README.md)). It answers in TypeSafe's
shapes: `{"detail": ...}` errors, 422 for invalid or oversized input, 429 with
`Retry-After` when `max_pending` requests already wait, and
`x-typesafe-request-id` echoed when it is a plain token. Queued requests are
batched into packed passes, so concurrent callers are the fast path.
`GET /v1/runtime` gains a `systemone` object (`pending`, `max_pending`,
`requests`, `rejected`, `batches`, `tokens`, `questions`). `GET /v1/models`
adds a TypeSafe `models` list beside OpenAI's `data`.

In llm-router, Kev is an auxiliary `systemone` deployment with `transport:
"gufo"` (its `docs/catalogue.md`); the router never swaps it for cloud Jev.

## Response privacy

The continuation cache is shared by every client of one Gufo process. Cache
responses report hits and miss reasons but not how far a miss matched other
entries (`usage.gufo` has no prefix lengths). Per-caller cache scopes are not
yet part of this contract, so give untrusted tenants separate Gufo processes.
