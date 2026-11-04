"""Metrics proxy for the MiMo vs Qwen agent benchmark.

The agent sends ordinary (non-streaming) chat completions to 127.0.0.1:PORT.
Each is forwarded upstream as a stream so time to first token and decode
speed can be measured, then reassembled into one completion (content,
reasoning, tool calls) for the agent. One JSONL row per request.
"""
import argparse, asyncio, hashlib, json, os, time
import aiohttp
from aiohttp import web

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--upstream", required=True)          # http://host:port/v1
ap.add_argument("--key-file", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--extra", default="{}")              # JSON merged into bodies
ap.add_argument("--header", action="append", default=[])  # "Name: value"
args = ap.parse_args()
KEY = open(args.key_file).read().strip()
EXTRA = json.loads(args.extra)
HEADERS = dict(h.split(": ", 1) for h in args.header)
LOG = open(args.log, "a", buffering=1)


def task_key(messages):
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            return hashlib.sha1((c or "")[:4000].encode()).hexdigest()[:16]
    return "none"


async def chat(request):
    body = await request.json()
    body.update(EXTRA)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    row = {"tag": args.tag, "task": task_key(body.get("messages", [])),
           "t0": time.time(), "messages": len(body.get("messages", []))}
    content, reasoning, calls = [], [], {}
    finish, usage, rid, model = None, {}, None, body.get("model")
    provider = None
    t_first = None
    try:
        timeout = aiohttp.ClientTimeout(total=3600, sock_read=1800)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(args.upstream + "/chat/completions", json=body,
                              headers={"Authorization": "Bearer " + KEY, **HEADERS}) as r:
                if r.status != 200:
                    text = await r.text()
                    row.update(status=r.status, error=text[:500])
                    return web.Response(status=r.status, text=text,
                                        content_type="application/json")
                async for raw in r.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    rid = ev.get("id", rid)
                    model = ev.get("model", model)
                    provider = ev.get("provider", provider)
                    if ev.get("usage"):
                        usage = ev["usage"]
                    for ch in ev.get("choices", []):
                        d = ch.get("delta") or {}
                        # Gufo streams reasoning_content, OpenRouter reasoning.
                        think = d.get("reasoning_content") or d.get("reasoning")
                        if (d.get("content") or think
                                or d.get("tool_calls")) and t_first is None:
                            t_first = time.time()
                        if d.get("content"):
                            content.append(d["content"])
                        if think:
                            reasoning.append(think)
                        for tc in d.get("tool_calls") or []:
                            c = calls.setdefault(tc.get("index", 0), {
                                "id": None, "type": "function",
                                "function": {"name": "", "arguments": ""}})
                            if tc.get("id"):
                                c["id"] = tc["id"]
                            f = tc.get("function") or {}
                            c["function"]["name"] += f.get("name") or ""
                            c["function"]["arguments"] += f.get("arguments") or ""
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
    except Exception as e:  # upstream died mid-request
        row.update(status=599, error=repr(e)[:500])
        return web.json_response({"error": {"message": repr(e)}}, status=502)
    finally:
        t_end = time.time()
        comp = usage.get("completion_tokens") or 0
        row.update(
            t_end=t_end, total_s=t_end - row["t0"],
            ttft_s=(t_first - row["t0"]) if t_first else None,
            prompt_tokens=usage.get("prompt_tokens"),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            completion_tokens=comp,
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            decode_tps=((comp - 1) / (t_end - t_first)) if t_first and comp > 1 and t_end > t_first else None,
            finish=finish, tool_calls=len(calls),
            server_prompt_tps=usage.get("prompt_tokens_per_second"),
            server_decode_tps=usage.get("completion_tokens_per_second"),
            gufo=usage.get("gufo"), provider=provider, cost=usage.get("cost"),
            id=rid)
        row.setdefault("status", 200)
        LOG.write(json.dumps(row) + "\n")
    msg = {"role": "assistant", "content": "".join(content) or None}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
        for i, c in enumerate(msg["tool_calls"]):
            c["id"] = c["id"] or f"call_{i}"
    return web.json_response({
        "id": rid or "chatcmpl-proxy", "object": "chat.completion",
        "created": int(row["t0"]), "model": model,
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": finish or ("tool_calls" if calls else "stop")}],
        "usage": usage})


async def passthrough(request):
    async with aiohttp.ClientSession() as s:
        async with s.request(request.method, args.upstream + request.path[3:],
                             headers={"Authorization": "Bearer " + KEY}) as r:
            return web.Response(status=r.status, body=await r.read(),
                                content_type=r.content_type)

app = web.Application(client_max_size=64 * 1024 * 1024)
app.router.add_post("/v1/chat/completions", chat)
app.router.add_get("/v1/{tail:.*}", passthrough)
web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)
