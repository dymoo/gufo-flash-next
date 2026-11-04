#!/usr/bin/env python3
"""Prefix-cache isolation between steering-adapter ids with never-used prompts, RAM and disk.

  cache_fresh.py seed  --base-url URL --state FILE [--models base,huihui,orca,orca-blocks]
  cache_fresh.py probe --base-url URL --state FILE        (against a RESTARTED server, same --cache-disk)

seed (server A, --cache-disk DIR):
  every prompt is fresh (time- and urandom-seeded text, asserted uncached on its first request).
  RAM tier  per id X: X prompts P_X (miss), every other id asks the same P_X (must miss), then
            every id repeats P_X (must hit its own entry, the positive control).
  disk tier per id X: X alone asks D_X (stored in RAM, written to disk). Per adapter id X: three
            conversations share one fresh system block S_X (the pin threshold), asked by X only.
probe (server B, restarted on the same directory, so RAM is empty and only disk can hit):
  per id X: every other id asks D_X first (must miss), then X asks D_X (must hit from disk).
  per adapter id X: every other id asks S_X + a new user turn (must miss), then X does (must hit
  the system-block checkpoint or pin, restored from disk).
Exit status 0 only when every cross-id request missed and every control hit.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qa"))
import qa  # noqa: E402


def fresh_seed() -> int:
    return (time.time_ns() ^ int.from_bytes(os.urandom(4), "little")) % (2**31 - 20_000) + 10_000


def text(tag: str, words: int, salt: int) -> str:
    return f"[{tag}] " + qa.synthetic_text(salt, words)


def ask(client: qa.Client, model: str, messages: list[dict], log: list[dict], label: str, expect: str) -> dict:
    client.model = model
    r = client.chat(messages=messages, max_tokens=8)
    usage = r.get("usage") or {}
    gufo = usage.get("gufo") or {}
    row = {"label": label, "model": model, "expect": expect, "prompt_tokens": usage.get("prompt_tokens"),
           "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
           "cache_restore_bytes": gufo.get("cache_restore_bytes"), "text": qa.content(r)}
    row["ok"] = row["cached_tokens"] == 0 if expect == "miss" else row["cached_tokens"] > 0
    log.append(row)
    print(f"{label:26s} {model:12s} cached {row['cached_tokens']:5d}/{row['prompt_tokens']} "
          f"expect {expect:4s} {'ok' if row['ok'] else 'VIOLATION'}", flush=True)
    return row


def user(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt + "\n\nName the three most frequent words above."}]


def system_chat(system: str, turn: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": turn}]


def seed(args, client: qa.Client, models: list[str], adapters: list[str]) -> dict:
    base = fresh_seed()
    state = {"models": models, "adapters": adapters, "seed": base, "ram": {}, "disk": {}, "system": {}}
    log: list[dict] = []
    k = 0
    print("== RAM tier: owner miss, cross-id miss, repeat hit")
    for owner in models:
        k += 1
        p = user(text(f"ram-{owner}", 1200, base + k))
        state["ram"][owner] = p
        ask(client, owner, p, log, f"ram {owner} owner", "miss")
        for other in models:
            if other != owner:
                ask(client, other, p, log, f"ram {owner} by {other}", "miss")
        for other in models:
            ask(client, other, p, log, f"ram {owner} repeat {other}", "hit")
    print("== disk tier seed: owner only")
    for owner in models:
        k += 1
        p = user(text(f"disk-{owner}", 1200, base + k))
        state["disk"][owner] = p
        ask(client, owner, p, log, f"disk {owner} owner", "miss")
    for owner in adapters:
        k += 1
        system = text(f"system-{owner}", 700, base + k)
        state["system"][owner] = system
        for turn in range(3):
            k += 1
            ask(client, owner, system_chat(system, text(f"turn{turn}", 60, base + k)), log,
                f"system {owner} conv{turn}", "miss" if turn == 0 else "any")
    state["seed_log"] = log
    return state


def probe(args, client: qa.Client, state: dict) -> None:
    models, adapters = state["models"], state["adapters"]
    log: list[dict] = []
    k = fresh_seed()
    print("== disk tier probe: cross-id first (miss), then owner (disk hit)")
    for owner in models:
        p = state["disk"][owner]
        for other in models:
            if other != owner:
                ask(client, other, p, log, f"disk {owner} by {other}", "miss")
        ask(client, owner, p, log, f"disk {owner} owner", "hit")
    print("== system block / pin probe: cross-id first (miss), then owner (hit)")
    for owner in adapters:
        system = state["system"][owner]
        for other in models:
            if other != owner:
                k += 1
                ask(client, other, system_chat(system, text("probe", 60, k)), log,
                    f"system {owner} by {other}", "miss")
        k += 1
        ask(client, owner, system_chat(system, text("probe", 60, k)), log, f"system {owner} owner", "hit")
    state["probe_log"] = log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["seed", "probe"])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--models", default="base,huihui,orca,orca-blocks")
    ap.add_argument("--api-key-file")
    args = ap.parse_args()
    client = qa.Client(args.base_url, Path(args.api_key_file).read_text().strip() if args.api_key_file else "")
    path = Path(args.state)
    if args.phase == "seed":
        models = args.models.split(",")
        state = seed(args, client, models, [m for m in models if m != "base"])
        logs = state["seed_log"]
    else:
        state = json.loads(path.read_text())
        probe(args, client, state)
        logs = state["probe_log"]
    path.write_text(json.dumps(state, indent=1) + "\n")
    bad = [r for r in logs if not r["ok"] and r["expect"] != "any"]
    cross = [r for r in logs if r["expect"] == "miss" and " by " in r["label"]]
    hits = [r for r in logs if r["expect"] == "hit"]
    print(f"{args.phase}: {len(cross)} cross-id requests (all must miss), {len(hits)} own-entry controls "
          f"(all must hit), violations {len(bad)}")
    print(f"cache isolation {args.phase} {'PASS' if not bad else 'FAIL'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
