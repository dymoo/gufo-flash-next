#!/usr/bin/env python3
"""Scope variants of a v2 steering adapter (same direction and alpha) in DIR:
orca-nomtp.json (writers blocks, embedding, ple) and orca-blocks.json (v1: block outputs),
plus a symlink to the direction file.

  make_variants.py ADAPTER.json DIR
"""
import json
import os
import sys

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
m = json.load(open(src))
f32 = os.path.join(os.path.dirname(os.path.abspath(src)), m["direction_file"])
link = os.path.join(out, m["direction_file"])
if os.path.lexists(link):
    os.unlink(link)
os.symlink(f32, link)
nomtp = dict(m, writers=["blocks", "embedding", "ple"])
blocks = {k: v for k, v in m.items() if k != "writers"}
blocks["format"] = "gufo steering adapter v1"
for name, meta in (("orca-nomtp", nomtp), ("orca-blocks", blocks)):
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=1)
