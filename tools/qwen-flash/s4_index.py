#!/usr/bin/env python3
"""Converts dense-w4-calib quantize.py output (index.json) to the index.tsv the
Flash-Next executor reads with GUFO_DENSE_I4_WEIGHTS=<dir> (name rows cols
offset bytes per line; dense.bin stays as is). Only S4G128 tensors."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
idx = json.loads((d / "index.json").read_text())
lines = []
for t in idx["tensors"]:
    if t["type"] != "S4G128":
        continue
    cols, rows = t["shape"][0], 1
    for n in t["shape"][1:]:
        rows *= n
    assert t["bytes"] == rows * (cols + cols // 128 + 4), t["name"]
    lines.append(f'{t["name"]} {rows} {cols} {t["offset"]} {t["bytes"]}')
(d / "index.tsv").write_text("\n".join(lines) + "\n")
print(len(lines), "tensors")
