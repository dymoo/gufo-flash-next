"""Merge the Kev-4B LoRA into Qwen3.5-4B-Base in fp32 (W + alpha/r * B @ A), keeping every other tensor.

The reference serves the adapter unmerged on fp32 weights (head.pt weights_dtype=fp32), so the merged fp32
checkpoint is the exact reference backbone; lower precisions are derived from it."""
import json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

base, adapter, out = sys.argv[1:4]
cfg = json.load(open(f"{adapter}/adapter_config.json"))
scale = cfg["lora_alpha"] / cfg["r"]
a = safe_open(f"{adapter}/adapter_model.safetensors", "pt")
lora = {}
for k in a.keys():
    mod, part = k.removeprefix("base_model.model.").rsplit(".lora_", 1)
    lora.setdefault(f"model.language_model.{mod}.weight", {})[part[0]] = a.get_tensor(k).float()
os.makedirs(out, exist_ok=True)
index = json.load(open(f"{base}/model.safetensors.index.json"))
weight_map, shard, shard_bytes, n, merged = {}, {}, 0, 0, 0
def flush():
    global shard, shard_bytes, n
    if not shard: return
    name = f"model-{n:05d}.safetensors"
    save_file(shard, f"{out}/{name}", metadata={"format": "pt"})
    weight_map.update({k: name for k in shard}); shard, shard_bytes, n = {}, 0, n + 1
for file in sorted(set(index["weight_map"].values())):
    with safe_open(f"{base}/{file}", "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k in lora:
                t = t.float() + scale * (lora[k]["B"] @ lora[k]["A"]); merged += 1
            elif t.dtype == torch.bfloat16 and "language_model" in k:
                t = t.float()
            shard[k] = t.contiguous(); shard_bytes += t.numel() * t.element_size()
            if shard_bytes > 4 << 30: flush()
flush()
assert merged == len(lora), (merged, len(lora))
json.dump({"metadata": {}, "weight_map": weight_map}, open(f"{out}/model.safetensors.index.json", "w"), indent=1)
for f in os.listdir(base):
    if not f.startswith("model") and os.path.isfile(f"{base}/{f}"): shutil.copy(f"{base}/{f}", out)
print(f"merged {merged} LoRA pairs (scale {scale}) into {out}")
