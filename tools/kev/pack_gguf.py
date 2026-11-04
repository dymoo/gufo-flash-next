#!/usr/bin/env python3
"""Add the Kev pointer head and its calibration to a converted Qwen3.5 backbone GGUF.

    python pack_gguf.py BACKBONE.gguf KEV_CHECKPOINT_DIR OUT.gguf --adapter REPO@REV --base REPO@REV

The backbone comes from merge_adapter.py and llama.cpp's convert_hf_to_gguf.py.
The head tensors are stored in F32 (kev.pointer.{q,k}.{weight,bias}); the
temperature fitted by the reference is kev.temperature.
"""
import argparse

import gguf
import numpy as np
import torch

FORMAT_VERSION = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone")
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args()

    meta = torch.load(f"{a.checkpoint}/head.pt", map_location="cpu", weights_only=False)
    if meta.get("option_isolation") or meta.get("special_embeddings") or meta.get("head_dim") != 256:
        raise SystemExit("unsupported Kev checkpoint layout")
    head = {k: v.float().numpy() for k, v in meta["head"].items()}

    reader = gguf.GGUFReader(a.backbone)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(a.out, arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), kind, sub_type=sub)
    writer.add_uint32("kev.format_version", FORMAT_VERSION)
    writer.add_float32("kev.temperature", float(meta["temperature"]))
    writer.add_uint32("kev.pointer_dim", int(meta["head_dim"]))
    writer.add_string("kev.adapter", a.adapter)
    writer.add_string("kev.base", a.base)

    extra = {f"kev.pointer.{name}": np.ascontiguousarray(t) for name, t in head.items()}
    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for name, t in extra.items():
        writer.add_tensor_info(name, t.shape, t.dtype, t.nbytes, gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    for t in extra.values():
        writer.write_tensor_data(t)
    writer.close()
    print(f"{a.out}: temperature {float(meta['temperature']):.6f}, {len(reader.tensors)} + {len(extra)} tensors")


if __name__ == "__main__":
    main()
