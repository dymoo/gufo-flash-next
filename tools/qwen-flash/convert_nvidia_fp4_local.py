#!/usr/bin/env python3
"""Convert NVIDIA's pinned Qwen3.8 FP4-Local checkpoint to native INT4 GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import os
import re
import shutil
import tempfile
import subprocess

REPO_ID = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
REVISION = "b9ce18a6887e18c089fdea8509a33e63429a3300"
EXPECTED_TENSOR_COUNT = 375_487
EXPECTED_SHARD_COUNT = 21
EXPECTED_SOURCE_BYTES = 102_138_950_916
EXPECTED_RECIPE_COUNTS = {
    "NVFP4": 49,
    "MXFP8": 438,
    "W4A16_NVFP4": 58,
}
# Identity of REVISION as published by the Hugging Face paths-info API: the
# index is an LFS object (SHA-256 oid); config.json is a plain git blob (SHA-1).
EXPECTED_INDEX_SHA256 = "5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1"
EXPECTED_CONFIG_GIT_BLOB_SHA1 = "1efa9baa6f332ef821a9468eddd0414ee9bfefb0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_tensor_types(gguf: object, paths: list[Path]) -> dict[str, str]:
    """Map every tensor name in the reference GGUF shards to its GGML type name."""
    types: dict[str, str] = {}
    for path in paths:
        for tensor in gguf.GGUFReader(path).tensors:
            if tensor.name in types:
                raise ValueError(f"tensor {tensor.name} appears twice in the reference GGUFs")
            types[tensor.name] = tensor.tensor_type.name
    if not types:
        raise ValueError("reference GGUFs contain no tensors")
    return types


def write_tensor_type_file(types: dict[str, str], path: Path) -> None:
    """Write llama-quantize --tensor-type-file lines: one exact-name regex per tensor."""
    lines = [f"^{re.escape(name)}$={types[name].lower()}\n" for name in sorted(types)]
    path.write_text("".join(lines), "utf-8")


def verify_tensor_types(gguf: object, path: Path, expected: dict[str, str]) -> dict[str, int]:
    """Require the GGUF's tensor names and types to equal `expected`; return type counts."""
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for tensor in gguf.GGUFReader(path).tensors:
        if tensor.name in actual:
            mismatches.append(f"{tensor.name}: duplicate tensor")
        actual[tensor.name] = tensor.tensor_type.name
    for name in sorted(expected.keys() - actual.keys()):
        mismatches.append(f"{name}: missing (expected {expected[name]})")
    for name in sorted(actual.keys() - expected.keys()):
        mismatches.append(f"{name}: unexpected (got {actual[name]})")
    for name in sorted(expected.keys() & actual.keys()):
        if actual[name] != expected[name]:
            mismatches.append(f"{name}: got {actual[name]}, expected {expected[name]}")
    if mismatches:
        raise ValueError(
            f"GGUF tensor types differ from reference in {path.name} "
            f"({len(mismatches)} mismatches); first 8: {'; '.join(mismatches[:8])}"
        )
    return dict(Counter(actual.values()))


COMPRESS_RATIOS_KEY = "qwen4exp.attention.compress_ratios"


def expected_compress_ratios(config: dict) -> list[int]:
    """Per-layer indexer compression the runtime requires: the ratio on every
    full-attention layer, zero on linear layers (raw config.json labels)."""
    text = config.get("text_config", config)
    ratio = text["indexer_compress_ratio"]
    layer_types = text["layer_types"]
    if len(layer_types) != text["num_hidden_layers"]:
        raise ValueError("layer_types does not describe every layer")
    if set(layer_types) != {"linear_attention", "full_attention"}:
        raise ValueError(f"unexpected layer types: {sorted(set(layer_types))}")
    return [ratio if kind == "full_attention" else 0 for kind in layer_types]


def verify_compress_ratios(gguf: object, path: Path, expected: list[int]) -> None:
    actual = [int(v) for v in gguf.GGUFReader(path).fields[COMPRESS_RATIOS_KEY].contents()]
    if actual != expected:
        raise ValueError(f"{COMPRESS_RATIOS_KEY} mismatch in {path.name}: {actual} != {expected}")


def make_converter_view(source: Path, directory: Path) -> Path:
    """Expose nested HF shards to llama.cpp's flat-shard discovery glob."""
    view = directory / "model"
    view.mkdir()
    for entry in source.iterdir():
        os.symlink(entry.resolve(), view / entry.name, target_is_directory=entry.is_dir())
    if not any(view.glob("model*.safetensors")):
        shard_dir = source / "tensors"
        shards = sorted(shard_dir.glob("model*.safetensors"))
        if len(shards) != EXPECTED_SHARD_COUNT:
            raise ValueError(f"expected {EXPECTED_SHARD_COUNT} nested shards, got {len(shards)}")
        for shard in shards:
            os.symlink(shard.resolve(), view / shard.name)
    return view


def make_ple_weight_loader(checkpoint, torch_module, lazy_tensor_class):
    """Build PLE shard loaders whose shape is known but whose payload decodes lazily.

    llama.cpp reads each shard's shape before materializing it, so the metadata
    here must be exact (float32, decoded shape) and the NVFP4 decode must sit
    inside the deferred function, which only to_eager() invokes.
    """

    def ple_weight_loader(model_name, source_name, decoded_shape):
        def load_lazy():
            return lazy_tensor_class(
                meta=lazy_tensor_class.meta_with_dtype_and_shape(
                    torch_module.float32, decoded_shape
                ),
                func=lambda: torch_module.from_numpy(
                    checkpoint.dequantized_weight(source_name)
                ),
            )

        return load_lazy

    return ple_weight_loader


class ExpertWeightStreamer:
    """Keep Qwen2Moe expert matrices off the heap until GGUF writes them."""

    def __init__(
        self, model, checkpoint, torch_module, gguf_module, model_base, expert_pattern
    ):
        self.model = model
        self.checkpoint = checkpoint
        self.torch_module = torch_module
        self.gguf_module = gguf_module
        self.model_base = model_base
        self.expert_pattern = expert_pattern
        self.expected_names = {
            name for name in model.model_tensors if expert_pattern.fullmatch(name)
        }
        if not self.expected_names:
            raise ValueError("converter exposed no Qwen2Moe expert projection weights")
        self.n_experts = model.find_hparam(["num_local_experts", "num_experts"])
        if type(self.n_experts) is not int or self.n_experts <= 0:
            raise ValueError(f"invalid expert count: {self.n_experts!r}")
        self.expected_groups: set[tuple[int, str]] = set()
        expected_ids: dict[tuple[int, str], set[int]] = {}
        for name in self.expected_names:
            match = expert_pattern.fullmatch(name)
            assert match is not None
            group = (int(match.group("layer")), match.group("projection"))
            expected_ids.setdefault(group, set()).add(int(match.group("expert")))
        expected_ids_range = set(range(self.n_experts))
        for group, expert_ids in expected_ids.items():
            if expert_ids != expected_ids_range:
                raise ValueError(
                    f"incomplete source expert set for {group}: "
                    f"expected {self.n_experts}, got {len(expert_ids)}"
                )
            self.expected_groups.add(group)
        self.registered: dict[str, tuple[str, tuple[int, ...]]] = {}
        self.seen: set[str] = set()
        self.pending: dict[
            tuple[int, str], dict[int, tuple[str, tuple[int, ...]]]
        ] = {}
        self.completed_groups: set[tuple[int, str]] = set()
        self.original_modify_tensors = model.modify_tensors
        self.original_prepare_tensors = model.prepare_tensors

    def placeholder_loader(
        self, model_name: str, source_name: str, shape: tuple[int, ...]
    ):
        match = self.expert_pattern.fullmatch(model_name)
        if match is None or model_name not in self.expected_names:
            raise ValueError(f"unexpected streamed expert tensor: {model_name}")
        if model_name in self.registered:
            raise ValueError(f"duplicate streamed expert tensor: {model_name}")
        if len(shape) != 2:
            raise ValueError(f"expert projection must be a matrix: {model_name} {shape}")
        if int(match.group("expert")) >= self.n_experts:
            raise ValueError(f"expert index outside configured range: {model_name}")
        self.registered[model_name] = (source_name, shape)
        return lambda shape=shape: self.torch_module.empty(
            shape, dtype=self.torch_module.float32, device="meta"
        )

    def assert_registered(self) -> None:
        missing = self.expected_names - self.registered.keys()
        if missing:
            raise ValueError(
                "expert projection weights are not all ModelOpt-dequantized: "
                + ", ".join(sorted(missing)[:8])
            )

    def install(self) -> None:
        self.assert_registered()
        if getattr(self.model, "fuse_gate_up_exps", False):
            raise ValueError("lazy expert streaming does not support gate/up fusion")
        self.model.modify_tensors = self.modify_tensors
        self.model.prepare_tensors = self.prepare_tensors

    def modify_tensors(self, data_torch, name: str, bid: int | None):
        registered = self.registered.get(name)
        if registered is None:
            if self.expert_pattern.fullmatch(name):
                raise ValueError(f"unregistered expert projection reached converter: {name}")
            yield from self.original_modify_tensors(data_torch, name, bid)
            return

        match = self.expert_pattern.fullmatch(name)
        assert match is not None
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        projection = match.group("projection")
        if bid != layer:
            raise ValueError(f"expert tensor block mismatch: {name}, bid={bid}")
        if name in self.seen:
            raise ValueError(f"expert tensor was processed twice: {name}")
        self.seen.add(name)

        group_key = (layer, projection)
        group = self.pending.setdefault(group_key, {})
        if expert in group:
            raise ValueError(f"duplicate expert {expert} in {group_key}")
        group[expert] = registered
        if len(group) < self.n_experts:
            return
        if set(group) != set(range(self.n_experts)):
            raise ValueError(f"non-contiguous expert indices for {group_key}")

        shapes = {shape for _source, shape in group.values()}
        if len(shapes) != 1:
            raise ValueError(f"expert matrix shapes differ for {group_key}")
        shape = next(iter(shapes))
        chunks = []
        for expert_id in range(self.n_experts):
            source_name, expected_shape = group[expert_id]

            def load_chunk(source_name=source_name, expected_shape=expected_shape):
                decoded = self.checkpoint.dequantized_weight(source_name)
                if tuple(decoded.shape) != expected_shape:
                    raise ValueError(
                        f"dequantized expert shape mismatch for {source_name}: "
                        f"{tuple(decoded.shape)} != {expected_shape}"
                    )
                return decoded

            chunks.append(load_chunk)

        lazy_tensor = self.gguf_module.LazyChunkedTensor(
            chunks, (self.n_experts, *shape), "float32"
        )
        del self.pending[group_key]
        self.completed_groups.add(group_key)
        merged_name = (
            f"model.layers.{layer}.mlp.experts.{projection}.weight"
        )
        yield from self.model_base.modify_tensors(
            self.model, lazy_tensor, merged_name, layer
        )

    def prepare_tensors(self) -> None:
        self.original_prepare_tensors()
        self.assert_complete()

    def assert_complete(self) -> None:
        self.assert_registered()
        missing = set(self.registered) - self.seen
        if missing:
            raise ValueError(
                "expert projection weights were skipped by converter: "
                + ", ".join(sorted(missing)[:8])
            )
        if self.pending:
            raise ValueError(f"incomplete streamed expert groups: {sorted(self.pending)}")
        if self.completed_groups != self.expected_groups:
            missing_groups = self.expected_groups - self.completed_groups
            raise ValueError(f"expert projection groups were not emitted: {sorted(missing_groups)}")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--llama-cpp-root", required=True, type=Path)
    parser.add_argument("--llama-quantize", type=Path)
    parser.add_argument(
        "--type-reference",
        required=True,
        nargs="+",
        type=Path,
        help="baseline GGUF shard(s) whose per-tensor types the output must reproduce",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="instantiate the pinned model and validate tensor replacements without writing GGUF",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint.resolve(strict=True)
    llama_root = args.llama_cpp_root.resolve(strict=True)
    quantizer = (
        args.llama_quantize.resolve(strict=True) if args.llama_quantize else None
    )
    output_dir = args.output_dir.resolve()
    if not args.preflight_only:
        if quantizer is None:
            raise ValueError("--llama-quantize is required unless running preflight")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(output_dir)

    repo_tools = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_tools))
    sys.path.insert(0, str(llama_root / "gguf-py"))
    sys.path.insert(0, str(llama_root))
    import torch
    import gguf
    from conversion import ModelBase, ModelType, get_model_architecture, get_model_class
    from conversion.base import LazyTorchTensor
    from gufo.qwen_fp4_local import (
        EXPERT_WEIGHT_PATTERN,
        Checkpoint,
        prepare_modelopt_tensors,
    )

    source = Checkpoint(checkpoint_dir)
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text("utf-8"))
    recipes = source.config["quantization_config"]["quantized_layers"]
    recipe_counts = Counter(
        recipe.get("quant_algo")
        for recipe in recipes.values()
        if isinstance(recipe, dict)
    )
    actual_shape = (
        len(source.weight_map),
        len(set(source.weight_map.values())),
        index.get("metadata", {}).get("total_size"),
    )
    expected_shape = (
        EXPECTED_TENSOR_COUNT,
        EXPECTED_SHARD_COUNT,
        EXPECTED_SOURCE_BYTES,
    )
    if actual_shape != expected_shape or dict(recipe_counts) != EXPECTED_RECIPE_COUNTS:
        raise ValueError(
            f"checkpoint identity mismatch: shape={actual_shape}, "
            f"recipes={dict(recipe_counts)}"
        )
    index_sha256 = sha256_file(checkpoint_dir / "model.safetensors.index.json")
    config_bytes = (checkpoint_dir / "config.json").read_bytes()
    config_blob_sha1 = hashlib.sha1(
        b"blob %d\0" % len(config_bytes) + config_bytes
    ).hexdigest()
    if (index_sha256, config_blob_sha1) != (
        EXPECTED_INDEX_SHA256,
        EXPECTED_CONFIG_GIT_BLOB_SHA1,
    ):
        raise ValueError(
            f"checkpoint is not {REPO_ID}@{REVISION}: index sha256 {index_sha256}, "
            f"config git blob {config_blob_sha1}"
        )

    bf16_path = output_dir / "qwen38-flash-next-nvidia-fp4-local-bf16.gguf"
    int4_path = output_dir / "qwen38-flash-next-nvidia-fp4-local-int4.gguf"
    if bf16_path.exists() or int4_path.exists():
        raise FileExistsError("refusing to overwrite an existing conversion artifact")

    # Read config.json directly, as the pinned converter expects. A transformers
    # release that knows qwen4_exp (5.19 does) relabels full_attention layers as
    # indexed_attention in AutoConfig, and qwen4exp.py then writes zero compress
    # ratios that the runtime refuses to load.
    hparams = json.loads((checkpoint_dir / "config.json").read_text("utf-8"))
    compress_ratios = expected_compress_ratios(hparams)
    architecture = get_model_architecture(hparams, ModelType.TEXT)
    if architecture != "Qwen4ExpForConditionalGeneration":
        raise ValueError(f"unexpected text architecture: {architecture}")
    model_class = get_model_class(architecture)
    if model_class.__name__ != "Qwen4ExpTextModel":
        raise ValueError(f"unexpected converter class: {model_class.__name__}")
    hparams = dict(hparams)
    hparams.pop("quantization_config", None)

    view_parent = Path(
        tempfile.mkdtemp(prefix="qwen4exp-converter-view-", dir="/tmp")
    )
    try:
        converter_dir = make_converter_view(checkpoint_dir, view_parent)
        model = model_class(
            converter_dir,
            gguf.LlamaFileType.MOSTLY_BF16,
            bf16_path,
            hparams=hparams,
            use_temp_file=True,
        )
        streamer = ExpertWeightStreamer(
            model, source, torch, gguf, ModelBase, EXPERT_WEIGHT_PATTERN
        )
        converted = prepare_modelopt_tensors(
            model,
            source,
            torch,
            expert_weight_loader=streamer.placeholder_loader,
            ple_weight_loader=make_ple_weight_loader(source, torch, LazyTorchTensor),
        )
        streamer.assert_registered()
        streamer.install()
        print(f"Replaced ModelOpt payloads: {converted}", flush=True)
        if args.preflight_only:
            print(
                f"Preflight passed: model_tensors={len(model.model_tensors)}, "
                f"replacements={converted}",
                flush=True,
            )
            return
        model.write()
    finally:
        shutil.rmtree(view_parent)
    if not bf16_path.is_file() or bf16_path.stat().st_size == 0:
        raise RuntimeError("llama.cpp did not produce the BF16 GGUF")

    reference_paths = [path.resolve(strict=True) for path in args.type_reference]
    reference_types = reference_tensor_types(gguf, reference_paths)
    type_file = output_dir / "tensor-types.txt"
    write_tensor_type_file(reference_types, type_file)
    command = [
        str(quantizer),
        "--tensor-type-file",
        str(type_file),
        str(bf16_path),
        str(int4_path),
        "Q4_K_M",
    ]
    print(
        f"Quantizing with Q4_K_M, per-tensor types from {len(reference_paths)} reference GGUF(s)",
        flush=True,
    )
    subprocess.run(command, check=True)
    if not int4_path.is_file() or int4_path.stat().st_size == 0:
        raise RuntimeError("llama-quantize did not produce the INT4 GGUF")
    type_counts = verify_tensor_types(gguf, int4_path, reference_types)
    for path in (bf16_path, int4_path):
        verify_compress_ratios(gguf, path, compress_ratios)

    manifest = {
        "source": {"repo": REPO_ID, "revision": REVISION},
        "source_checkpoint": {
            "config_sha256": sha256_file(checkpoint_dir / "config.json"),
            "index_sha256": sha256_file(
                checkpoint_dir / "model.safetensors.index.json"
            ),
            "indexed_tensors": len(source.weight_map),
            "shards": len(set(source.weight_map.values())),
            "source_bytes": index["metadata"]["total_size"],
            "recipe_counts": dict(recipe_counts),
        },
        "converter": {
            "llama_cpp_revision": "68d9053afd4f4d0752ced6187585f862355a40be",
            "converted_modelopt_tensors": converted,
            "writer_uses_disk_backed_tempfile": True,
        },
        "quantization": {
            "format": "Q4_K_M",
            "type_reference": {
                "paths": [str(path) for path in reference_paths],
                "sha256": [sha256_file(path) for path in reference_paths],
            },
            "tensor_type_file_sha256": sha256_file(type_file),
            "verified_tensor_types_match_reference": True,
            "verified_type_counts": type_counts,
            "verified_compress_ratios": compress_ratios,
        },
        "artifacts": {
            "bf16": {
                "path": bf16_path.name,
                "bytes": bf16_path.stat().st_size,
                "sha256": sha256_file(bf16_path),
            },
            "int4": {
                "path": int4_path.name,
                "bytes": int4_path.stat().st_size,
                "sha256": sha256_file(int4_path),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = output_dir / ".manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    temporary_manifest.replace(manifest_path)
    print(f"Conversion complete: {int4_path}", flush=True)
    print(f"INT4 SHA-256: {manifest['artifacts']['int4']['sha256']}", flush=True)


if __name__ == "__main__":
    main()
