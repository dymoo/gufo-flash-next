"""Numerical decoding primitives for NVIDIA's Qwen3.8 FP4-Local export.

The checkpoint stores ModelOpt NVFP4 and MXFP8 payloads in Safetensors. These
helpers decode the source values without depending on NVIDIA CUDA kernels.
"""

from __future__ import annotations

import json
import math
import struct
import re
from collections.abc import Callable
from pathlib import Path

EXPERT_WEIGHT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>down_proj|gate_proj|up_proj)\.weight$"
)
PLE_SHARD_WEIGHT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.ple\.ple_embedding\.ngram_embedding\."
    r"shard_(?P<index>\d+)\.weight$"
)



import numpy as np


def decode_scale_bytes(fixed: np.ndarray, exceptions: np.ndarray) -> np.ndarray:
    """Reconstruct E4M3 scale bytes from ModelOpt's CSF byte-window stream."""
    fixed = np.asarray(fixed)
    exceptions = np.asarray(exceptions)
    if fixed.dtype != np.dtype("uint8") or fixed.ndim != 2:
        raise ValueError("CSF fixed stream must be a 2D uint8 tensor")
    slabs, width = fixed.shape
    if slabs == 0 or width <= 16 or width % 16:
        raise ValueError("CSF fixed stream has invalid slab dimensions")
    if exceptions.dtype != np.dtype("<u4") or exceptions.ndim != 1:
        raise ValueError("CSF exceptions must be a 1D little-endian uint32 tensor")

    rows = slabs * 16
    cols = (width // 16 - 1) * 2
    bases = fixed[:, :16].reshape(rows, 1)
    if np.any(bases > 240):
        raise ValueError("CSF row bases must be at most 240")
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    result = np.empty((rows, cols), dtype=np.uint8)
    result[:, 0::2] = bases + (packed & 15)
    result[:, 1::2] = bases + (packed >> 4)

    positions = exceptions & np.uint32(0xFFFFFF)
    if positions.size:
        if np.any(positions >= result.size):
            raise ValueError("CSF exception position is outside the scale matrix")
        if np.any(positions[1:] <= positions[:-1]):
            raise ValueError("CSF exception positions must be strictly increasing")
        result.reshape(-1)[positions] = (exceptions >> 24).astype(np.uint8)
    if np.any(result & np.uint8(0x80)) or np.any(result >= 127):
        raise ValueError("CSF scale bytes must be finite non-negative E4M3")
    return result


def e4m3_values(bits: np.ndarray) -> np.ndarray:
    """Decode E4M3FN bit patterns to float32, including subnormals."""
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> 3) & 15).astype(np.int32)
    mantissa = (bits & 7).astype(np.float32)
    values = np.where(
        exponent == 0,
        mantissa * np.float32(2**-9),
        (1 + mantissa / 8) * np.exp2(exponent - 7),
    ).astype(np.float32)
    values = np.copysign(values, np.where(bits & 128, -1.0, 1.0))
    return np.where((bits & 127) == 127, np.nan, values).astype(np.float32)


def dequant_nvfp4(
    weight: np.ndarray,
    scales: np.ndarray,
    scale2: float,
    *,
    chunk_rows: int = 16,
) -> np.ndarray:
    """Decode NVFP4 in bounded row chunks to avoid full-tensor temporaries."""
    weight = np.asarray(weight)
    scales = np.asarray(scales)
    if weight.dtype != np.dtype("uint8") or scales.dtype != np.dtype("uint8"):
        raise ValueError("NVFP4 codes and scale bits must be uint8")
    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError("NVFP4 weights and scales must be matrices")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if weight.shape[1] % 8 or scales.shape != (
        weight.shape[0], weight.shape[1] // 8
    ):
        raise ValueError("NVFP4 needs one block scale per 16 unpacked values")
    if not np.isfinite(scale2) or scale2 <= 0:
        raise ValueError("NVFP4 global scale must be finite and positive")

    rows, packed_columns = weight.shape
    block_count = packed_columns // 8
    result = np.empty((rows, packed_columns * 2), dtype=np.float32)
    codebook = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=np.float32,
    )
    for row_start in range(0, rows, chunk_rows):
        row_end = min(row_start + chunk_rows, rows)
        scale_bits = scales[row_start:row_end]
        if np.any(scale_bits & np.uint8(0x80)) or np.any(scale_bits >= 127):
            raise ValueError("NVFP4 scales must be finite non-negative E4M3")
        scale_values = e4m3_values(scale_bits)
        if not np.isfinite(scale_values).all():
            raise ValueError("NVFP4 scale contains a non-finite E4M3 value")

        packed = weight[row_start:row_end]
        codes = np.empty((row_end - row_start, packed_columns * 2), dtype=np.uint8)
        codes[:, 0::2] = packed & np.uint8(15)
        codes[:, 1::2] = packed >> np.uint8(4)
        values = result[row_start:row_end]
        np.take(codebook, codes, out=values)
        values.reshape(row_end - row_start, block_count, 16)[:] *= (
            scale_values[:, :, None]
        )
        values *= np.float32(scale2)
        if not np.isfinite(values).all():
            raise ValueError("NVFP4 dequantization produced a non-finite value")
    return result

def dequant_mxfp8(
    weight: np.ndarray, scales: np.ndarray, *, chunk_rows: int = 16
) -> np.ndarray:
    """Decode MXFP8 in bounded row chunks to avoid full-tensor temporaries."""
    weight = np.asarray(weight)
    scales = np.asarray(scales)
    if weight.dtype != np.dtype("uint8") or scales.dtype != np.dtype("uint8"):
        raise ValueError("MXFP8 weight and scales must be uint8")
    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError("MXFP8 weights and scales must be matrices")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if weight.shape[1] % 32 or scales.shape != (
        weight.shape[0], weight.shape[1] // 32
    ):
        raise ValueError("MXFP8 needs one E8M0 scale per 32 values")

    rows, columns = weight.shape
    block_count = columns // 32
    result = np.empty(weight.shape, dtype=np.float32)
    for row_start in range(0, rows, chunk_rows):
        row_end = min(row_start + chunk_rows, rows)
        scale_bits = scales[row_start:row_end]
        if np.any(scale_bits == 255):
            raise ValueError("MXFP8 scale has a NaN encoding")
        with np.errstate(over="ignore", invalid="ignore"):
            scale_values = np.exp2(
                scale_bits.astype(np.float32) - np.float32(127)
            )
            values = result[row_start:row_end].reshape(
                row_end - row_start, block_count, 32
            )
            values[:] = e4m3_values(weight[row_start:row_end]).reshape(
                row_end - row_start, block_count, 32
            )
            values *= scale_values[:, :, None]
        if not np.isfinite(values).all():
            raise ValueError("MXFP8 dequantization produced a non-finite value")
    return result


_SAFE_TENSOR_DTYPES = {
    "U8": (np.dtype("u1"), 1),
    "I8": (np.dtype("i1"), 1),
    "F8_E4M3": (np.dtype("u1"), 1),
    "F8_E4M3FN": (np.dtype("u1"), 1),
    "F8_E5M2": (np.dtype("u1"), 1),
    "U16": (np.dtype("<u2"), 2),
    "I16": (np.dtype("<i2"), 2),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
    "U32": (np.dtype("<u4"), 4),
    "I32": (np.dtype("<i4"), 4),
    "F32": (np.dtype("<f4"), 4),
    "U64": (np.dtype("<u8"), 8),
    "I64": (np.dtype("<i8"), 8),
    "F64": (np.dtype("<f8"), 8),
}


class Checkpoint:
    """Bounded Safetensors reader for ModelOpt Qwen4-Exp weight tensors."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)
        config_path = self.root / "config.json"
        index_path = self.root / "model.safetensors.index.json"
        self.config = json.loads(config_path.read_text("utf-8"))
        if not isinstance(self.config, dict) or self.config.get("model_type") != "qwen4_exp":
            raise ValueError("checkpoint is not a Qwen4-Exp model")
        index = json.loads(index_path.read_text("utf-8"))
        if not isinstance(index, dict):
            raise ValueError("Safetensors index must be an object")
        weight_map = index.get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(not isinstance(shard, str) for shard in weight_map.values())
        ):
            raise ValueError("Safetensors index has no valid weight map")
        self.weight_map = weight_map
        self.model_to_source: dict[str, str] = {}
        self.source_to_model: dict[str, str] = {}
        for source_name in weight_map:
            model_name = source_name.replace("language_model.", "")
            if model_name.endswith("e_score_correction_bias"):
                model_name = model_name.replace(
                    "e_score_correction_bias", "e_score_correction.bias"
                )
            if model_name in self.model_to_source:
                raise ValueError(f"duplicate normalized tensor name: {model_name}")
            self.model_to_source[model_name] = source_name
            self.source_to_model[source_name] = model_name
        self._headers: dict[str, dict] = {}
        for shard in set(weight_map.values()):
            path = (self.root / shard).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"Safetensors shard path escapes checkpoint: {shard}")
            if not path.is_file():
                raise ValueError(f"missing Safetensors shard: {shard}")

    def _shard_path(self, tensor_name: str) -> Path:
        try:
            shard = self.weight_map[tensor_name]
        except KeyError as error:
            raise ValueError(f"tensor is not indexed: {tensor_name}") from error
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Safetensors shard path escapes checkpoint: {shard}")
        return path

    def _header(self, path: Path) -> dict:
        key = str(path)
        if key in self._headers:
            return self._headers[key]
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated Safetensors header: {path.name}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size == 0 or header_size > min(100 << 20, size - 8):
                raise ValueError(f"invalid Safetensors header length: {path.name}")
            try:
                header = json.loads(stream.read(header_size))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid Safetensors header: {path.name}") from error
        if not isinstance(header, dict):
            raise ValueError(f"invalid Safetensors header object: {path.name}")
        self._headers[key] = header
        return header

    def read_array(self, tensor_name: str) -> np.ndarray:
        """Map one indexed tensor after validating its header and byte range."""
        path = self._shard_path(tensor_name)
        try:
            descriptor = self._header(path)[tensor_name]
            dtype_name = descriptor["dtype"]
            shape = descriptor["shape"]
            begin, end = descriptor["data_offsets"]
            dtype, itemsize = _SAFE_TENSOR_DTYPES[dtype_name]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid Safetensors descriptor: {tensor_name}") from error
        if (
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or type(begin) is not int
            or type(end) is not int
            or begin < 0
            or end < begin
        ):
            raise ValueError(f"invalid Safetensors shape or range: {tensor_name}")
        elements = math.prod(shape) if shape else 1
        if elements * itemsize != end - begin:
            raise ValueError(f"Safetensors byte count does not match shape: {tensor_name}")
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated Safetensors header: {path.name}")
            header_size = struct.unpack("<Q", prefix)[0]
            offset = 8 + header_size + begin
            if offset + end - begin > path.stat().st_size:
                raise ValueError(f"Safetensors data range exceeds shard: {tensor_name}")
        if begin == end:
            return np.empty(shape, dtype=dtype)
        if dtype_name in {"F8_E4M3", "F8_E4M3FN", "F8_E5M2"}:
            dtype = np.dtype("u1")
        return np.memmap(
            path, mode="r", dtype=dtype, offset=offset, shape=tuple(shape), order="C"
        )

    def raw_shape(self, tensor_name: str) -> tuple[int, ...]:
        """Return an indexed tensor's Safetensors shape without mapping its payload."""
        path = self._shard_path(tensor_name)
        try:
            shape = self._header(path)[tensor_name]["shape"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid Safetensors descriptor: {tensor_name}") from error
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise ValueError(f"invalid Safetensors shape: {tensor_name}")
        return tuple(shape)

    def _recipe_entry(self, tensor_name: str) -> tuple[str, dict]:
        quantization = self.config.get("quantization_config", {})
        if not isinstance(quantization, dict):
            raise ValueError("ModelOpt quantization config must be an object")
        recipes = quantization.get("quantized_layers", {})
        if not isinstance(recipes, dict):
            raise ValueError("ModelOpt quantized_layers must be an object")
        parts = tensor_name.split(".")
        for end in range(len(parts), 0, -1):
            prefix = ".".join(parts[:end])
            if prefix in recipes:
                recipe = recipes[prefix]
                if not isinstance(recipe, dict):
                    raise ValueError(f"invalid ModelOpt recipe for {prefix}")
                return prefix, recipe
        raise ValueError(f"no ModelOpt quantization recipe for {tensor_name}")

    def recipe_for(self, tensor_name: str) -> dict:
        return self._recipe_entry(tensor_name)[1]

    def nvfp4_global_scale_name(self, tensor_name: str) -> str:
        """Name of the F32 global scale for an NVFP4 weight.

        Most weights carry ``<stem>.weight_scale_2``. Sharded tables (the PLE
        ``ngram_embedding.shard_N``) share one scale on the module that owns the
        quantization recipe. A missing scale is an error, never an implicit 1.0.
        """
        own = tensor_name.removesuffix(".weight") + ".weight_scale_2"
        if own in self.weight_map:
            return own
        module = self._recipe_entry(tensor_name)[0] + ".weight_scale_2"
        if module in self.weight_map:
            return module
        raise ValueError(f"NVFP4 weight has no global scale: {tensor_name}")

    def _nvfp4_global_scale(self, tensor_name: str) -> float:
        scale2 = self.read_array(self.nvfp4_global_scale_name(tensor_name))
        if scale2.size != 1 or scale2.dtype != np.dtype("<f4"):
            raise ValueError(f"NVFP4 global scale must be one F32 value: {tensor_name}")
        return float(scale2.reshape(-1)[0])

    def dequantized_weight(self, tensor_name: str) -> np.ndarray:
        """Decode an NVFP4 or MXFP8 weight together with its checkpoint scales."""
        recipe = self.recipe_for(tensor_name)
        algorithm = recipe.get("quant_algo")
        stem = tensor_name.removesuffix(".weight")
        if algorithm == "NVFP4":
            packed = self.read_array(tensor_name)
            if recipe.get("weight_scale_encoding") == "csf":
                fixed = self.read_array(stem + ".weight_scale.nvfp4_csf_fixed")
                exceptions = self.read_array(
                    stem + ".weight_scale.nvfp4_csf_exceptions"
                )
                scales = decode_scale_bytes(fixed, exceptions)
                global_scale = self._nvfp4_global_scale(tensor_name)
            elif recipe.get("weight_scale_encoding") is None:
                scales = self.read_array(stem + ".weight_scale")
                global_scale = self._nvfp4_global_scale(tensor_name)
            else:
                raise ValueError(f"unsupported NVFP4 scale encoding: {tensor_name}")
            return dequant_nvfp4(packed, scales, global_scale)
        if algorithm == "MXFP8":
            weight = self.read_array(tensor_name)
            if weight.dtype != np.dtype("uint8"):
                raise ValueError(f"MXFP8 weight payload must be byte-addressable: {tensor_name}")
            scales = self.read_array(stem + ".weight_scale")
            return dequant_mxfp8(weight, scales)
        raise ValueError(f"unsupported ModelOpt quantization {algorithm!r}: {tensor_name}")


def prepare_modelopt_tensors(
    model,
    checkpoint: Checkpoint,
    torch_module,
    *,
    expert_weight_loader: Callable[
        [str, str, tuple[int, ...]], Callable[[], object]
    ] | None = None,
    ple_weight_loader: Callable[
        [str, str, tuple[int, ...]], Callable[[], object]
    ] | None = None,
) -> dict[str, int]:
    """Replace ModelOpt loaders, streaming expert matrices and PLE shards."""
    replacements: dict[str, int] = {}
    auxiliary_names: set[str] = set()
    for model_name in model.model_tensors:
        source_name = checkpoint.model_to_source.get(model_name)
        if source_name is None or not source_name.endswith(".weight"):
            continue
        try:
            recipe = checkpoint.recipe_for(source_name)
        except ValueError as error:
            if str(error).startswith("no ModelOpt quantization recipe"):
                continue
            raise
        algorithm = recipe.get("quant_algo")
        if algorithm not in {"NVFP4", "MXFP8"}:
            raise ValueError(
                f"unsupported ModelOpt quantization {algorithm!r}: {source_name}"
            )
        source_stem = source_name.removesuffix(".weight")
        helper_suffixes = [".weight_scale"]
        if algorithm == "NVFP4":
            if recipe.get("weight_scale_encoding") not in (None, "csf"):
                raise ValueError(f"unsupported NVFP4 scale encoding: {source_name}")
            helper_suffixes.extend(
                [
                    ".weight_scale.nvfp4_csf_fixed",
                    ".weight_scale.nvfp4_csf_exceptions",
                    ".input_scale",
                ]
            )
        helper_sources = [source_stem + suffix for suffix in helper_suffixes]
        if algorithm == "NVFP4":
            helper_sources.append(checkpoint.nvfp4_global_scale_name(source_name))
        for helper_source in helper_sources:
            helper_model = checkpoint.source_to_model.get(helper_source)
            if helper_model in model.model_tensors:
                auxiliary_names.add(helper_model)
        expert_match = EXPERT_WEIGHT_PATTERN.fullmatch(model_name)
        ple_match = PLE_SHARD_WEIGHT_PATTERN.fullmatch(model_name)
        if ple_match is not None:
            if algorithm != "NVFP4":
                raise ValueError(f"PLE shard must be NVFP4: {source_name}")
            if ple_weight_loader is None:
                raise ValueError(f"PLE shard needs a lazy weight loader: {source_name}")
            raw_shape = checkpoint.raw_shape(source_name)
            if len(raw_shape) != 2 or raw_shape[1] % 8:
                raise ValueError(
                    f"invalid packed NVFP4 PLE shard shape: {source_name} {raw_shape}"
                )
            loader = ple_weight_loader(
                model_name, source_name, (raw_shape[0], raw_shape[1] * 2)
            )
            if not callable(loader):
                raise TypeError(f"PLE loader is not callable: {model_name}")
            model.model_tensors[model_name] = loader
        elif expert_match is not None and expert_weight_loader is not None:
            raw_shape = tuple(checkpoint.read_array(source_name).shape)
            if algorithm == "NVFP4":
                if len(raw_shape) != 2 or raw_shape[1] % 8:
                    raise ValueError(
                        f"invalid packed NVFP4 expert matrix shape: {source_name} {raw_shape}"
                    )
                shape = (raw_shape[0], raw_shape[1] * 2)
            else:
                shape = raw_shape
            loader = expert_weight_loader(model_name, source_name, shape)
            if not callable(loader):
                raise TypeError(f"expert loader is not callable: {model_name}")
            model.model_tensors[model_name] = loader
        else:
            model.model_tensors[model_name] = (
                lambda name=source_name: torch_module.from_numpy(
                    checkpoint.dequantized_weight(name)
                )
            )
        replacements[algorithm] = replacements.get(algorithm, 0) + 1

    if not replacements.get("NVFP4") or not replacements.get("MXFP8"):
        raise ValueError(
            f"expected NVFP4 experts and MXFP8 projections, got {replacements}"
        )
    for tensor_name in auxiliary_names:
        del model.model_tensors[tensor_name]
    replacements["auxiliary_tensors_removed"] = len(auxiliary_names)
    return replacements
