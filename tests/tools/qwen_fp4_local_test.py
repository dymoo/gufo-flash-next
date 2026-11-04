#!/usr/bin/env python3
"""Format-level tests for the NVIDIA Qwen FP4-Local converter."""

from __future__ import annotations

import re
import sys
import unittest
import json
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gufo import qwen_fp4_local  # noqa: E402
sys.path.insert(0, str(ROOT / "tools" / "qwen-flash"))
import convert_nvidia_fp4_local as nvidia_converter  # noqa: E402


class QwenFp4LocalCodecTest(unittest.TestCase):
    def test_csf_reconstructs_base_plus_nibbles_and_exception(self):
        fixed = np.full((1, 144), 0x20, dtype=np.uint8)
        fixed[:, 16:] = 0x21
        exceptions = np.array([0x23000005], dtype="<u4")

        scales = qwen_fp4_local.decode_scale_bytes(fixed, exceptions)

        self.assertEqual(scales.shape, (16, 16))
        np.testing.assert_array_equal(scales[:, 0], np.full(16, 0x21))
        np.testing.assert_array_equal(scales[:, 1], np.full(16, 0x22))
        self.assertEqual(scales.reshape(-1)[5], 0x23)

    def test_csf_rejects_bad_streams(self):
        fixed = np.full((1, 144), 0x20, dtype=np.uint8)
        unsorted = np.array([0x22000002, 0x23000001], dtype="<u4")
        out_of_range = np.array([0x22000100], dtype="<u4")

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            qwen_fp4_local.decode_scale_bytes(fixed, unsorted)
        with self.assertRaisesRegex(ValueError, "outside"):
            qwen_fp4_local.decode_scale_bytes(fixed, out_of_range)
        with self.assertRaisesRegex(ValueError, "E4M3"):
            qwen_fp4_local.decode_scale_bytes(
                np.full((1, 144), 0x7F, dtype=np.uint8), np.array([], dtype="<u4")
            )

    def test_nvfp4_codebook_and_global_scale(self):
        packed = np.array([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=np.uint8)
        scales = np.array([[0x38]], dtype=np.uint8)

        decoded = qwen_fp4_local.dequant_nvfp4(packed, scales, 2.0)

        np.testing.assert_array_equal(
            decoded,
            [[0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]],
        )


    def test_nvfp4_chunked_decode_across_row_boundaries(self):
        rng = np.random.default_rng(154)
        packed = rng.integers(0, 256, size=(37, 32), dtype=np.uint8)
        scales = rng.integers(0x20, 0x50, size=(37, 4), dtype=np.uint8)
        scale2 = 1.125

        decoded = qwen_fp4_local.dequant_nvfp4(
            packed, scales, scale2, chunk_rows=7
        )
        codebook = np.array(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=np.float32,
        )
        expected = np.empty((37, 64), dtype=np.float32)
        for row in range(37):
            for column in range(64):
                byte = int(packed[row, column // 2])
                code = (byte >> (4 * (column % 2))) & 15
                scale = qwen_fp4_local.e4m3_values(
                    np.array([scales[row, column // 16]], dtype=np.uint8)
                )[0]
                expected[row, column] = codebook[code] * scale * np.float32(scale2)

        np.testing.assert_allclose(decoded, expected, rtol=1e-6, atol=0)

    def test_mxfp8_chunked_decode_across_row_boundaries(self):
        rng = np.random.default_rng(271)
        weights = rng.integers(0, 256, size=(19, 64), dtype=np.uint8)
        weights[(weights == 127) | (weights == 255)] = 0
        scales = rng.integers(120, 135, size=(19, 2), dtype=np.uint8)

        decoded = qwen_fp4_local.dequant_mxfp8(
            weights, scales, chunk_rows=5
        )
        expected = np.empty((19, 64), dtype=np.float32)
        for row in range(19):
            for column in range(64):
                value = qwen_fp4_local.e4m3_values(
                    np.array([weights[row, column]], dtype=np.uint8)
                )[0]
                scale = np.exp2(np.float32(scales[row, column // 32]) - 127)
                expected[row, column] = value * scale

        np.testing.assert_allclose(decoded, expected, rtol=1e-6, atol=0)
    def test_nvfp4_rejects_wrong_scale_shape_and_nonfinite_scale(self):
        packed = np.zeros((1, 8), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "one block scale"):
            qwen_fp4_local.dequant_nvfp4(packed, np.zeros((1, 2), dtype=np.uint8), 1.0)
        with self.assertRaisesRegex(ValueError, "global scale"):
            qwen_fp4_local.dequant_nvfp4(
                packed, np.full((1, 1), 0x38, dtype=np.uint8), float("nan")
            )

    def test_mxfp8_decodes_e4m3_values_with_e8m0_scales(self):
        weights = np.array([[0x38, 0x40, 0xB8, 0xC0] * 8], dtype=np.uint8)
        scales = np.full((1, 1), 127, dtype=np.uint8)

        decoded = qwen_fp4_local.dequant_mxfp8(weights, scales)

        np.testing.assert_array_equal(decoded[0, :4], [1, 2, -1, -2])

    def test_mxfp8_rejects_nan_scale_encoding(self):
        weights = np.full((1, 32), 0x38, dtype=np.uint8)
        scales = np.full((1, 1), 255, dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "NaN"):
            qwen_fp4_local.dequant_mxfp8(weights, scales)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    dtype_map = {
        "U8": np.dtype("u1"),
        "U32": np.dtype("<u4"),
        "F32": np.dtype("<f4"),
        "F8_E4M3": np.dtype("u1"),
    }
    header = {}
    payload = bytearray()
    for name, (dtype, value) in tensors.items():
        array = np.asarray(value, dtype=dtype_map[dtype])
        raw = array.tobytes()
        begin = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [begin, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


PLE_MODULE_SCALE = (
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2"
)


def make_checkpoint(root: Path) -> tuple[str, str]:
    fp4 = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    mxfp8 = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    ple_stem = ple.removesuffix(".weight")
    fixed = np.full((1, 32), 0, dtype=np.uint8)
    fixed[:, :16] = 0x38
    tensors = {
        fp4: ("U8", np.full((16, 16), 0x10, dtype=np.uint8)),
        fp4.removesuffix(".weight") + ".weight_scale.nvfp4_csf_fixed": (
            "U8",
            fixed,
        ),
        fp4.removesuffix(".weight") + ".weight_scale.nvfp4_csf_exceptions": (
            "U32",
            np.array([0x38000000], dtype="<u4"),
        ),
        fp4.removesuffix(".weight") + ".weight_scale_2": (
            "F32",
            np.array(2, dtype="<f4"),
        ),
        ple: ("U8", np.full((1, 8), 0x10, dtype=np.uint8)),
        ple_stem + ".weight_scale": ("F8_E4M3", np.array([[0x40]], dtype=np.uint8)),
        # The real checkpoint stores one global scale on the ngram_embedding module,
        # shared by every PLE shard, not a per-shard weight_scale_2.
        PLE_MODULE_SCALE: ("F32", np.array(0.5, dtype="<f4")),
        mxfp8: ("U8", np.full((1, 32), 0x38, dtype=np.uint8)),
        mxfp8.removesuffix(".weight") + ".weight_scale": (
            "U8",
            np.full((1, 1), 127, dtype=np.uint8),
        ),
    }
    write_safetensors(root / "model.safetensors", tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(v[1].nbytes for v in tensors.values())},
                "weight_map": {name: "model.safetensors" for name in tensors},
            }
        ),
        "utf-8",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quantized_layers": {
                        "model.language_model.layers.0.mlp.experts": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                            "weight_scale_encoding": "csf",
                        },
                        "model.language_model.layers.0.linear_attn.in_proj_qkv": {
                            "quant_algo": "MXFP8",
                            "group_size": 32,
                        },
                        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                        },
                    },
                },
            }
        ),
        "utf-8",
    )
    return fp4, mxfp8


def make_split_ple_checkpoint(root: Path) -> tuple[str, str, str]:
    """Write two PLE NVFP4 shards and one MXFP8 projection across two Safetensors files."""
    shard0 = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    shard1 = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight"
    mxfp8 = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    first = {
        shard0: ("U8", np.full((2, 80), 0x10, dtype=np.uint8)),
        shard0.removesuffix(".weight") + ".weight_scale": (
            "F8_E4M3",
            np.full((2, 10), 0x40, dtype=np.uint8),
        ),
        mxfp8: ("U8", np.full((1, 32), 0x38, dtype=np.uint8)),
        mxfp8.removesuffix(".weight") + ".weight_scale": (
            "U8",
            np.full((1, 1), 127, dtype=np.uint8),
        ),
    }
    second = {
        shard1: ("U8", np.full((2, 80), 0x54, dtype=np.uint8)),
        shard1.removesuffix(".weight") + ".weight_scale": (
            "F8_E4M3",
            np.full((2, 10), 0x40, dtype=np.uint8),
        ),
        PLE_MODULE_SCALE: ("F32", np.array(0.5, dtype="<f4")),
    }
    write_safetensors(root / first_name, first)
    write_safetensors(root / second_name, second)
    weight_map = {name: first_name for name in first}
    weight_map.update({name: second_name for name in second})
    total_size = sum(v[1].nbytes for v in (*first.values(), *second.values()))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        "utf-8",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quantized_layers": {
                        "model.language_model.layers.0.linear_attn.in_proj_qkv": {
                            "quant_algo": "MXFP8",
                            "group_size": 32,
                        },
                        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                        },
                    },
                },
            }
        ),
        "utf-8",
    )
    return shard0, shard1, mxfp8


class _RecordingCheckpoint(qwen_fp4_local.Checkpoint):
    """Checkpoint that records every real dequantization request."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.dequantized: list[str] = []

    def dequantized_weight(self, tensor_name: str) -> np.ndarray:
        self.dequantized.append(tensor_name)
        return super().dequantized_weight(tensor_name)


class _FakeLazyTorchTensor:
    """Minimal lazy wrapper with deferred decoding and visible metadata."""

    def __init__(self, *, meta, func):
        self.meta = meta
        self.func = func
        self.shape = meta.shape
        self.dtype = meta.dtype

    @staticmethod
    def meta_with_dtype_and_shape(dtype, shape):
        return SimpleNamespace(dtype=dtype, shape=tuple(shape))

    def to_eager(self):
        array = self.func()
        if tuple(array.shape) != tuple(self.shape):
            raise ValueError(f"decoded shape {array.shape} != {self.shape}")
        if np.dtype(array.dtype) != np.dtype(self.dtype):
            raise ValueError(f"decoded dtype {array.dtype} != {self.dtype}")
        return array



def _fake_gguf(**files):
    """Fake `gguf` module whose reader serves {path-name: {tensor name: type name}}."""

    def reader(path):
        types = files[Path(path).name]
        return SimpleNamespace(
            tensors=[
                SimpleNamespace(name=name, tensor_type=SimpleNamespace(name=type_name))
                for name, type_name in types.items()
            ]
        )

    return SimpleNamespace(GGUFReader=reader)


class NvidiaConverterTypeReferenceTest(unittest.TestCase):
    REFERENCE = {
        "blk.0.attn_qkv.weight": "Q8_0",
        "blk.0.ssm_alpha.weight": "F32",
        "blk.1.ffn_down_exps.weight": "Q5_1",
        "blk.2.ffn_down_exps.weight": "Q5_1",
        "per_layer_token_embd.weight": "IQ4_NL",
    }

    def test_matching_types_pass_and_return_counts(self):
        gguf = _fake_gguf(**{"out.gguf": dict(self.REFERENCE)})

        counts = nvidia_converter.verify_tensor_types(
            gguf, Path("out.gguf"), self.REFERENCE
        )

        self.assertEqual(counts, {"Q8_0": 1, "F32": 1, "Q5_1": 2, "IQ4_NL": 1})

    def test_reference_types_merge_shards_and_reject_duplicates(self):
        gguf = _fake_gguf(
            **{
                "a.gguf": {"blk.0.attn_q.weight": "Q8_0"},
                "b.gguf": {"blk.1.attn_q.weight": "Q4_K"},
                "dup.gguf": {"blk.0.attn_q.weight": "Q4_K"},
            }
        )

        merged = nvidia_converter.reference_tensor_types(
            gguf, [Path("a.gguf"), Path("b.gguf")]
        )

        self.assertEqual(
            merged, {"blk.0.attn_q.weight": "Q8_0", "blk.1.attn_q.weight": "Q4_K"}
        )
        with self.assertRaisesRegex(ValueError, "appears twice"):
            nvidia_converter.reference_tensor_types(
                gguf, [Path("a.gguf"), Path("dup.gguf")]
            )

    def test_type_mismatch_raises(self):
        wrong = dict(self.REFERENCE, **{"blk.0.ssm_alpha.weight": "Q4_K"})
        gguf = _fake_gguf(**{"out.gguf": wrong})

        with self.assertRaisesRegex(ValueError, r"blk\.0\.ssm_alpha\.weight: got Q4_K, expected F32"):
            nvidia_converter.verify_tensor_types(gguf, Path("out.gguf"), self.REFERENCE)

    def test_missing_or_extra_tensor_raises(self):
        missing = dict(self.REFERENCE)
        del missing["per_layer_token_embd.weight"]
        extra = dict(self.REFERENCE, **{"blk.9.extra.weight": "Q4_K"})
        gguf = _fake_gguf(**{"missing.gguf": missing, "extra.gguf": extra})

        with self.assertRaisesRegex(ValueError, "per_layer_token_embd.weight: missing"):
            nvidia_converter.verify_tensor_types(
                gguf, Path("missing.gguf"), self.REFERENCE
            )
        with self.assertRaisesRegex(ValueError, r"blk\.9\.extra\.weight: unexpected"):
            nvidia_converter.verify_tensor_types(
                gguf, Path("extra.gguf"), self.REFERENCE
            )

    def test_error_reports_only_the_first_eight_mismatches(self):
        expected = {f"blk.{i}.w": "Q8_0" for i in range(12)}
        actual = {f"blk.{i}.w": "Q4_K" for i in range(12)}
        gguf = _fake_gguf(**{"out.gguf": actual})

        with self.assertRaises(ValueError) as caught:
            nvidia_converter.verify_tensor_types(gguf, Path("out.gguf"), expected)

        message = str(caught.exception)
        self.assertIn("12 mismatches", message)
        self.assertIn("blk.5.w: got Q4_K", message)
        self.assertNotIn("blk.6.w", message)

    def test_type_file_lines_round_trip_through_fullmatch(self):
        types = dict(self.REFERENCE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensor-types.txt"
            nvidia_converter.write_tensor_type_file(types, path)
            lines = path.read_text("utf-8").splitlines()

        self.assertEqual(len(lines), len(types))
        parsed = [line.rsplit("=", 1) for line in lines]
        self.assertEqual([pattern for pattern, _ in parsed], sorted(pattern for pattern, _ in parsed))
        for pattern, type_name in parsed:
            matches = [name for name in types if re.fullmatch(pattern, name)]
            self.assertEqual(len(matches), 1, pattern)
            self.assertEqual(type_name, types[matches[0]].lower())
        # A dotted name must not match a different tensor where "." is a wildcard.
        self.assertIsNone(re.fullmatch(parsed[0][0], "blk-0-attn_qkv-weight"))


class NvidiaConverterCompressRatiosTest(unittest.TestCase):
    @staticmethod
    def config(layer_types):
        return {
            "text_config": {
                "indexer_compress_ratio": 4,
                "num_hidden_layers": len(layer_types),
                "layer_types": layer_types,
            }
        }

    def test_raw_labels_give_the_ratio_on_full_attention_layers(self):
        config = self.config(
            ["linear_attention", "linear_attention", "full_attention", "linear_attention"]
        )

        self.assertEqual(nvidia_converter.expected_compress_ratios(config), [0, 0, 4, 0])

    def test_unknown_label_such_as_indexed_attention_raises(self):
        config = self.config(["linear_attention", "indexed_attention"])

        with self.assertRaisesRegex(ValueError, "unexpected layer types"):
            nvidia_converter.expected_compress_ratios(config)


class _MetaTensor:
    def __init__(self, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device


class _LazyChunks:
    def __init__(self, chunks, shape, dtype):
        self.chunks = chunks
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def numpy(self):
        return self


class NvidiaMoeStreamingTest(unittest.TestCase):
    def test_experts_are_dequantized_only_when_the_merged_tensor_is_written(self):
        names = [
            f"model.layers.0.mlp.experts.{expert}.{projection}.weight"
            for expert in range(2)
            for projection in ("down_proj", "gate_proj", "up_proj")
        ]
        nonexpert = "model.layers.0.linear_attn.in_proj_qkv.weight"
        all_names = [*names, nonexpert]
        source_names = {
            name: "model.language_model." + name.removeprefix("model.")
            for name in all_names
        }

        class Checkpoint:
            model_to_source = {name: source for name, source in source_names.items()}
            source_to_model = {source: name for name, source in source_names.items()}

            def __init__(self):
                self.dequantized = []

            def recipe_for(self, source):
                algorithm = "MXFP8" if "linear_attn" in source else "NVFP4"
                return {"quant_algo": algorithm, "weight_scale_encoding": "csf"}

            def nvfp4_global_scale_name(self, source):
                return source.removesuffix(".weight") + ".weight_scale_2"

            def _packed_codes(self, source):
                expert = int(source.split(".experts.")[1].split(".")[0])
                code = 2 if expert == 0 else 4
                return np.full((2, 8), code | (code << 4), dtype=np.uint8)

            def read_array(self, source):
                return self._packed_codes(source)

            def dequantized_weight(self, source):
                self.dequantized.append(source)
                packed = self._packed_codes(source)
                scales = np.full((2, 1), 0x38, dtype=np.uint8)
                return qwen_fp4_local.dequant_nvfp4(packed, scales, 1.0)

        class Model:
            def __init__(self):
                self.model_tensors = {name: lambda: None for name in all_names}
                self.fuse_gate_up_exps = False
                self.prepare_calls = 0

            def find_hparam(self, _names):
                return 2

            def modify_tensors(self, _data, _name, _bid):
                raise AssertionError("streamed experts must bypass eager MoE aggregation")

            def prepare_tensors(self):
                self.prepare_calls += 1

        checkpoint = Checkpoint()
        model = Model()
        torch_module = SimpleNamespace(
            float32=np.dtype("float32"),
            empty=lambda shape, dtype, device: _MetaTensor(shape, dtype, device),
            from_numpy=lambda _array: self.fail("expert weights must not be eagerly loaded"),
        )
        gguf_module = SimpleNamespace(LazyChunkedTensor=_LazyChunks)
        model_base = SimpleNamespace(
            modify_tensors=lambda _model, data, name, _bid: [(f"mapped:{name}", data)]
        )
        streamer = nvidia_converter.ExpertWeightStreamer(
            model, checkpoint, torch_module, gguf_module, model_base,
            qwen_fp4_local.EXPERT_WEIGHT_PATTERN,
        )

        counts = qwen_fp4_local.prepare_modelopt_tensors(
            model,
            checkpoint,
            torch_module,
            expert_weight_loader=streamer.placeholder_loader,
        )
        streamer.assert_registered()
        streamer.install()

        self.assertEqual(counts["NVFP4"], 6)
        self.assertEqual(counts["MXFP8"], 1)
        self.assertEqual(checkpoint.dequantized, [])
        outputs = []
        for name in (
            names[1],
            names[5],
            names[3],
            names[0],
            names[4],
            names[2],
        ):
            meta = model.model_tensors[name]()
            self.assertEqual(meta.device, "meta")
            self.assertEqual(meta.shape, (2, 16))
            outputs.extend(model.modify_tensors(meta, name, 0))

        self.assertEqual(len(outputs), 3)
        self.assertEqual(checkpoint.dequantized, [])
        self.assertEqual(
            {name for name, _data in outputs},
            {
                "mapped:model.layers.0.mlp.experts.down_proj.weight",
                "mapped:model.layers.0.mlp.experts.gate_proj.weight",
                "mapped:model.layers.0.mlp.experts.up_proj.weight",
            },
        )
        model.prepare_tensors()
        self.assertEqual(model.prepare_calls, 1)

        for _name, data in outputs:
            self.assertIsInstance(data, _LazyChunks)
            self.assertEqual(data.shape, (2, 2, 16))
            before = len(checkpoint.dequantized)
            np.testing.assert_array_equal(data.chunks[0](), np.ones((2, 16)))
            self.assertEqual(len(checkpoint.dequantized), before + 1)
            np.testing.assert_array_equal(data.chunks[1](), np.full((2, 16), 2))
            self.assertEqual(len(checkpoint.dequantized), before + 2)


class QwenFp4LocalCheckpointTest(unittest.TestCase):
    def test_dequantizes_csf_nvfp4_and_mxfp8_from_indexed_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fp4, mxfp8 = make_checkpoint(root)
            checkpoint = qwen_fp4_local.Checkpoint(root)

            decoded_fp4 = checkpoint.dequantized_weight(fp4)
            decoded_mxfp8 = checkpoint.dequantized_weight(mxfp8)

            self.assertEqual(decoded_fp4.shape, (16, 32))
            np.testing.assert_array_equal(decoded_fp4[:, :2], [[0, 1]] * 16)
            self.assertEqual(decoded_mxfp8.shape, (1, 32))
            np.testing.assert_array_equal(decoded_mxfp8, np.ones((1, 32)))

    def test_dequantizes_direct_e4m3_scale_tensor_for_ple(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root)
            checkpoint = qwen_fp4_local.Checkpoint(root)
            ple = (
                "model.language_model.layers.1.ple.ple_embedding."
                "ngram_embedding.shard_0.weight"
            )

            decoded = checkpoint.dequantized_weight(ple)

            self.assertEqual(decoded.shape, (1, 16))
            np.testing.assert_array_equal(decoded, np.tile([0, 0.5], (1, 8)))

    def test_rejects_nvfp4_weight_without_global_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root)
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text("utf-8"))
            del index["weight_map"][PLE_MODULE_SCALE]
            index_path.write_text(json.dumps(index), "utf-8")
            checkpoint = qwen_fp4_local.Checkpoint(root)
            ple = (
                "model.language_model.layers.1.ple.ple_embedding."
                "ngram_embedding.shard_0.weight"
            )

            with self.assertRaisesRegex(ValueError, "no global scale"):
                checkpoint.dequantized_weight(ple)

    def test_ple_shards_stay_lazy_until_materialized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard0, shard1, mxfp8 = make_split_ple_checkpoint(root)
            checkpoint = _RecordingCheckpoint(root)
            self.assertNotEqual(
                checkpoint.weight_map[shard0], checkpoint.weight_map[shard1]
            )
            model_names = [
                checkpoint.source_to_model[source] for source in (shard0, shard1, mxfp8)
            ]
            scale_model_name = checkpoint.source_to_model[PLE_MODULE_SCALE]
            model = SimpleNamespace(
                model_tensors={
                    name: lambda: None for name in (*model_names, scale_model_name)
                }
            )
            torch_module = SimpleNamespace(
                float32=np.dtype(np.float32),
                from_numpy=lambda array: array,
            )
            ple_weight_loader = nvidia_converter.make_ple_weight_loader(
                checkpoint, torch_module, _FakeLazyTorchTensor
            )

            counts = qwen_fp4_local.prepare_modelopt_tensors(
                model,
                checkpoint,
                torch_module,
                ple_weight_loader=ple_weight_loader,
            )

            self.assertEqual(counts["NVFP4"], 2)
            self.assertEqual(counts["MXFP8"], 1)
            self.assertEqual(checkpoint.dequantized, [])
            self.assertEqual(counts["auxiliary_tensors_removed"], 1)
            self.assertNotIn(scale_model_name, model.model_tensors)

            lazy0 = model.model_tensors[model_names[0]]
            lazy1 = model.model_tensors[model_names[1]]
            self.assertTrue(callable(lazy0))
            self.assertTrue(callable(lazy1))
            wrapper0 = lazy0()
            wrapper1 = lazy1()
            self.assertIsInstance(wrapper0, _FakeLazyTorchTensor)
            self.assertIsInstance(wrapper1, _FakeLazyTorchTensor)
            self.assertIsNot(lazy0(), wrapper0)
            self.assertIsNot(lazy1(), wrapper1)
            self.assertEqual([wrapper0.shape, wrapper1.shape], [(2, 160), (2, 160)])
            self.assertEqual(wrapper0.dtype, torch_module.float32)
            self.assertEqual(wrapper1.dtype, torch_module.float32)
            self.assertEqual(checkpoint.dequantized, [])

            decoded0 = wrapper0.to_eager()
            self.assertEqual(checkpoint.dequantized, [shard0])
            np.testing.assert_array_equal(decoded0, np.tile([0, 0.5], (2, 80)))

            decoded1 = wrapper1.to_eager()
            self.assertEqual(checkpoint.dequantized, [shard0, shard1])
            np.testing.assert_array_equal(decoded1, np.tile([2, 3], (2, 80)))

    def test_rejects_checkpoint_index_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root)
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text("utf-8"))
            index["weight_map"]["escape.weight"] = "../outside.safetensors"
            index_path.write_text(json.dumps(index), "utf-8")

            with self.assertRaisesRegex(ValueError, "escapes"):
                qwen_fp4_local.Checkpoint(root)



if __name__ == "__main__":
    unittest.main()
