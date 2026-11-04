# xgrammar (vendored)

Grammar-constrained decoding for structured outputs
([`src/core/grammar`](../../core/grammar/grammar.hpp)) uses the C++ core of
[xgrammar](https://github.com/mlc-ai/xgrammar), Apache-2.0.

| Component | Upstream | Pinned revision | License |
| --- | --- | --- | --- |
| xgrammar | `mlc-ai/xgrammar` tag `v0.2.8` | `97787376faee5ed8466cfad57c99855e4ce2f6aa` | Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)) |
| DLPack header | `dmlc/dlpack` (xgrammar's submodule pin) | `bbd2f4d32427e548797929af08cfe2a9cbb3cf12` | Apache-2.0 ([LICENSE](3rdparty/dlpack/LICENSE)) |
| picojson | `3rdparty/picojson/picojson.h` from the xgrammar tag | as above | BSD-2-Clause (in the header) |

## Import boundary

Copied unmodified from the pinned tag:

- `include/xgrammar/*.h`
- `cpp/**/*.cc` and `cpp/**/*.h`, except `cpp/tvm_ffi/` (Python bindings)
- `3rdparty/picojson/picojson.h`
- `3rdparty/dlpack/include/dlpack/dlpack.h` and its `LICENSE`
- `LICENSE` and `NOTICE`

The root `CMakeLists.txt` builds these as the private C++17 target
`gufo_xgrammar`, without Python bindings, cpptrace or upstream tests. The
format check skips `vendor` paths; keep the files byte-identical to upstream.

## Updating

```sh
git clone --branch vX.Y.Z https://github.com/mlc-ai/xgrammar /tmp/xgrammar
git -C /tmp/xgrammar submodule update --init 3rdparty/dlpack
```

Replace the files listed above, update the revisions here and in
`THIRD_PARTY_NOTICES.md`, then run `grammar_test` and
`structured_output_test`. Structural-tag JSON (`structured_output.cpp`)
follows xgrammar's `qwen_3_5` and `mimo` builtin tool-call tags; recheck them
against the new tag's `python/xgrammar/builtin_structural_tag.py`.
