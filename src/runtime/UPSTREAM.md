# Runtime provenance

This directory is a source snapshot owned and built by `ds4f-mac-mini`; the build
does not clone, copy, or link a separate DwarfStar checkout.

The initial implementation was derived from the MIT-licensed `antirez/ds4`
tree at commit `b0309611041655f4e45671cfd9c9886aff161406`, then materialized
after applying this project's fixed DeepSeek V4 Flash 0731 Q4-trunk,
packed-expert, cache-aware decode, exact-prefill, and unified-memory patches.
The upstream license is preserved in `src/runtime/LICENSE`.

Only the Apple Metal runtime and HTTP server files needed by the fixed model
are kept here. Generated objects, binaries, CUDA/ROCm code, tests, agents, and
unrelated model variants are intentionally excluded.
