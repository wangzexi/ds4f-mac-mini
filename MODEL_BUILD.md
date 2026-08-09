# 0731 Mini model build

The Mini model build changes only large resident, non-router tensor families to
Q4_K. Routed MoE tensors are copied byte-for-byte from the verified 0731
IQ2_XXS/Q2_K GGUF. Router, normalization, mHC, compressor, and indexer tensors
keep the template types unless a later measured experiment explicitly changes
them.

## Storage layout

- Windows WSL ext4: official Hugging Face snapshot, quantizer, output, temporary files.
- Windows D drive: immutable backup/template of the verified 81 GB GGUF.
- Mac mini: active runtime model only after the Windows output is validated.

The D-drive template is consumed sequentially for routed tensors. It must not
be used as the Hugging Face source directory because shard and tensor lookup can
be random.

## Reproducibility inputs

- Official repository: `deepseek-ai/DeepSeek-V4-Flash`
- Official snapshot SHA: `60d8d70770c6776ff598c94bb586a859a38244f1`
- Quantizer base: `antirez/ds4` plus `tools/deepseek4-quantize.c`
- Context target: 32768 tokens

## Dry-run contract

For the verified 0731 template, the build must report exactly 346 type changes:

- 1 token embedding: F16 to Q4_K
- 215 main attention projections: 43 layers times 5, Q8_0 to Q4_K
- 129 shared-expert projections: 43 layers times 3, Q8_0 to Q4_K
- 1 output projection: Q8_0 to Q4_K

Expected output size is 82,853,553,024 bytes (77.163 GiB), saving 3.601
GiB from the 86,720,111,488-byte template. The 21 indexer attention
projections remain F16. Any different type-change count or unexpected router,
indexer, normalization, mHC, compressor, or routed-expert change aborts the
build.

## Required validation gates

1. Source and backup GGUF sizes and SHA-256 hashes match their origins.
2. Dry-run reports no routed-expert type changes.
3. Every routed tensor in the output is byte-identical to the template.
4. Q4 output opens and produces finite logits on Windows metadata checks.
5. Mini baseline prompts are compared against the verified model before speed tuning.
6. Runtime memory remains safe at a 32768-token context.
7. Decode and prefill measurements report quality mode and expert-cache settings.
