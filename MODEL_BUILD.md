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

The development-Mac CLI `scripts/resume-template-to-windows.sh` copies the
immutable Mini Q4/IQ2 template to `D:\ds4f-build\template` through an SSH
stream with a MiB-aligned resume point and final SHA-256 check.  It does not
stage model data on the development Mac, never overwrites an existing validated
destination, and refuses to run while Hugging Face download lock files exist.

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

## Experimental Q2 trunk candidates

The Q4 deployment remains the only accepted production model.  Two Q2_K trunk
candidates exist solely to test whether saved static-trunk memory can buy
enough additional expert cache on this fixed M4/16GB machine to compensate for
their quality loss.  Both retain the routed IQ2_XXS/Q2_K payload byte-for-byte;
they therefore share the existing packed-expert sidecar only after its payload
identity has been verified.

- `scripts/build-q2-core-model.sh` changes 258 Q4_K tensors (attention Q/KV
  projections and all shared-expert projections) to Q2_K.  It leaves the 86
  attention-output projections and embedding/output heads at Q4_K, retaining
  the specialized batched prefill kernels.  Expected raw reduction from Q4 is
  655,196,160 bytes (0.610 GiB).
- `scripts/build-q2-full-model.sh` changes all 346 Q4_K tensors to Q2_K for an
  expected raw reduction of 1,579,745,280 bytes (1.471 GiB).  This is a more
  aggressive candidate and can lose both quality and prefill speed.

Build scripts require `HF_DIR`, `TEMPLATE`, and `OUT`; they abort unless their
dry run reports exactly 258 or 346 changes respectively.  After building, run:

```sh
python3 tools/verify_q2_core_model.py --profile core "$TEMPLATE" "$OUT"
python3 tools/verify_routed_copy.py "$TEMPLATE" "$OUT"
```

Use `--profile full` for the full candidate.  The second command is deliberately
a full routed-payload scan, not a header-only comparison.

Before considering a candidate for deployment, test it on Mini with the shared
sidecar override enabled, record a current-Q4 fixed-suite baseline, then compare
the candidate:

```sh
scripts/evaluate-fixed-greedy-suite.py ds4f-q4-speed Q4.gguf \
  --out results/quality/q4-fixed-greedy.tsv --repeat 2

DS4_METAL_STREAMING_EXPERT_PACK_PATH=IQ2Experts-packed.bin \
DS4_METAL_STREAMING_EXPERT_PACK_ALLOW_SAME_EXPERTS=1 \
scripts/evaluate-fixed-greedy-suite.py ds4f-q4-speed Q2-candidate.gguf \
  --reference results/quality/q4-fixed-greedy.tsv \
  --out results/quality/q2-fixed-greedy.tsv --repeat 2
```

The suite is a small deterministic divergence gate, not a claim of broad model
quality.  The candidate must also pass a 32K memory-plan, prefill, and decode
I/O A/B with an expert-cache ceiling above the old fixed 1,800-slot limit.
