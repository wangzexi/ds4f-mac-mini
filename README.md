# ds4f-mini

`ds4f-mini` is a single-purpose DeepSeek V4 Flash 0731 inference runtime for
an Apple M4 Mac mini with 16GB unified memory. It intentionally supports one
model layout, one 32K context target, and the Metal SSD-streaming path only.

The runtime source is vendored in `runtime/`; building and running do not
clone, patch, copy, or link a separate DwarfStar checkout. The local
`reference-*` directories are research material and are ignored by Git.

## Current model

The deployed model is:

```text
models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf
models/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin
```

Its routed experts are unchanged from the verified DwarfStar template:

- gate/up experts: IQ2_XXS
- down experts: Q2_K
- router: F16
- normalization: F32

The Mini variant changes 346 non-routed tensors to Q4_K: token embedding,
five attention projections per layer, three shared-expert projections per
layer, and the output projection. This saves 3.601GiB on disk. It is lossy:
the Q4 model has a reproducible greedy-output divergence from the Q8 template,
including the `深度求趣公司` hallucination for a fresh `你好` chat request.
Do not describe the Q4 trunk as numerically equivalent to Q8.

See [MODEL_BUILD.md](MODEL_BUILD.md) for the exact tensor contract and
[docs/32k-q4-runtime.md](docs/32k-q4-runtime.md) for measured memory, quality,
prefill, and decode results.

## Build

```sh
make
```

This builds only the two supported executables:

- `ds4f-q4-speed`: fixed 32K greedy runner used for numerical and performance checks
- `ds4f-server`: OpenAI-, Responses-, and Anthropic-compatible HTTP server

The old self-written prototype graph was intentionally removed after its
Metal path was shown to diverge numerically. Historical experiments remain in
Git history and under `results/`; they are not deployment entry points.

## Run

Start the HTTP server:

```sh
scripts/run-server.sh
```

Run a one-shot exact prompt:

```sh
scripts/run-32k.sh exact '你好' 32
```

`balanced` and `turbo` are explicit approximate-decode experiments. Prefill
stays exact, but routed-expert mass is dropped during decode. They are not the
default quality baseline.

Run the deterministic token regression:

```sh
make check-production \
  MODEL=models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf
```

The server API and terminal-client examples are documented in
[docs/server.md](docs/server.md).

## Repository layout

```text
runtime/   bundled Metal runtime and HTTP server
src/       fixed one-shot runner adapter
scripts/   production launch, model build, and regression scripts
tools/     GGUF inventory, quantization, packing, and verification tools
docs/      current runtime and server documentation
results/   historical benchmark and quality evidence
models/    local weights; ignored by Git
```

The supported path deliberately excludes Ollama, DSpark/MTP, CUDA/ROCm,
multi-model compatibility, and general hardware auto-tuning.
