# Explicit model I/O

The fixed Mac mini production path owns every model-payload read.

When `--metal --ssd-streaming` is active:

- the model file descriptor is duplicated with `F_NOCACHE`;
- GGUF metadata remains mapped, but the 82.85 GB tensor payload mapping is
  removed after startup;
- requested tensor spans are read with `pread` into owned shared Metal
  buffers;
- Metal kernels can bind only ranges covered by those buffers;
- CPU tensor access is resolved through the same buffers;
- an unplanned CPU or GPU payload access is logged as `BLOCKED` and fails;
- startup retains only the token embedding; prefill replaces it with one
  transformer layer at a time and releases that layer after execution;
- layer-major prefill keeps the selected expert working set and its larger
  workspace in the memory otherwise occupied by the complete trunk;
- the canonical fixed-target prefill path changes only the I/O nesting: each
  transformer layer is read once, while prompt rows execute sequentially with
  the accepted single-token decode arithmetic inside that layer;
- future trunk spans and planned packed experts are read into owned staging
  buffers; activation transfers those buffers into the live view without a
  second payload read, and cancellation is checked at each 32 MiB boundary;
- prefill maps only the output head for its final logits, then the server
  releases prefill workspace and shrinks the expert cache;
- the first decode token explicitly reads the complete Q4 trunk, token
  embedding, and output head into owned buffers and retains them for later
  decode tokens;
- the request planner prices the small streamed prefill residency and the
  complete decode residency separately.

The canonical prefill path was checked against the frozen mmap/token-major
baseline over all 129,280 output logits. The dumps are byte-identical
(`max_abs = 0`, identical greedy token), so the phase-specific residency does
not introduce a numerical approximation.

The prompt-length policy, paging experiments, and persistent per-layer
measurement format are documented in `docs/prefill-planner.md`.

Each logical read emits one `ds4-io` record:

```text
ds4-io: seq=24 kind=prefill_trunk layer=22 spans=2 \
useful=116314072 io=116359168 read_ms=54.6 rate=2031.1_MiB_s buffers=2
```

`useful` is the exact tensor byte count. `io` includes hardware-page alignment.
`read_ms` is blocking explicit-I/O time and `rate` is the resulting throughput.
Expert-cache misses use `kind=expert` and identify their transformer layer,
bytes read, read latency, and throughput.
Because the payload is no longer mapped, model-weight VM faults cannot hide
inside a Metal command. Remaining latency is attributable to an explicit read,
expert-cache work, or GPU execution.

The frozen mmap implementation is available in the sibling worktree
`/Users/zexi/workspace/ds4f-mini-mmap-baseline` on branch `mmap-baseline`.
