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
- the complete Q4 trunk, token embedding, and output head are explicitly read
  once into permanent owned buffers because every decode token consumes them;
- the request planner treats those bytes as resident and assigns only the
  exact remaining budget to routed-expert slots.

Each logical read emits one `ds4-io` record:

```text
ds4-io: seq=24 kind=decode_layer layer=22 spans=2 \
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
