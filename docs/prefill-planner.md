# Fixed-Mini prefill planner

This planner is intentionally specialized for the 16 GiB M4 Mini and the
fixed DeepSeek V4 Flash 0731 Q4-trunk/IQ2-expert artifact.

The first three MoE layers use the frozen `ffn_gate_tid2eid` lookup. The
server preloads all 256 layer-0 experts before it listens. Once a request has
been tokenized, Prefill computes the exact expert unions for hash layers 1 and
2. Layer 0 starts from the startup-resident set; layer 1 is read while layer 0
computes, and layer 2 is read while layer 1 computes. Only later learned-router
layers use speculative hotlist Prefill.

## Request plan

The exact layer-major path keeps decode arithmetic unchanged and varies only
the I/O schedule:

| Prompt rows | Learned-router staging per future layer | Lookahead |
| ---: | ---: | ---: |
| 1-63 | demand-loaded only | 3 trunk layers |
| 64-127 | 128 hot experts | 3 layers |
| 128-255 | 192 hot experts | 3 layers |
| 256+ | all 256 experts | 1 layer |

For layers 3 and later, partial expert sets come from the fixed Flash hotlist,
then are sorted by expert ID before `pread` so the packed sidecar is consumed
in forward file order. Hash layer 0 is instead fully resident before the
server listens; hash layers 1 and 2 use their exact token-id unions. A
completed layer is released immediately. Staged entries reserve the
same exact slot budget as active cache entries, so prefetch cannot silently
exceed the request planner's expert allocation.

Complete learned-layer 256-expert staging is intentionally one layer ahead.
Holding three complete 1.69 GiB Metal expert layers produced different
long-prompt logits on this machine; one layer ahead is byte-identical to demand
loading and still reduced the measured 300-row run from 60.15 s to 40.11 s.

## Explicit I/O and paging

- Model and packed-expert descriptors use `F_NOCACHE`.
- Trunk and expert staging own their Metal buffers; activation never rereads a
  prefetched payload.
- Reads are cancellable between 32 MiB chunks. The tested 2/8/16/32/64 MiB
  times for the 100-row fixture were 26.26/26.10/26.06/25.95/26.00 s.
- The exact row loop also polls the request cancellation callback. A client
  disconnect stops the current layer, cancels every queued read, releases its
  staging buffers, and leaves the server ready for the next request.
- `F_RDADVISE` is not used by the staged path. Combining it with explicit
  multi-layer reads regressed the same fixture from 25.55 s to 39.92 s.
- More lookahead is not automatically better: one/two/three/four-layer partial
  windows measured 26.93/26.27/25.55/26.95 s before chunk tuning.

## Measurement table

The server appends activation measurements to
`cache/prefill-measurements.tsv` when
`DS4_METAL_PREFILL_MEASUREMENTS_PATH` is set. Columns are:

```text
unix_time  pid  kind  layer  units  bytes  read_ms  activation_wait_ms
```

`kind=trunk` uses tensor-span count as `units`; `kind=experts` uses staged
expert count. The append is tiny and occurs after activation, outside the
weight-read critical path. Run `scripts/summarize-prefill-measurements.py` to
aggregate the table by kind and layer.

Set `DS4_METAL_PREFILL_LAYER_PROFILE=1` to log `prepare_ms`, `compute_ms`, and
`total_ms` for every exact layer-major Prefill layer. `prepare_ms` includes
activation waits for the current layer; `compute_ms` measures command encoding
and completion while future-layer reads may run in the background.

## Numerical and performance gates

- 11-row canonical fixture: all 129,280 logits byte-identical to the frozen
  mmap/token-major baseline.
- 100-row fixture, 128 experts x three layers: byte-identical to demand loading;
  33.35 s to 25.95 s after tuning (about 22%).
- 200-row fixture, 192 experts x three layers: byte-identical to demand loading;
  50.20 s to 33.29 s (about 34%).
- 300-row fixture, 256 experts x one layer: byte-identical to demand loading;
  60.15 s to 40.11 s (about 33%).

These figures are regression anchors for this exact machine, not portable
performance claims.
