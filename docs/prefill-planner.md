# Fixed-Mini prefill planner

This planner is intentionally specialized for the 16 GiB M4 Mini and the
fixed DeepSeek V4 Flash 0731 Q4-trunk/IQ2-expert artifact.

The first three MoE layers use the frozen `ffn_gate_tid2eid` lookup. The
server preloads all 256 layer-0 experts before it listens. Once a request has
been tokenized, Prefill computes the exact expert unions for hash layers 1 and
2. Layer 0 starts from the startup-resident set; layer 1 is read while layer 0
computes, and layer 2 is read while layer 1 computes. Later learned-router
layers never use a corpus-global fixed expert order.

## Request plan

The exact layer-major path keeps decode arithmetic unchanged and varies only
the I/O schedule:

This applies equally to a cold prompt and to the uncached suffix after a
disk-KV hit.  The older token-major suffix path is disabled for this fixed
Flash runtime: it bypassed the SSD pipeline and could retain experts from many
layers at once.

| Prompt rows | Cold learned-router staging | KV-prefix heat staging |
| ---: | ---: | ---: |
| 1-15 | wait for Router, then exact-read | 12 candidates |
| 16-63 | wait for Router, then exact-read | 24 candidates |
| 64-255 | wait for Router, then exact-read | 64 candidates |
| 256+ | all 256 experts | all 256 experts |

For layers 3 and later, a cold prompt keeps the conservative all-or-nothing
policy: short and medium prompts wait for the actual Router result, while long
prompts read all experts in packed-file order. A KV-prefix hit additionally
restores decayed Router-weight heat. Below the 256-token full-layer crossover,
that heat selects 12/24/64 candidates. They are sorted by heat, then each
eight-expert priority group is put in packed-file order for SSD locality.

The candidate reader publishes only fully-read expert slots. When the first
real Router result for that layer arrives, it stops at the next slot boundary,
copies completed selected slots into ordinary cache ownership, releases the
speculative slab, and reads only the selected misses. Thus an unneeded partial
slab never becomes a long-lived 1.69 GiB allocation. Hash layer 0 is fully
resident before the server listens; hash layers 1 and 2 use exact token-ID
unions. A completed layer is released immediately.

Complete learned-layer 256-expert staging is intentionally one layer ahead.
Holding three complete 1.69 GiB Metal expert layers produced different
long-prompt logits on this machine; one layer ahead is byte-identical to demand
loading and still reduced the measured 300-row run from 60.15 s to 40.11 s.

## Explicit I/O and paging

- Model and packed-expert descriptors use `F_NOCACHE`.
- Trunk and expert staging own their Metal buffers; activation never rereads a
  prefetched payload.
- Candidate reads are cancellable at an expert boundary (and the underlying
  expert read remains cancellable between 32 MiB chunks). The tested
  2/8/16/32/64 MiB times for the 100-row fixture were
  26.26/26.10/26.06/25.95/26.00 s.
- The exact row loop also polls the request cancellation callback. A client
  disconnect stops the current layer, cancels every queued read, releases its
  staging buffers, and leaves the server ready for the next request.
- `F_RDADVISE` is not used by the staged path. Combining it with explicit
  multi-layer reads regressed the same fixture from 25.55 s to 39.92 s.
- Full-layer lookahead stays at one layer so reads do not compete for the
  Mini's single SSD queue.

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
- The retired fixed Flash hotlist covered only 28 of 74 experts on the
  101-token L3 fixture while reading 100 unused experts; it is no longer part
  of Prefill or Decode cache decisions.
- On the same 302-token prompt, exact demand reads measured 5.27 t/s and full
  one-layer lookahead measured 7.63 t/s, setting the default crossover at 256.
- 300-row fixture, 256 experts x one layer: byte-identical to demand loading;
  60.15 s to 40.11 s (about 33%).

These figures are regression anchors for this exact machine, not portable
performance claims.
