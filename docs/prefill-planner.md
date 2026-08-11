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
| 1-255 | wait for Router, then exact-read | disabled by default (measured faster) |
| 256+ | all 256 experts | all 256 experts |

For layers 3 and later, a cold prompt keeps the conservative all-or-nothing
policy: short and medium prompts wait for the actual Router result, while long
prompts read all experts in packed-file order. A KV-prefix hit additionally
restores decayed Router-weight heat. Below the 256-token full-layer crossover,
candidate prediction is disabled by default below the full-layer crossover.
It remains available through
`DS4_METAL_PREFILL_HOT_EXPERT_PREFETCH_COUNT=0..256` for reproducible Mini
A/B measurements. When enabled, candidates are sorted by heat, then each
eight-expert priority group is put in packed-file order for SSD locality.

The candidate reader publishes only fully-read expert slots. When the first
real Router result for that layer arrives, it stops at the next slot boundary,
copies completed selected slots into ordinary cache ownership, releases the
speculative slab, and reads only the selected misses. Thus an unneeded partial
slab never becomes a long-lived 1.69 GiB allocation. Hash layer 0 is fully
resident before the server listens; hash layers 1 and 2 use exact token-ID
unions. A completed layer is released immediately.

An experimental bridge, `DS4_METAL_PREFILL_TAIL_EXPERT_HANDOFF=1`, changes
only that final release: it retains the six most recently used expert entries
from each completed layer. They are already-resident prompt-tail entries, so
the handoff neither reads another byte nor changes Router/MoE arithmetic; it
only gives the first decode token a chance to hit them. On `你好`, exact 32-token
generation measured 1.96 versus 2.01 t/s and 56.4% versus 57.4% cache hit
rate. That small difference is within run-to-run variance, so it stays opt-in
rather than becoming a production default. The trace IDs were identical.

The default 1–255-token cutoff is measured rather than guessed. With the same
24-token disk-KV prefix, demand-only versus heat candidates were: 10-token
suffix 6.84 s versus 7.36 s (12) / 7.55 s (24); 32-token suffix 13.18 s
versus 14.09 s (24); 85-token suffix 23.97 s versus 25.01 s (64). Candidate
copying does not repay speculative reads on this Mini below the already-tested
256-token full-layer crossover.

Complete learned-layer 256-expert staging is intentionally one layer ahead.
Holding three complete 1.69 GiB Metal expert layers produced different
long-prompt logits on this machine; one layer ahead is byte-identical to demand
loading and still reduced the measured 300-row run from 60.15 s to 40.11 s.

For the 256-expert long-prompt case only, the packed layer is now read through
the bounded (nine-worker) `pread` pool. It is safe to parallelize because the
whole layer is demanded—there is no candidate cancellation to preserve. A
300-token server request went from 47.72 s to 40.24 s: summed expert-activation
wait fell from 12.27 s to 7.70 s (42 activations). The runner's serial versus
parallel 300-token prefill logits were byte-identical, and its throughput rose
from 7.99 to 8.46 t/s. A same-fixture queue-depth sweep measured 8.24 t/s at
4 workers, 8.46 at the default 9, and 7.80 at 18, so the pool deliberately
stays at 9. This is enabled by the production launcher; set
`DS4F_SERVER_PREFILL_FULL_LAYER_PARALLEL_PREAD=0` to reproduce the serial path.
Shorter candidate reads deliberately remain slot-cancellable and serial.

The server deliberately retains a 2,048-row idle scheduling quantum even
though a test workspace can hold 8,192 rows.  On the current service path,
the first 2,048 rows of a 14.75K cold prompt completed at 10.83 token/s.  A
separate 3,982-row one-shot HTTP request, with the same 8K workspace and the
same exact layer-major path, took 368.35 s (10.81 token/s).  In both runs SSD
expert activation waits stayed at roughly 0.002–0.008 ms.  The current exact
per-row kernel therefore scales without a meaningful batching gain past 2K;
increasing the outer quantum is not a service Prefill speedup.  The
`--idle-prefill-quantum` switch exists only to reproduce this calibration; its
production default remains 2,048.

## Decode-trunk handoff experiment

After Prefill, the first decode transition normally maps 90 static trunk spans
(4.60 GiB of I/O).  On this Mini that synchronous map takes about 2.32--2.64 s.
`DS4_METAL_PREFILL_STATIC_DECODE_PREFETCH=1` starts that exact same explicit
read while the final Prefill layer computes, then transfers the completed
buffers to the normal decode-static map.  It does not change any tensor,
Router decision, or logits.  The feature is off by default and is limited to
at most 1,024 Prefill rows; the limit can be adjusted only for an explicit
experiment with `DS4_METAL_PREFILL_STATIC_DECODE_PREFETCH_MAX_TOKENS`.

On the 994-token server request, the 4.60 GiB read took 2.236 s but only
8.5 ms remained exposed at the handoff; the direct runner produced the same
first two greedy IDs as the unmodified path.  On a shorter roughly 250-token
fixture, 1.420 s remained exposed: Prefill fell from 4.87 to 4.67 t/s, while
the following two-token generation rose from 0.61 to 2.11 t/s.  Thus it can
reduce total post-request latency for a long prompt, but it deliberately moves
some work before the first streamed token.  It remains an opt-in long-prompt
latency experiment rather than a default chat-server behavior.

## Rejected scheduler experiments

The project retains a true-batch Prefill implementation for diagnostic work;
set `DS4_METAL_EXACT_PREFILL_ROWS=0` only when explicitly measuring it.  On a
304-token Mini fixture it reached 14.78 t/s versus 8.16 t/s for the canonical
row path, but all 129,280 output logits differed (RMSE 0.426, maximum absolute
difference 2.27).  The argmax happened to agree on that one prompt, which is
not a correctness criterion.  It is not a user-facing turbo mode.

An additional exact-row experiment deferred a full expert-layer activation
while every row ran Attention and Router, preserving the row intermediates for
a later MoE suffix.  Although its first deferred-layer output could be made
byte-identical, later layers diverged and the 256-token fixture improved only
from 6.86 to 6.93 t/s.  The experiment was reverted rather than leave an
unverified alternate schedule in the runtime.  The remaining safe Prefill
opportunity is therefore a future kernel-level split that preserves every
dependency while avoiding the extra per-row mailbox copies; it must clear the
same byte-identical logits gate before becoming selectable.

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
