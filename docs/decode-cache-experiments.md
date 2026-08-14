# Decode expert-cache experiments

This note records only exact, one-session measurements on the fixed M4/16GB
Mini and the Flash-0731 Q4-trunk/IQ2-expert model.  Cache policy changes do
not alter Router outputs, selected experts, or token traces.

## What the continuous-chat trace shows

The production cache is one global selected-expert pool of 967 entries (6.37
GiB), not a per-layer partition.  A real `你好` -> eight-token answer ->
ten-token continued Prefill -> 30-token answer trajectory recorded 2,279
routed-MoE selections.

At the second, ten-token continued Prefill boundary, the direct residency
snapshot reports:

| cache state | experts |
| --- | ---: |
| resident before Prefill | 942 |
| resident after Prefill | 967 |
| previous residents surviving | 565 |
| previous residents displaced | 377 |
| new residents | 402 |

So the retained Decode cache is real: the suffix replaces 40% of it, but keeps
60% of the previously useful entries.  Clearing the cache is substantially
worse in the matching continuity controls.

The Decode-only churn profile also records zero entries evicted within six
subsequent expert lookups.  The harmful churn is not an expert from layer N
being immediately removed by layer N+1.  It is a longer-window problem:
some single-use loads displace an entry that the next generated tokens need
again.  On this trajectory 1,234 of 2,122 Decode loads were one-shot evictions
and 413 were later read again (2.72 GiB logical reread).

## Rejected cache policies

All rows below emitted the same 32-token trace and response hash.

| trajectory | policy | Decode time / rate | result |
| --- | --- | --- | --- |
| 993-token prompt, 32-token decode | global LRU | 18.827s / 1.70 t/s | baseline |
| same | 55% two-queue probation | 20.138s / 1.59 t/s | slower |
| same | historical route heat | 30.730s / 1.04 t/s | much slower |
| 562-token prompt, 32-token decode | global LRU | 72.913s / 0.4389 t/s | baseline |
| same | one-token probation | 77.339s / 0.4138 t/s | slower; 3.82 vs 2.85 GiB reread |

For a genuinely short continued suffix the runtime keeps a one-token
probation window; the policy switches to plain LRU above 18 uncached Prefill
rows.  This is a fixed-target decision from the measured trajectories, not a
general-purpose tuning knob.

The short-window tenure itself was also swept on the live eight-token answer
followed by the usual ten-token suffix and 30-token response.  Every row had
the same complete second-turn token trace and response SHA-256:

| probation tenure (expert lookups) | second-turn time |
| ---: | ---: |
| 64 | 16.193 s |
| 128 | 16.228 s |
| 192 | 16.472 s |
| 258 (one full 43 x 6 Decode pass; production) | 16.271 s |

The 64--258 difference is below normal SSD variation, while 192 is plainly
slower.  There is no repeatable gain for a smaller window, so the production
one-token value remains fixed and the A/B-only environment override was not
retained.

One zero-memory policy did clear the repeatability bar.  For short continued
Prefills only, the cache protects the prior Decode token's six selected experts
for each layer until that layer receives its next exact Router result; it then
replaces that layer's protected set.  This is a cache-victim restriction only:
the Router, selected IDs, weights, reads for true misses, and Metal kernels are
unchanged.  It deliberately remains off for long Prefills, whose plain-LRU
working set needs the entire pool as possible victims.

On the same eight-token answer -> ten-token suffix -> 30-token response,
two interleaved exact trials measured baseline second turns of 16.226 s and
16.125 s (mean 16.175 s), versus 15.795 s and 15.798 s with the protection
(mean 15.797 s): a 2.3% improvement.  The protected run had 77 fewer misses,
0.51 GiB less logical expert read, and 282 ms less selected-slot bind time.
All token IDs and reply SHA-256 values matched.  This is the fixed short-turn
production policy.

It is intentionally all-layer rather than a hand-picked high-overlap subset.
On an isolated same-session sweep, protecting L0--L42 took 15.966 s; beginning
only at L3 took 16.201 s, and beginning only at L12 took 16.213 s.  Although
the early hash layers have little direct adjacent-token set overlap, excluding
them makes the global cache evolve less favorably.  The complete 43-layer
protection set is therefore retained.

Keeping two historical route generations was also tried in isolation.  It is
still exact and zero-buffer, but can protect as many as 516 entries: the paired
short-session runs were 15.897 s for the existing one-generation policy and
15.945 s for two generations, with identical token IDs and response hashes.
That difference is noise rather than a gain, so the larger victim restriction
is rejected.

Protecting only the largest exact Router weights from the immediately previous
token is also worse.  A test-only policy kept the top 6 (the production
control), 5, or 4 of each layer's six prior selected experts.  The same live
continuation produced the accepted complete token trace and response hash in
all cases, but reducing the protected set increased misses and read volume:

| protected experts per prior layer | second turn | total misses | logical expert read |
| ---: | ---: | ---: | ---: |
| 6 | 16.022 s | 5,046 | 31.23 GiB |
| 5 | 16.089 s | 5,075 | 31.42 GiB |
| 4 | 16.165 s | 5,093 | 31.54 GiB |

The low-weight members still carry useful route continuity at this cache size;
all six remain protected.  The exact-weight ranking code was test-only and is
not retained.

Hard per-layer quotas are also not promising.  Replaying the 2,279-selection
trace with the same 967-entry capacity yields a 70.7% second-Decode hit rate
for both global LRU and a 22/23-entry-per-layer partition.  A six-entry
minimum per layer produces the identical replay result.  Global borrowing is
therefore retained because it costs no extra policy complexity and wins on
longer measured prompts.

## Narrow next experiment: transient small-Prefill admission

The ten-token continued Prefill contains 2,580 selections for 1,282 unique
`(layer, expert)` pairs; 770 pairs occur only once.  An offline replay of the
recorded route, with the existing cache state, gives the following second
Decode result:

| short-Prefill admission | Prefill misses | subsequent Decode misses |
| --- | ---: | ---: |
| current immediate admission | 1,084 | 2,267 |
| never retain new suffix loads | 1,275 | 2,258 |
| retain a new pair on its second suffix selection | 1,100 | 2,210 |

The last row saves 57 later Decode reads (about 0.38 GiB) for only 16 added
Prefill misses.  It is the only remaining cache-policy direction with a
positive trace-level signal.

It is intentionally not enabled yet: implementing it correctly requires an
isolated six-to-seven-expert (about 48 MiB) transient Metal staging tier.  A
miss that is not admitted must still be bound for the current exact operation,
without overwriting the global Decode LRU.  Reusing global slots and restoring
them afterwards would itself evict the retained data and invalidate the
experiment.  The future implementation must preserve the exact token trace
and compare repeated live-session timings against the present baseline.

### Rejected implementation attempt

An initial six-slot implementation was tested on 2026-08-14.  It redirected
first-sighting short-Prefill reads into separate Metal buffers and used the
ordinary global path on a second sighting.  It did reduce the observed suffix
displacement from 377 to 235 entries, but this is not an acceptable result:
its generated continuation diverged from the accepted baseline at output token
two.  Repeating the modified implementation only reproduced the same wrong
trace, which proves neither mathematical nor resource-lifetime correctness.

The implementation was reverted immediately and the production numerical gate
was rerun.  Any future version needs an explicit command-buffer ownership fence
for every transient expert buffer (including the per-layer address binding),
followed by a baseline token/hash comparison.  No performance figure from this
rejected path should be used for decisions.

A follow-up made that ownership explicit without inserting per-layer command
buffer waits: it gave every one of the 43 routed layers its own six-slot
transient group.  That removes the cross-layer overwrite hazard, at a lazily
allocated maximum of 258 slots (about 1.70 GiB).  The live `你好` -> eight-token
answer -> ten-token suffix -> 30-token answer test was numerically exact: all
30 second-turn token IDs and the response SHA-256 matched the accepted
baseline.  It is nevertheless rejected.  The second turn took 17.886 s versus
the current exact path's 15.968--16.204 s; memory footprint rose from 11.69 to
13.39 GiB, and total logical expert reads rose from 31.73 to 32.25 GiB.  Thus
the extra tier creates paging pressure and makes first-sighting loads less
reusable; it does not buy back enough Decode residency to pay for itself.

## Rejected packed-offset read ordering

The packed sidecar puts each complete expert in a contiguous 6.75 MiB region.
An A/B-only queue sort submitted the already-selected missing experts to the
five existing `pread` workers in ascending sidecar offset order. It changed
neither Router IDs, cache admission/eviction, destination buffers, nor the
number of read tasks. Four interleaved live `你好` -> eight-token answer ->
ten-token suffix -> 30-token answer trials all passed the complete token trace
and response SHA-256 gate:

| queue order | second-turn seconds |
| --- | ---: |
| normal | 15.959, 16.128, 15.884, 15.797 (mean 15.942) |
| packed offset | 15.771, 16.002, 15.785, 15.902 (mean 15.865) |

The apparent 77 ms / 0.5% mean difference is inside run-to-run variation: in
the deliberately reversed final pair normal ordering was faster (15.797 s
versus 15.902 s). Matching profiler deltas had the same selected routes and
2,045 expert read tasks. The Mini NVMe's concurrent queue is therefore already
doing the useful scheduling; the sort and its test-only environment switch are
not retained.

## Rejected Decode resident/miss overlap

The normal exact Decode path must read the Router's six selected IDs back to
the CPU, bind their cached Metal buffers, and synchronously load any misses.
On a 562-token prompt followed by 32 generated tokens, that path left the GPU
busy for only about 3.96 seconds of the 16.12-second Decode phase.  This made
an exact overlap experiment worthwhile: submit the already-resident selected
experts first, read the missing experts while that command buffer runs, then
submit the missing subset and the single final down-projection.

The experiment used ordinary selected-slot `MTLBuffer` bindings rather than
the raw GPU address table.  The latter has an independently confirmed
cross-request resource-lifetime regression and remains disabled.  The direct
slot version was run only from an isolated worktree; it did not modify the
production binary.

Every trial below produced the accepted 32-token trace
`17839,4597,3500,...,21458,223` and response SHA-256
`4188a0ba1fb8b823fe8512ca0527c6800ae3d41ae4e6d6ffea4f362564b24fb9`.

| 562-token prompt + 32 Decode tokens, five `pread` workers | request time | result |
| --- | ---: | --- |
| production, repeat 1 | 71.307 s | exact baseline |
| production, repeat 2 | 71.082 s | exact baseline |
| direct overlap, split at one miss, repeat 1 | 71.040 s | exact |
| direct overlap, split at one miss, repeat 2 | 71.240 s | exact |

The means are 71.195 s and 71.140 s respectively: a 55 ms (0.08%) difference,
well inside run-to-run variation.  A first sweep also measured thresholds 1,
3, and 4 at 70.927 s, 70.958 s, and 71.268 s; it confirms that a threshold of
four loses useful overlap, but cannot establish a meaningful win for one or
three.  The short live continuation is worse: `你好` -> 8-token answer ->
10-token suffix -> 30-token answer took 16.284 s in production and 16.919 s
with direct overlap, despite both complete token traces and response hashes
matching.

The direct version did execute its intended work: 777 of 1,376 long Decode
layers split into resident and missing stages.  But its second Metal command
submission and the required ownership fence consume the same budget that the
SSD read is meant to hide.  It is therefore rejected and remains out of the
runtime.  The next viable Decode direction is not another split threshold; it
is reducing the selected-slot loading and binding boundary itself without
reintroducing the unsafe GPU-address-table lifetime behaviour.  The production
Flash path already uses its exact CPU Router before this boundary, so it has no
GPU Router-ID readback to remove.
