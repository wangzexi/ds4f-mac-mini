# Prefill / Decode boundary

This engine is dedicated to DeepSeek-V4-Flash-0731 on the 16 GiB M4 Mini.
Prefill and Decode are separate execution pipelines.  A cache policy must not
silently cross this boundary.

## Prefill pipeline

Prefill receives a known batch of prompt tokens.  It is layer-major:

1. stream the current layer's trunk and complete routed-expert layer payload;
2. run that layer for the prompt batch in parallel;
3. hand off only useful tail state, then release the layer's temporary expert
   staging before moving to the next layer.

There is no per-expert Decode LRU in this pipeline.  Its goal is sequential
SSD throughput and batch GPU occupancy.

## Decode pipeline

Decode receives one autoregressive token.  Each learned layer selects six
routed experts only after its Router has run.  Its future implementation owns
a dedicated per-layer cache and layer-local LRU/heat policy.  Its goal is to
avoid selected-expert SSD misses and to overlap reads with computation.

## Allowed sharing

The pipelines may share only mechanical primitives:

- packed-expert file layout and `pread` worker pool;
- Metal buffer/slab allocation and lifetime-safe release helpers;
- quantized tensor decoding and numerical kernels;
- route heat persisted with a KV prefix as prediction data.

They must not share eviction, cache-capacity, prefetch, or release policy.
Any new Decode cache API must have a Decode-specific entry point; it must not
be enabled through a generic environment switch inside the Prefill cache path.
