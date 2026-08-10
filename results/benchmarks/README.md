# Long-file prefill benchmark

Hardware: Apple M4 Mac mini, 16GB unified memory. Runtime uses the fixed 32K Q4-trunk/IQ2-expert model, exact prefill, 600 expert cache slots, and the packed expert sidecar.

The prompt files are deterministic slices/repetitions of the repository README. Token counts include the chat wrapper; the 15K run reported 14,735 prompt tokens internally and used 4,096-token chunks.

| Prompt file | Token count | Prefill speed |
|---|---:|---:|
| `long-prompt-1k.txt` | 1,001 | 10.63 t/s |
| `long-prompt-4k.txt` | 3,977 | 30.19 t/s |
| `long-prompt-15k.txt` | 14,735 runtime prompt tokens | 27.36 t/s |

The 4K case fits one prefill chunk. The 15K case needs four chunks and does more attention/KV work, so throughput does not continue rising. At 27.36 t/s, a full 32K prefill would take about 19.5 minutes; that is an extrapolation, not a new full-32K measurement.

## 8K chunk experiment

The original memory report was incomplete: its 0.25/0.50GiB "buffers" values counted only attention score/mask storage and omitted most batch activations. The startup estimator now reports the complete graph workspace.

Without workspace aliasing, `8192` chunk + 256 expert slots failed to lock the cache: macOS locked only 0.09GiB and the runtime reduced 256 requested slots to 12. See `prefill-15k-chunk8k-cache256-exact.log`.

`DS4_METAL_PREFILL_STAGE_ALIAS=1` reuses attention-only buffers for FFN-only activations after attention completes. It saves about 337,968 bytes per token: 1.29GiB at 4096 and 2.58GiB at 8192. With aliasing, the 8192/256 configuration completed the full 14,735-token exact run:

| Configuration | Corrected planned memory | Prefill | First token |
|---|---:|---:|---|
| 4096 chunk, old baseline, 600 slots | historical report incomplete | 27.36 t/s | `I` |
| 8192 chunk, 256 slots, no alias | cannot lock; falls to 12 slots | not completed | — |
| 8192 chunk, 256 slots, stage alias | 8.72GiB | **41.55 t/s** | `I` |

The corrected 8.72GiB plan consists of 0.92GiB KV, 5.84GiB complete graph buffers, 0.28GiB resident model spans, and 1.69GiB expert cache. No `mlock` degradation occurred. The speedup over the old long-prompt baseline is about 51.9%. A 3,977-token one-chunk alias run measured 38.80 t/s, so doubling the chunk beyond 4096 adds only about 7% over the already-optimized one-chunk path; the 8192 mode remains opt-in because its memory margin is much smaller.

Exact 1,001-token regression with 4096 chunk produced `你好` both with and without alias (24.83 vs 24.97 t/s). This checks that the buffer views do not change that greedy result.

KV-to-SSD is not the next optimization target. Even the 8192 plan has only 0.92GiB of KV, so offloading it recovers less than 1GiB while adding writes, reads, and synchronization to every layer. Batch workspace reuse and a 256-slot current-layer expert cache provide substantially more useful memory leverage.

Run a file directly by prefixing its path with `@`:

```sh
scripts/run-32k.sh exact @results/benchmarks/long-prompt-4k.txt 1
```

Run the validated 8192 experiment with:

```sh
env \
  DS4F_FAST_CACHE_EXPERTS=256 \
  DS4_METAL_PREFILL_STAGE_ALIAS=1 \
  DS4_METAL_PREFILL_CHUNK=8192 \
  scripts/run-32k.sh exact @results/benchmarks/long-prompt-15k.txt 1
```
