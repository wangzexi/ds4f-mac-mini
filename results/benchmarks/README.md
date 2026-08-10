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
| automatic 8192/256 + stage alias | runtime-selected after startup guard | **41.61 t/s** | `I` |

The corrected 8.72GiB plan consists of 0.92GiB KV, 5.84GiB complete graph buffers, 0.28GiB resident model spans, and 1.69GiB expert cache. No `mlock` degradation occurred. The speedup over the old long-prompt baseline is about 52.1%. A 3,977-token one-chunk alias run measured 38.80 t/s, so doubling the chunk beyond 4096 adds only about 7% over the already-optimized one-chunk path.

The fixed 32K runner now enables stage aliasing automatically. On the first request of a cold engine, prompts above 4,096 tokens automatically select an 8,192-token workspace and cap the cache to one full routed layer (256 experts). Warm reusable engines retain their existing cache instead of discarding useful expert pages. `DS4_METAL_DISABLE_PREFILL_AUTO_MEMORY=1` disables this policy; explicit prefill/cache environment variables remain available for diagnostics. The startup memory line is a conservative guard computed before the prompt is known; the later `prefill runtime plan` line reports the actual prompt-sized workspace and expert path.

Exact 1,001-token regression with 4096 chunk produced `你好` both with and without alias (24.83 vs 24.97 t/s). This checks that the buffer views do not change that greedy result.

KV-to-SSD is not the next optimization target. Even the 8192 plan has only 0.92GiB of KV, so offloading it recovers less than 1GiB while adding writes, reads, and synchronization to every layer. Batch workspace reuse and a 256-slot current-layer expert cache provide substantially more useful memory leverage.

## Rejected pipeline experiments

The production path keeps the original exact full-layer batch kernel above 760 prompt tokens. Raising the packed selected-address path to 8K changed the 992-token greedy output from `你好` to `#`, so it is not a valid optimization even though throughput was similar. The prototype and its asynchronous loader were removed from the build chain.

Additional adjacent A/B tests did not justify more scheduling complexity:

- two-layer preparation lookahead reduced the 992-token exact run from 11.22 to 9.95 t/s;
- asynchronous selected-expert staging measured 2.73 t/s versus 2.87 t/s synchronously on the same short prompt;
- after removing that prototype, the 992-token exact regression again produced `你好`, and the automatic 14,735-token run produced the historical first token `I`.

Run a file directly by prefixing its path with `@`:

```sh
scripts/run-32k.sh exact @results/benchmarks/long-prompt-4k.txt 1
```

The normal command now selects the validated long-prompt plan automatically:

```sh
scripts/run-32k.sh exact @results/benchmarks/long-prompt-15k.txt 1
```

The equivalent explicit diagnostic command is:

```sh
env \
  DS4F_FAST_CACHE_EXPERTS=256 \
  DS4_METAL_PREFILL_STAGE_ALIAS=1 \
  DS4_METAL_PREFILL_CHUNK=8192 \
  scripts/run-32k.sh exact @results/benchmarks/long-prompt-15k.txt 1
```
