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

`DS4_METAL_PREFILL_CHUNK=8192` raised the explicit plan from 5.09GiB to 5.66GiB: raw/compressed KV became 0.92GiB and graph buffers 0.50GiB, while the requested expert cache remained 3.96GiB. A 3,977-token prompt still fits one chunk and measured 30.41 t/s, effectively unchanged from 30.19 t/s.

The 14,735-token run was rejected as a production configuration. Once it began filling the expert cache, macOS could lock only 0.10GiB; the runtime reduced 600 requested expert slots to 13. At that point system wired memory was about 9.3GiB against a `vm.global_user_wire_limit` of 13.06GB even though `memory_pressure` reported 35% ordinary memory free. This demonstrates that free RAM is not interchangeable with Metal/wired/`mlock` budget.

A useful future design would make memory phase-specific: use the larger workspace and a transient expert set during prefill, release that workspace, then lock the 600-slot expert cache for decode. The current graph keeps both allocations alive, so 4,096 remains the safe default.

Run a file directly by prefixing its path with `@`:

```sh
scripts/run-32k.sh exact @results/benchmarks/long-prompt-4k.txt 1
```
