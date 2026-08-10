# Long-file prefill benchmark

Hardware: Apple M4 Mac mini, 16GB unified memory. Runtime uses the fixed 32K Q4-trunk/IQ2-expert model, exact prefill, 600 expert cache slots, and the packed expert sidecar.

The prompt files are deterministic slices/repetitions of the repository README. Token counts include the chat wrapper; the 15K run reported 14,735 prompt tokens internally and used 4,096-token chunks.

| Prompt file | Token count | Prefill speed |
|---|---:|---:|
| `long-prompt-1k.txt` | 1,001 | 10.63 t/s |
| `long-prompt-4k.txt` | 3,977 | 30.19 t/s |
| `long-prompt-15k.txt` | 14,735 runtime prompt tokens | 27.36 t/s |

The 4K case fits one prefill chunk. The 15K case needs four chunks and does more attention/KV work, so throughput does not continue rising. At 27.36 t/s, a full 32K prefill would take about 19.5 minutes; that is an extrapolation, not a new full-32K measurement.

Run a file directly by prefixing its path with `@`:

```sh
scripts/run-32k.sh exact @results/benchmarks/long-prompt-4k.txt 1
```
