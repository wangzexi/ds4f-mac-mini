# 32K Q4 runtime

目标硬件固定为 Apple M4 Mac mini 16GB，模型固定为 DeepSeek V4 Flash 0731。这里不承诺兼容其他模型。

## 模型布局

`DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf` 大小为 82,853,553,024 bytes，SHA-256：

```
e62ac715fd77449a655d2319b8c74b0cc0d11dd81067cccb640f4d89a689aa3d
```

- 129 个 routed expert tensor 原样保留 IQ2/Q2 payload。
- token embedding、215 个 attention projection、129 个 shared-expert projection 和 output 共 346 个 tensor 改为 Q4_K。
- router、norm、compressor 和 indexer 保持模板精度。

100 题官方 score 对比 Q8 模板：平均 NLL 由 0.402654590 变为 0.425920109（+5.78%）；top-1 agreement 由 85.992% 变为 85.344%。这是 16GB 目标下明确接受的 trunk 量化损失，不把它描述为无损。

## 32K 内存与速度

早期日志把 `--ssd-streaming-cache-experts 600` 的计划内存打印为 5.09GiB，其中 graph buffer 只有 0.25GiB。该数字只计算了 attention score/mask 等少数矩阵，漏掉了 Q/K/V、attention 输出、shared FFN 和 routed FFN 的 batch tensor。现在启动预算改为使用完整 graph workspace 估算；旧日志中的 5.09GiB 只能作为历史数据，不能再用于决定 batch 上限。

同一短中文提示、32 token 输出的 exact 路径在 8/16/24/32K 上分别为 2.26/2.24/2.25/2.26 token/s，因此把上限从 32K 降到 8K 对短 decode 没有实际收益。

cache-aware router 的初步速度档位：

| retained router mass | generation | 观察 |
|---:|---:|---|
| 100% | 约 2.30 t/s | exact top-6；默认可靠档 |
| 70% | 3.59 t/s | exact prefill + approximate decode；balanced |
| 30% | 5.06–5.42 t/s | exact prefill + aggressive decode；turbo，仅限短答案 |

因此“达到 5 token/s”已经在 turbo 近似模式、不同提示上实测达到；通用 exact 档尚未达到 5 token/s。质量和速度必须分开报告。

近似只在逐 token decode 生效，prompt prefill 始终 exact。这比从 prompt 开始就裁专家稳定得多：完整 100 题中，70% decode-only 相对 exact 的 NLL 增加 17.72%，top-1 从 85.34% 变为 83.35%，首 token 命中同为 51/100，平均 greedy 前缀从 4.38 变为 4.26；同样 70% 若连 prefill 一起近似，19 题筛选的 NLL 增加 30.83%。30% turbo 的 13 题筛选则让 NLL 增加 138.36%、top-1 从 85.26% 变为 72.12%，因此它即使在若干短决策和算术提示上回答正确，也只能作为显式选择的极速档。

`DS4F_SPEED_CACHE_AWARE_MAX_ENTROPY_PCT` 是实验性的 router 熵保护。80/95% 阈值会让大量层回到 exact，实测只剩 2.17/2.86 t/s；25% mass + 99% 阈值虽有 5.18 t/s，但 10 题 NLL 相比无保护 30% 只改善约 2%，top-1 和 greedy 前缀反而更差。因此三个正式模式把该值设为 100（禁用保护）；代码保留用于后续研究，不把它宣传成现有优化。

设置 `DS4F_SPEED_CACHE_AWARE_PROFILE=1` 会逐层记录 router 的总 mass、实际保留 mass、命中缓存数和保留专家数。它只用于寻找更可靠的自适应 decode 阈值，不改变默认 exact 路径。

Decode 的 cache-hit/miss split 也以 950 槽 exact 路径重新测过。默认只在至少
3 个专家未命中时先算 resident 部分、并行读取 miss；把阈值降到 2 或 1 虽保持
完整 greedy trace 一致，但 `你好` 32-token 分别为 2.11 / 2.13 t/s，默认为
2.15 t/s。额外 command stage 没有换来足够的 I/O 隐藏，因此仍用 3；
`DS4_METAL_STREAMING_EXPERT_SPLIT_MIN_MISSES=1..6` 仅保留为复现实验的开关。

2026-08-11 在当前 600-slot Mini 部署上，以 `你好` 生成 32 token 的新对照为：exact 为 1.95 token/s，30% mass turbo 为 4.49 token/s，20% mass 为 4.55 token/s。20% 的收益不足 2%，却在第一个生成 token 即与 exact 分叉，不能作为可用档位。当前瓶颈不是再少保留一个专家，而是让保留集合更常命中真正需要的专家。

相同提示在服务实际 decode 档位附近的 950 槽重测，30% turbo 为 4.67 t/s；
缓存变大带来小幅收益，但仍未稳定达到 5 t/s，且第一生成 token 即与 exact
分叉。它继续只作为显式极速实验，不进入服务默认配置。

“只放弃未命中且很小的路由权重”也在 950 槽、`你好` 32-token exact
基线上单独扫过：0/5/10/15/20/30% 相对层内最大权重的 generation 分别为
2.04/2.06/2.08/2.08/2.17/2.34 t/s。5% 与 10% 保持完整 greedy trace，
但收益处于测量噪声范围；15% 与 20% 都在第 6 个 token 分叉，30% 在第 3
个 token 分叉。因此 `DS4F_SPEED_CACHE_AWARE_DROP_MISS_BELOW_TOP_PCT` 继续
仅作诊断开关，不形成新的 balanced 档位。瓶颈仍是提高所保留专家的真实命中，
不是在 miss 时再少算一个专家。

## 预填充 workspace 复用

预填充是 layer-major：同一批 token 完成某层 attention 后才进入该层 FFN。因此 attention-only 和 FFN-only 中间 tensor 生命周期不重叠。`DS4_METAL_PREFILL_STAGE_ALIAS=1` 会释放独立 FFN batch buffer，并让它们成为已结束 attention buffer 的不重叠 view；KV、跨阶段 HC、权重和计算公式均不改变。

在 Flash 0731 上，这会减少约 337,968 bytes/token：4096 batch 约省 1.29GiB，8192 batch 约省 2.58GiB。启动内存估算也同步计入该复用。数值与速度回归：

| 配置 | prompt tokens | prefill | 首 token | 备注 |
|---|---:|---:|---|---|
| 旧 exact，4096 chunk，600 slots | 14,735 | 27.36 t/s | `I` | 历史基线 |
| exact，4096 chunk，256 slots，stage alias | 3,977 | 38.80 t/s | 换行 | 满 4K 单块完成 |
| exact，8192 chunk，256 slots，无 alias | 14,735 | 未完成 | — | `mlock` 后只剩 12 slots |
| exact，8192 chunk，256 slots，stage alias | 14,735 | **41.55 t/s** | `I` | 完整运行，无 `mlock` 降级 |

8192 alias 配置的修正预算为 8.72GiB：KV 0.92、完整 graph buffer 5.84、resident model 0.28、expert cache 1.69GiB。相对旧 4096 长提示基线提升约 51.9%；相对同版 4K 单块的 38.80 t/s 只再提高约 7%，说明更大 batch 的边际收益已经明显下降。一次性 runner 在冷 engine 首个 prompt 超过 4096 token 时会自动选择该模式；常驻聊天 server 保持 4096 workspace 和较大的专家缓存，因为后续轮次只预填充新增后缀。

```sh
env \
  DS4F_FAST_CACHE_EXPERTS=256 \
  DS4_METAL_PREFILL_STAGE_ALIAS=1 \
  DS4_METAL_PREFILL_CHUNK=8192 \
  scripts/run-32k.sh exact @PROMPT.txt 1
```

1,001-token 对照中，4096 无 alias 与 alias 分别为 24.97/24.83 t/s，输出均为 `你好`，说明 view 复用没有改变该精准回归结果。完整日志位于 `results/benchmarks/prefill-15k-chunk8k-cache256-stage-alias-exact.log`。

把 KV 换到 SSD 不是当前优先方向：即使 8192 batch 下 KV 也只有 0.92GiB，最多回收不到 1GiB，却会给每层增加写入、读取和同步。现阶段应保留 KV 在统一内存，把可用空间优先分给 batch workspace 和当前层的 256 个专家槽。

## 连续专家 sidecar

原 GGUF 中一个 expert 的 gate/up/down 是三段读取。`tools/build_expert_pack.py` 可生成按 layer/expert 连续排列的只读 sidecar，使每次 miss 变成一次约 6.75MiB 的 `pread`。

Mini 上当前 sidecar 大小为 77,913,395,200 bytes，SHA-256：

```
95f68538cf7c54cac121ac4e5a08bc4ace8bb40613de684f9c78a77ad6d875de
```

构建与抽样校验：

```sh
tools/build_expert_pack.py MODEL.gguf IQ2Experts-packed.bin --sync
tools/build_expert_pack.py MODEL.gguf IQ2Experts-packed.bin --verify-only
```

构建运行器：

```sh
make ds4f-q4-speed
```

32K balanced 启动示例：

```sh
env \
  DS4F_SPEED_CACHE_AWARE_MASS_PCT=70 \
  DS4_METAL_STREAMING_EXPERT_PACK_PATH=models/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin \
  DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER=1 \
  ./ds4f-q4-speed \
  models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf \
  '你好' 32
```

也可以使用固定三档入口；默认应选 `exact`，`balanced` 与 `turbo` 都是显式近似：

```sh
scripts/run-32k.sh exact '你好' 32
scripts/run-32k.sh balanced '你好' 32
scripts/run-32k.sh turbo '你好' 32
```

打包前后 8-token greedy ID 完全一致。64-token、70% mass 对照中，原布局为 3.65 t/s，sidecar 为 3.78 t/s；它减少了系统调用和 readahead 开销，但不是从 3.8 到 5 t/s 的决定性加速。跨过 5 t/s 主要来自“exact prefill + 低 mass decode”，代价由上面的质量数据明确记录。
