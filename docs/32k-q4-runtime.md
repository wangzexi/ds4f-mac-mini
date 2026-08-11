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

## 单活跃会话待命

部署只允许一个活跃会话，但不放弃磁盘 KV：当前会话的 KV、路由热度和 Decode
专家池留在统一内存；每次响应结束后，同一 KV 和热度也写入 10 GiB 磁盘 LRU。
切换到另一段旧会话或服务重启时，先尝试恢复该磁盘快照；未命中才重新 Prefill。
因此磁盘 KV 是“会话切换/重启恢复层”，不是与当前会话竞争的第二个内存会话。

固定启动器还常驻预热 hash layer 0 的全部 256 个专家（1.69 GiB）。它们在
启动时只从 SSD 读取一次；因 L0 的路由仅依赖 token ID，任何 KV 未命中都可直接
复用。2026-08-12 的 Mini 烟雾测试中，5-token 新提示加 2-token exact 输出的
请求在 server ready 后 L0 SSD expert-read 为 0；Prefill→Decode 相位转换保留
了 5.68 GiB 实际专家缓冲，低于 6.37 GiB Decode 预算。完整数值回归仍输出
`6111 2004 28` 与 `23385 33951 6573 303`。

同一启动器还会在监听前预读 Decode 的 90 个静态主干 span（4.60 GiB）。因此
`server ready` 的含义是 L0 和完整 Decode 主干均已就绪；第一条请求直接以
`static_trunk=reuse` 进入 exact Prefill，而不是在首 token 前再花约 2.3--2.6 秒
读取主干。2026-08-12 的冷启动后 13-token 请求实测 prompt 7.20 s、首 token
0.97 s、总计 8.16 s；同量级的 lazy-trunk 冷路径约 11 s。该策略只在单活跃会话
启动器启用，可通过 `DS4F_SERVER_PRELOAD_STATIC_DECODE_TRUNK=0` 回退。

这不是盲目锁死所有权重：相位规划按真实 Metal 缓冲字节数（包括 shared Prefill
staging 与 slab 预留）而不是逻辑条目数判断。若下一阶段装不下，缓存仍会整体释放，
以保证 16GB 机器不因保留热缓存进入内存压力。

## 连续追问的 Decode 主干复用

一次 Decode 会读取完整非路由主干的 90 个 span（4.60 GiB）。旧路径在下一轮
Prefill 一开始又把这份 map 替换成逐层 map，因而下一次首 token 必须再次读取同一
4.60 GiB。现在只要同一 server 的上轮 Decode map 仍在，增量 Prefill 直接在它上面
执行；路由专家仍按原有逐层 exact 调度，KV 恢复和模型数学均不变。

内存规划会先把这份主干计入 Prefill 基础占用，并把专家缓存从短 Prefill 的约
10.65 GiB 自动下调到约 6.33 GiB，故不是把两份模型叠加。2026-08-12 的 Mini 两轮
回归中，第一轮 7-token prompt + 1-token output 后，第二轮复用 8-token KV 并处理
5-token 后缀：输出 token 与旧路径相同（`I`），第二轮没有 `static_decode` SSD read，
总时延由 7.57 s 降至 3.74 s（Prefill 4.24→2.75 s，首 Decode 3.34→0.99 s）。
服务重启、没有上一轮 Decode，或内存计划不允许保留时，仍安全回退到原来的完整读取。

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

又尝试只在低熵 router 层执行这种舍弃：20% 阈值分别限制在归一化熵不高于
80% 与 85%，仍都在第 6 个 token 分叉，15%/80% 也同样在第 6 个 token
分叉且只有 2.04 t/s。低熵不能区分出“可安全补零”的 miss 专家；该保护实验
已撤回，避免把一个看似合理但无质量收益的近似开关留在运行时。

解码时第 1、2 个 hash 路由层的专家可由 token ID 精确提前算出；也试过在
前一层 MoE 执行期间预读这些 six-expert 集合。开/关在同一 950 槽 32-token
trace 中完全一致，但 generation 都是 2.05 t/s。现有 selected-expert early
load 已处于同一同步边界，额外读请求没有形成可用重叠，因此该实验未保留。

Prefill 的尾部专家交接仍是独立的精确实验：
`DS4_METAL_PREFILL_TAIL_EXPERT_HANDOFF=1` 默认留下每层最后一行的 6 个
已在内存中的专家；`DS4_METAL_PREFILL_TAIL_EXPERT_HANDOFF_KEEP=1..256`
只用于测试更长尾部并集。`你好`、两个输出 token、600 槽的对照中，关闭/保留
6 个的累计 cache hit/miss 为 332/1017 与 519/840，generation 为 0.60 与
0.70 t/s。它减少了首个实际 decode 的 SSD miss，但会占用全局 decode cache；
32-token 连续生成尚未显示稳定净收益，故服务默认仍关闭。短提示只有一行可留，
设为 18 与 6 得到相同集合，不能据此推断长尾效果。

主干启动预热后的受控 A/B 也否决了较小交接：相同启动、相同预热、11-token
prompt + 4-token exact 输出中，不交接为 2.73 s Decode；保留每层 6 个或仅 1 个
tail expert 均约 4.70--4.76 s，输出文本相同。它会扰乱全局专家缓存的生命周期，
故不应默认开启，即使其不改变数值。

一次 8-token 精确 Decode 运行的底层计时也确认了下一阶段应优先解决的问题：600
专家槽时命中 2,056、miss 2,270（47.5%），未命中专家合计读取 14.34 GiB；688 次
selected-expert 绑定平均 6.40 ms，其中 miss 的载入约 9.91 ms、`pread` 本身约
9.22 ms/载入任务。CPU Router 和 30,720-entry 淘汰扫描分别只占极小部分。因此不再
投入 Router 或扫描微优化；真正可量级的路径是以较低位宽主干腾出空间给专家缓存，
或在不破坏数值的前提下隐藏 expert miss I/O。

专用边界 profiler 显示 `A` / `你好` 的 40 个可学习层中，若按最后一行实际 Router
权重挑选 top-1 尾部专家，下一 token 平均分别会复用 0.80 / 0.90 个（保留全部六个
则为 3.33 / 3.83 个），证明局部性本身存在。曾实现 weight-ranked K=1 交接来验证
这一点；runner 的 `6111 2004 28` trace 一致且只有 0.76→0.77 t/s 的噪声差异，但
真实 HTTP 服务短请求分叉为 `Two.` 与 `One word: "`。该实现已撤回，服务默认继续
关闭尾部交接。根因是 Prefill 到 Decode 的缓存所有权/同步仍不够严谨，不能因为命中
看似增加就破坏精确 greedy 结果。

同一受控 fixture 还比较了静态主干常驻下的 Decode 缓存上限：11.50 GiB 工作集为
967 槽 / 6.37 GiB，12.00 GiB 为 1,043 槽 / 6.88 GiB；两者输出相同，4-token
Decode 分别为 2.50 s（1.60 t/s）和 2.53 s（1.58 t/s）。额外 0.50 GiB 没有可测
收益，因此正式配置仍保持 11.50 GiB，把余量留给 Metal 和系统而非盲目扩张缓存。

短专家读取的 `F_RDADVISE` 也保留开启。相同干净服务、相同 13-token prompt +
4-token 输出中，关闭 `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1` 后 Prefill
从 6.35 s 增至 6.70 s，Decode 从 2.50 s 增至 2.71 s；此前对“完整 staged layer”
的否决不适用于这条按 selected experts 读取的路径。

Prefill 结束时把最后一行实际选择的 six experts 复制/保留给 Decode 的精确开关
`DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED=1` 也完成了主干启动预热后的受控
A/B。隔离 KV 目录、相同 13-token prompt 与 4-token greedy 输出 `您好！您提到的` 下，
启用 `K=1` 的 Prefill/Decode/总时间为 6.42/2.49/8.91 s；默认路径为
6.35/2.50/8.85 s。输出完全一致但没有净收益，故保持关闭。它可在后续改变缓存实现后
重新评估，但当前不应为了“首 token 命中”干扰全局 Decode cache。

也完整复查了 CPU router 是否应回退到 GPU router。关闭
`DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER` 后，正式 `check-production` 两组 greedy
token-ID 回归都通过，短 4-token fixture 的输出与耗时也近似相同；但干净服务、5-token
prompt、32-token greedy 输出的受控 A/B 明确显示 GPU router 为 3.54 s Prefill +
19.31 s Decode（1.66 t/s），CPU router 为 3.53 s + 16.27 s（1.97 t/s），文本完全相同。
GPU 路由的 selected-id readback/异步交接没有在该模型上抵消调度开销；CPU router 能在
MoE 编码前直接给出 six experts，故仍为 Mini 的默认精确路径。

专家 route heat 的时间衰减也按真实连续会话测过。夹具先对 `你好` 解码 32 tokens，
再带着该精确 assistant 历史追加新追问并解码 8 tokens；半衰期 8/16/64 的第二轮 Decode
分别为 1.90/1.87/1.89 t/s（增量 Prefill 8.74/8.74/8.34 s）。未出现超过 SSD/Metal
运行波动的稳定收益，故固定保留 16-token 半衰期，不把实验参数暴露到生产运行时。

### 单会话 KV 与 L0 的独立性

服务空闲时仍保留三个彼此独立的状态：完整 L0（256 experts）、静态 Decode 主干映射，
以及最近会话的内存 KV。响应结束后，KV 还会以 LRU 形式写入 10 GiB 磁盘缓存，并携带
对应的 router heat。

- 内存 KV 命中：不读磁盘 KV，也不读 L0。
- 内存未命中、磁盘中有精确 token-prefix：只读回该 KV/heat（实测约数毫秒），L0 和
  静态主干继续复用。
- 磁盘也未命中：从新消息重新 Prefill 并建立新的 KV；依然不会重读 L0。后续层的可换
  专家按正常调度补入，但 L0 预热与 KV 是否命中无关。

真实 server 的内存检查进一步否决了把它设为默认：5-token 输入、2-token输出时，
开启 6-entry handoff 的进程 RSS 约为 7.1 GiB，而空闲默认服务约 2.0 GiB。
原因不是逻辑条目数，而是这些条目仍引用各层完整的 Prefill staging buffer。除非先将
保留专家复制为紧凑的普通 cache slots 并释放整层 buffer，否则该路径只能作为诊断，
不能占用 Mini 宝贵的 Decode 预算。

尾部交接也没有改变“低权重 miss 补零”的结论。保留 6 个尾部专家、600 槽、
`你好` 16-token 对照中，0/10/15/20% 的 generation 为
1.82/1.81/1.80/1.81 t/s；10% 与 exact 的全部 token ID 一致，15% 与 20%
均在第 6 个 token 分叉。因此 cache 冷启动不是低权重补零没有收益的原因，
该近似仍不进入任何默认档位。

同一轮 profile 还显示，前三个 hash 层因为路由近似均匀，950 槽全局热度
缓存中的命中率只有约 6–11%，而可学习路由层约为 62–85%。曾尝试把完整 hash
层永久保留在缓存里；但 Mini 实际稳定可锁定的 expert slab 约 4GiB。仅固定第
0 层就使后续批量加载在该上限处无法获得可复用槽并出现 Metal buffer allocation
failure。因此它不是可行的缓存优化；保留更多专家必须以可回收的全局缓存方式进行，
不能硬分区。

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
