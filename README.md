# ds4f-mini

面向 Apple M4 Mac mini（16GB 统一内存）的最小 DeepSeek V4 Flash 0731 推理实验。

目标被刻意限制为：只支持固定的量化 GGUF 权重，在 16GB 统一内存上尽可能完成连续生成，不引入 Ollama。自写运行器用于独立实现与数值验证；可选的 `ds4f-fast` 部署适配器会链接 DwarfStar 的公开 engine API，以便在完整自研 GPU 图完成前使用其已验证的 Metal graph。Kimi 工程仅作格式参考。

当前模型：

`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`

16GB Mini 的当前优化模型是 `DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf`：保持 routed IQ2 专家不变，把 embedding、attention、shared expert 与 output trunk 从 Q8 改为 Q4_K。32K 的构建、启动、质量与性能数据见 [`docs/32k-q4-runtime.md`](docs/32k-q4-runtime.md)。

当前 Q4 预填充实验可用 `DS4_METAL_PREFILL_STAGE_ALIAS=1` 复用 attention/FFN 的非重叠 batch workspace。M4/16GB 上，8192 chunk + 256 expert slots 已完整处理 14,735-token prompt，exact prefill 为 **41.55 t/s**，首 token 与旧 4096 基线同为 `I`；修正后的总计划内存为 8.72GiB。旧日志中只显示 0.25GiB graph buffer 的数字漏算了多数 batch tensor，不能用于新的内存规划。

当前状态：

- 独立 GGUF 解析、按张量 `pread` 读取和 Q8/Q2/IQ2 计算模块已完成。
- 固定版 GPT2 byte-level BPE tokenizer 已完成；chat prompt 采用 DwarfStar 同样的 `<｜User｜>...<｜Assistant｜></think>` 结构，不额外发出 BOS。
- 已完成压缩 KV cache、位置编码、prefill、连续 greedy generation 和多 token 输出。
- 已加入可选 Metal Q8_0 dense matvec；`DS4F_METAL_CACHE_GIB` 控制统一内存权重缓存，默认 10 GiB，上限 13 GiB。
- routed IQ2/Q2 专家切片有独立 LRU 缓存，`DS4F_EXPERT_CACHE_GIB` 控制预算，默认 4 GiB，上限 8 GiB；Metal 版本已加入 IQ2/Q2 专家线程组 kernel 和 gate/up 批处理。
- Metal 权重 cache 对专家 buffer 命中会刷新 LRU 次序；默认 10GiB 的空 prompt 8-token 回归中，专家 SSD 读取从 11.85GiB 降至 10.75GiB（miss 5739→5238），greedy token 不变。
- 路由结果确定后，会在 gate/up Metal command buffer 执行期间预取同一批专家的 down 权重；8-token 空 prompt 对照中 decode 从约 2.75 秒/token 降至约 2.59 秒/token，`DS4F_DISABLE_DOWN_PREFETCH=1` 可作诊断开关。
- `ds4f-fast` 的部署路径仍是 target-only；DSpark 仅在下文隔离的 DwarfStar 实验中接入，不能作为部署默认值。
- `ds4f-fast` 是可选的快速部署入口：仅支持固定 Flash 0731 模型，复用 reference-ds4 的公开 engine API 与 Metal graph；默认 128K context；`DS4F_FAST_CONTEXT_K` 只接受 `8/16/24/32/128/256`。8–32K 默认直接锁定 440 个专家（约 2.90GiB），不再为长 prefill 预留 3.38GiB 图缓存；显式设置 `DS4F_FAST_CACHE_GIB` 才回到旧的总预算布局，供诊断与历史对照使用。

2026-08-08/09 在 Mini 上重新验证：target-only SSD streaming 路径能够在 M4/16GB 上实际输出 token。`ds4f-fast` 通过 DwarfStar 的公开 engine API 复用已验证的完整 Metal graph；空 chat prompt（3 个 prompt token）连续生成 8 token，输出 `DeepSeek-V2, released in`，generation 为 **1.83 token/s**（6GiB cache）。在同一 16-token 对照中，6GiB 为 **1.86 token/s**，4GiB 为 **1.67 token/s**；连续生成应保持 6GiB 默认。因此“16GB 机器上把这个固定模型跑起来”的目标已有可部署入口。

自写运行器已完成独立 GGUF、tokenizer、原始/压缩 KV cache、连续 greedy decode 和 Metal Q8/IQ2/Q2 路径。空 chat prompt 的 8-token 历史回归只能证明这一条极小输入可连续运行，不能外推为通用数值正确性。

2026-08-09 实际生成审计：同一渲染 chat 输入下，自写 Metal 与 `ds4f-fast` 在中文短提示词前 4 个 token ID 一致；但英文 `Explain one plus one in one word.` 在共同首 token `One` 后分叉为自写 Metal 的 `.` 与部署路径的 ` word`。同一自写代码的 CPU 路径输出 `One word`，因此问题在自写 Metal 数值累积，而非模型、tokenizer 或硬件随机性。`ds4f-generate-metal` 仅作研究/差分工具，**不得作为部署或数值基线**；部署只使用 `ds4f-fast` 的参考完整 Metal graph。设置 `DS4F_FAST_TRACE_IDS=1` 可输出部署路径 token ID 以做回归。
同日的差分定位进一步确认：将参考图导出的 Q、raw KV 与 compressed KV 注入自写的短上下文合并 FlashAttention 后，`kqv_back` 相对误差为 `5.0e-8`；attention 规约已足够接近。剩余分叉来自自写通用 Q8 图在前序层留下约 `1e-5` 的状态差，少数值跨过 E4M3/FP8 舍入中点后放大。因此不能以自写 runner 的速度或输出作部署结论。生产路径的实际 token-ID 回归由 `make check-production MODEL=/path/to/Flash-0731.gguf` 固定：英文 `6111 2004 28`，中文 `28669 8570 988 819`；所有速度改动都必须通过它。


自写路径仍是实验运行器：默认 10GiB Metal weight cache 下，短 prompt prefill 约 10 秒，decode 约 2.6–2.8 秒/token（约 0.37 token/s）。性能差距来自逐层 CPU/Metal 同步和专家切片调度。`ds4f-fast` 已把同一模型接入完整单-token Metal graph、GPU router 与 SSD expert cache，当前是 Mini 上的实际推荐入口；下一阶段才是把这张图改为项目自有实现。

DS4F_PROFILE=1 可输出 attention、FFN 与 head 的分项时间。10GiB 是自写路径的安全默认值；提高到 11GiB 以上会造成统一内存压力并使连续 decode 变慢，因此不把它设为默认值。

当前最快的 target-only 启动命令：

```sh
cd /Users/zexi/workspace/ds4f-mini
make fast
./ds4f-fast \
  reference-ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  "你好" 16
```

`ds4f-fast` 是受限部署适配器：只暴露这个固定模型的 greedy SSD-streaming 路径，静态链接 `reference-ds4` 的公开 engine API，并在运行时使用其 Metal shader；它不修改 reference 源码。自写路径保持独立，作为下一步将整张 GPU 图迁入项目本身的数值和结构基线。

`--ssd-streaming` 是在 16GB 机器上成立的关键；不要省略。对于 8–32K 紧凑布局，默认是直接锁定 440 个专家（约 2.90GiB），不使用旧的 6GiB 总预算；对于 128K/256K，默认仍是 6GiB 总专家预算。旧的 16-token 空 prompt 对照中，6GiB 为 **1.86 token/s**，4GiB 为 **1.67 token/s**；8GiB 会因 macOS 无法锁住足够的专家页而显著退化。

面向这台 Mini 的 target-only 部署只保留 `8/16/24/32/128/256K` 六档。8–32K 是紧凑短任务布局：固定 440 个可锁定专家，随上下文上限增加的仅是 KV 与图 buffer；128K/256K 保持旧的“总专家预算 + prefill 保留区”布局。这样把 32K 作为容量上限时，不会为实际不会发生的长 prefill 固定占用 3.38GiB。

2026-08-09 的旧 32K/6GiB 总预算布局已完成一次 32,024-token 输入预填充并输出 `Bob=34`、`Alice=52`（prefill **27.58 t/s**，满 32K generation **0.16 t/s**），因此模型的 32K KV 路径本身可执行。针对现在的“1–3 回合短任务”目标，新紧凑布局将 440 个专家直接锁定：8/16/24/32K 的内存计划分别为 **4.37/4.50/4.62/4.75GiB**（32K 其中 KV 0.61GiB、图 buffer 0.25GiB、常驻非专家 0.99GiB、专家 cache 2.90GiB）。同一 33-token 中文不可逆删除决策均正确输出“先备份再确认”；prefill 为 **2.21/2.23/2.25/2.23 t/s**，generation 为 **0.82/0.89/0.87/0.90 t/s**。这证明把上限降到 8K 不会显著提高短请求 decode；瓶颈是 SSD 专家 miss，而非 KV 容量。
2026-08-09 在 M4/16GB 的实测中，128K 内存计划为 9.35GiB（KV 1.36GiB、图 buffer 1.00GiB、398 个动态专家 2.62GiB）；256K 为 10.35GiB（KV 2.36GiB、buffer 2.00GiB）。256K 会触发图工作集保护，把总专家预算从 6GiB 降至 5GiB，只保留 246 个动态专家，因此连续 8-token generation 实测约 **1.21 token/s**；128K 保留 398 个专家，实测约 **1.42 token/s**，是推荐默认值。两档均通过英文和中文 token-ID 回归；512 个重复片段的 batch prefill 均完成 43/43 层并输出 token（128K 22.65 t/s，256K 24.81 t/s）。


### 16GB 专家缓存的已验证边界

旧的 128K/256K 正式路径采用 6GiB 总专家预算：其中 3.38GiB 是图预填充保留区，2.62GiB 是 398 个专家的可锁定动态缓存。每个 decode token 最多路由 `43 × 6 = 258` 个专家，因此这个容量小于两个 token 的工作集；持续 decode 的主要瓶颈是 miss 后从内部 SSD 读取权重，而不是算术 kernel。

2026-08-09 的隔离实验均先通过生产 token-ID 回归后再计时，结论如下：

- 预填充后把全部 6GiB 改为 910 个**未锁定**专家，英文和中文回归均正确，但 macOS 大量分页，generation 降为 0.25 token/s；不采用。
- 将预填充保留区从两层降为一层，虽可请求 654 个专家，但 `mlock` 只能锁约 2.99GiB，保护逻辑最终把运行时缓存降到 151 个，generation 为 0.31 token/s；不采用。
- 将专家 slab 从 Shared 改为 CPU 可写的 Managed 也不改变 token ID，但 654 槽仅 0.73 token/s、910 槽仅 0.23 token/s；在 M4 统一内存上它不会形成额外的常驻缓存层，不采用。
- 将 SSD `pread` 并发从默认 9 提至 18 不改变 token ID，但在常驻两请求复测中没有稳定胜过默认值；保持参考默认值。
- 同一 32-token 对照下，将 `pread` 并发降至 6 为 1.61 token/s，默认 9 为 1.63 token/s；前者没有稳定优势，保持默认 9。
- 设置 `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1` 后，16-token 对照在第一个输出 token 后超过一分钟仍未完成；`F_RDADVISE` 预读是这条 SSD 路径必要的默认策略。
- 临时将“命中/缺失专家分阶段执行”的阈值从至少 3 个缺失降到任意混合命中，首个生产 token 向量仍一致，但 32-token generation 从 1.63 降至 1.62 token/s；不采用。
- eviction 时的文件页 `DONTNEED` 在参考实现中默认关闭，不是可回收的默认性能损失。
- 在旧 128K 布局中，prefill 后把动态缓存从 398 增至 440 槽（约 2.90GiB）会在首次 decode 的新增 6.75MiB 专家页上 `mlock` 失败并中断生成；这是旧布局无法采用 440 槽的原因。紧凑 8–32K 布局不分配该 prefill 保留区，因而可以从启动时直接锁定 440 槽。
- 在 128K 下临时请求 7GiB 总专家预算（549 个专家）同样在约 2.99GiB `mlock` 处失败，保护逻辑将缓存降至 151 槽；不采用。

因此，若目标是隐私优先的短任务子代理，推荐 DS4F_FAST_CONTEXT_K=32 的紧凑布局与 ds4f-reuse 常驻进程。2026-08-09 的 exact CPU-router 变体会自动启用：它在 CPU 上对同一量化 gate 做完整 top-6 路由，再把相同专家 ID 与权重写回 Metal；没有减少专家或采用近似。英文、中文固定 token-ID 回归均通过；同一短中文请求连续两次各生成 8 token 为 **1.82 / 1.77 token/s**，约为原 GPU-router SSD 路径的两倍。设置 DS4_METAL_DISABLE_STREAMING_IQ2_CPU_ROUTER=1 可退回旧路径。它仍低于 5 token/s：profile 显示每层平均仍缺失约 3.76 个所选专家，单纯再降低 context 上限并不能解决 SSD expert miss。

### 多请求复用缓存

对于连续的独立请求，使用 `ds4f-reuse` 让 engine 与当前所选专家 cache 保持常驻；每一行 stdin 都是一个新的 chat prompt，上一条的 KV / 对话历史不会混入下一条，但 Metal SSD 专家 cache 会留下来：

```sh
cd /Users/zexi/workspace/ds4f-mini
make fast
printf '%s\n' '你好' '介绍一下你自己' | env DS4F_FAST_CONTEXT_K=32 ./ds4f-reuse \
  reference-ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  16
```

`ds4f-reuse` 不改变输出算术，也不会混入上一条的 KV。它保留专家 cache，能避免进程重启，但缓存命中取决于后续请求的路由；对短任务应实测自己的提示词，不要把它当作固定的吞吐加速器。

## DwarfStar 的 SSD + DSpark 实验

`patches/` 与 `scripts/build-dwarfstar-dspark-ssd-experiment.sh` 提供一个**不修改** `reference-ds4` 的可重复实验构建。它修复 DwarfStar 原先拒绝 `--ssd-streaming --mtp` 的限制；target 与 DSpark support 保留独立 Metal views，最终仍由 target 的普通逐 token SSD decode 验证草稿，因此提交的 KV cache 和 greedy token 与 target-only 路径一致。

2026-08-09 的非驻留原型用 `DS4_DSPARK_SSD_NONRESIDENT=1` 开启：support 只映射 19 段非专家权重（0.52GiB），路由专家仍由主模型现有 SSD expert cache 按需 `pread`；5 个 draft row 会逐行执行，从而复用单 token 的 IQ2/Q2 专家路径。它不是把完整 5.58GiB support GGUF 常驻到 16GB 统一内存。

在 Mini 上构建并做确定性数值验收：

~~~sh
cd /Users/zexi/workspace/ds4f-mini
./scripts/build-dwarfstar-dspark-ssd-experiment.sh

cd reference-ds4
env DS4_DSPARK_SSD_NONRESIDENT=1 DS4_DSPARK_SCHEDULER=0 DS4_DSPARK_STATS=1 DS4_DSPARK_SPEC_LOG=1 ../bin/ds4-dspark-ssd -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf --mtp gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf --dspark --ssd-streaming --ssd-streaming-cache-experts 6GB -c 2048 --temp 0 -p "Explain one plus one in one word." -n 2
~~~

完整 support 映射与 0.52GiB 非驻留路径的同一审计都得到 `proposal0=1309`、`confidence0=2.114`；因此默认 confidence 0.7 会保留草稿前缀。`-n 2` 回归会提交一个草稿 token，最终输出 `We need` 与同一 `--temp 0` 的 target-only 输出完全一致。`-n 6` 的确定性复验中两次各提出并接受 2 个草稿；但 sequential verifier 总计约 6506ms，generation 约 0.65 token/s，仍**不加速**。它的价值是把实验从“全量 support 映射会挤压 16GB”推进到可运行的按专家读取、且与完整 support proposal 数值对齐的闭环；默认部署仍使用上文 `ds4f-fast` target-only 路径。

每次提案把 GPU 视图切到 support 后，运行器都会使 target 的静态 decode-view 缓存状态失效；下一次 target 验证会重新安装 target spans。该修复已用短 raw prompt（Hello，输出 #World）与 target-only 做逐字节对照。

随后在 `/tmp` 做的 target batch-verifier 探针会先重装 target 静态 views，再以 SSD expert cache 验证整段草稿并回滚/replay；6-token 输出也逐字节等于 target-only。然而一次 5-token 草稿只接受 2 个 draft token，batch verify 为约 7685ms、精确 replay 为约 2630ms、总 generation 约 0.33 token/s，慢于顺序 verifier。该探针没有进入 `patches/` 或部署二进制；在 16GB SSD 约束下，只有能同时提高 acceptance 且避免 replay 的 verifier 才有加速可能。

在 Mini 上运行：

```sh
cd /Users/zexi/workspace/ds4f-mini
make
./ds4f-tokenize /path/to/model.gguf "你好，世界" --chat
./ds4f-generate /path/to/model.gguf "你好，世界" 8 4096
make dspark
./ds4f-dspark-probe /path/to/model.gguf /path/to/DeepSeek-V4-Flash-DSpark-support-0731.gguf

# Metal 版本；10GiB 是当前的安全默认值
make metal
DS4F_METAL_CACHE_GIB=10 ./ds4f-generate-metal /path/to/model.gguf "你好，世界" 8 4096

# 可选：输出每个 token 的 attention/FFN/head 分项时间
DS4F_PROFILE=1 DS4F_METAL_CACHE_GIB=10 ./ds4f-generate-metal /path/to/model.gguf "你好，世界" 1 4096
```

`ds4f-first-token` 仍保留作为单 token 回归测试；`ds4f-generate` 是当前连续生成入口。`ds4f-dspark-probe` 已能验证 Flash 0731 support 文件；自写 runtime 尚未接入 DSpark 闭环，而上文的隔离 DwarfStar 实验已实现可验收的非驻留原型。Metal 目前已覆盖 Q8_0 共享矩阵和 routed IQ2/Q2 专家，但还没有 DwarfStar 那样的完整 fused MoE/批量 verifier。
