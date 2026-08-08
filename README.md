# ds4f-mini

面向 Apple M4 Mac mini（16GB 统一内存）的最小 DeepSeek V4 Flash 0731 推理实验。

目标被刻意限制为：只支持固定的量化 GGUF 权重，在 16GB 统一内存上尽可能完成连续生成，不引入 Ollama。自写运行器用于独立实现与数值验证；可选的 `ds4f-fast` 部署适配器会链接 DwarfStar 的公开 engine API，以便在完整自研 GPU 图完成前使用其已验证的 Metal graph。Kimi 工程仅作格式参考。

当前模型：

`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`

当前状态：

- 独立 GGUF 解析、按张量 `pread` 读取和 Q8/Q2/IQ2 计算模块已完成。
- 固定版 GPT2 byte-level BPE tokenizer 已完成；chat prompt 采用 DwarfStar 同样的 `<｜User｜>...<｜Assistant｜></think>` 结构，不额外发出 BOS。
- 已完成压缩 KV cache、位置编码、prefill、连续 greedy generation 和多 token 输出。
- 已加入可选 Metal Q8_0 dense matvec；`DS4F_METAL_CACHE_GIB` 控制统一内存权重缓存，默认 10 GiB，上限 13 GiB。
- routed IQ2/Q2 专家切片有独立 LRU 缓存，`DS4F_EXPERT_CACHE_GIB` 控制预算，默认 4 GiB，上限 8 GiB；Metal 版本已加入 IQ2/Q2 专家线程组 kernel 和 gate/up 批处理。
- Metal 权重 cache 对专家 buffer 命中会刷新 LRU 次序；默认 10GiB 的空 prompt 8-token 回归中，专家 SSD 读取从 11.85GiB 降至 10.75GiB（miss 5739→5238），greedy token 不变。
- 路由结果确定后，会在 gate/up Metal command buffer 执行期间预取同一批专家的 down 权重；8-token 空 prompt 对照中 decode 从约 2.75 秒/token 降至约 2.59 秒/token，`DS4F_DISABLE_DOWN_PREFETCH=1` 可作诊断开关。
- DSpark support GGUF 已下载并通过结构验证（约 5.58 GiB）；尚未接入运行时，因此当前程序仍是 target-only decoding。
- `ds4f-fast` 是可选的快速部署入口：仅支持固定 Flash 0731 模型，复用 reference-ds4 的公开 engine API 与 Metal graph；默认 6GiB SSD expert cache，可用 `DS4F_FAST_CACHE_GIB=1..6` 调整。

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

`--ssd-streaming` 是在 16GB 机器上成立的关键；不要省略。`ds4f-fast` 默认使用 6GiB 专家 cache：完整 8-token 空 prompt 回归为 1.83 token/s，输出与数值基线一致；最新相同 16-token 请求对照中，6GiB 为 **1.86 token/s**，4GiB 为 **1.67 token/s**，因此保持默认 6GiB。8GiB 会因 macOS 无法锁住足够的专家页而显著退化。

为这个限定的 Mini 部署，`ds4f-fast` 将 context 固定为 2048 token：这把 KV 预算从约 0.61GiB 降到约 0.18GiB，并将短 prompt 的 prefill 从约 0.33 提升到 0.61 token/s；它不改变连续 greedy 输出，decode 仍主要受 SSD expert streaming 限制。


### 16GB 专家缓存的已验证边界

正式路径固定采用 6GiB 总专家预算：其中 3.38GiB 是图预填充保留区，2.62GiB 是 398 个专家的可锁定动态缓存。每个 decode token 最多路由 `43 × 6 = 258` 个专家，因此这个容量小于两个 token 的工作集；持续 decode 的主要瓶颈是 miss 后从内部 SSD 读取权重，而不是算术 kernel。

2026-08-09 的隔离实验均先通过生产 token-ID 回归后再计时，结论如下：

- 预填充后把全部 6GiB 改为 910 个**未锁定**专家，英文和中文回归均正确，但 macOS 大量分页，generation 降为 0.25 token/s；不采用。
- 将预填充保留区从两层降为一层，虽可请求 654 个专家，但 `mlock` 只能锁约 2.99GiB，保护逻辑最终把运行时缓存降到 151 个，generation 为 0.31 token/s；不采用。
- 将专家 slab 从 Shared 改为 CPU 可写的 Managed 也不改变 token ID，但 654 槽仅 0.73 token/s、910 槽仅 0.23 token/s；在 M4 统一内存上它不会形成额外的常驻缓存层，不采用。
- 将 SSD `pread` 并发从默认 9 提至 18 不改变 token ID，但在常驻两请求复测中没有稳定胜过默认值；保持参考默认值。
- eviction 时的文件页 `DONTNEED` 在参考实现中默认关闭，不是可回收的默认性能损失。
- 保持 6GiB 总预算但在 prefill 后只将动态缓存从 398 增至 440 槽（约 2.90GiB）也不可行：首次 decode 的新增 6.75MiB 专家页就 `mlock` 失败并中断生成。这直接验证 398 槽已是完整图在本机的可执行上限，不是保守留量。

因此当前可部署且数值受回归保护的速度范围是短请求约 1.6–2.6 token/s，具体取决于路由、SSD 页缓存和已有专家 cache。对多条独立请求，应优先使用 `ds4f-reuse`；它保留专家 cache、重建各自的 KV，避免串话。若要跨过这个上限，需要更多可锁定统一内存、更快的外部存储，或实现不需常驻 support 权重的真正层级 speculative verifier；单纯把更大的专家 cache 交给 macOS 分页不会加速。

### 多请求复用缓存

对于连续的独立请求，使用 `ds4f-reuse` 让 engine 保持常驻；每一行 stdin 都是一个新的 chat prompt，上一条的 KV / 对话历史不会混入下一条，但 Metal SSD 专家 cache 会留下来：

```sh
cd /Users/zexi/workspace/ds4f-mini
make fast
printf '%s\n' '你好' '介绍一下你自己' | ./ds4f-reuse \
  reference-ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  16
```

Mini 上两条 4-token 空请求的验证中，首条 cold generation 为 0.85 token/s，第二条在已保留专家 cache 下为 2.17 token/s；它不改变输出算术，只减少后续独立请求的 SSD cache 冷启动。

## DwarfStar 的 SSD + DSpark 实验

patches/ 与 scripts/build-dwarfstar-dspark-ssd-experiment.sh 提供一个**不修改**
reference-ds4 的可重复实验构建。它修复了 DwarfStar 原先拒绝
--ssd-streaming --mtp 的限制，并在 SSD 模式下：

- 保留 target 和 DSpark support 的独立 Metal model views；
- 在每次提案前恢复 support 映射；
- 用 target 的普通逐 token SSD decode 验证草稿，保证 KV cache 和最终 greedy
  token 与 target-only 完全一致。

在 Mini 上构建并运行：

~~~sh
cd /Users/zexi/workspace/ds4f-mini
./scripts/build-dwarfstar-dspark-ssd-experiment.sh

cd reference-ds4
DS4_DSPARK_STATS=1 \
../bin/ds4-dspark-ssd \
  -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 4GB \
  --mtp gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --temp 0 --nothink -n 16 \
  -p '<｜User｜>a<｜Assistant｜></think>'
~~~

2026-08-08 的验证中，DSpark 16-token 输出与 target-only 输出逐字节相同；共提出
7 个草稿 token、接受 4 个，且没有 verifier error。它当前只有功能价值：每个草稿
仍由 target 逐 token 验证，速度明显慢于上面的 target-only SSD streaming 路径。
因此默认仍使用 target-only 命令；DSpark 留作正确性验证和后续实现真正的层级
streaming verifier 的基础。

2026-08-09 的 Mini 复验表明，这条正确性实验不能作为部署加速器：即使主模型
cache 设为 4GiB，DSpark support 映射仍会触发明显 swap；36 秒只生成到 `Deep`，
故已主动终止。它没有数值错误，但 16GiB M4 上的支持模型常驻和 target SSD
streaming 无法同时满足低延迟。除非实现真正的层级 streaming verifier 并显著降低
support 模型常驻页，否则不要把 `ds4-dspark-ssd` 用于实际服务。

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

`ds4f-first-token` 仍保留作为单 token 回归测试；`ds4f-generate` 是当前连续生成入口。`ds4f-dspark-probe` 已能验证 Flash 0731 support 文件，但 DSpark 推理闭环仍未接入。Metal 目前已覆盖 Q8_0 共享矩阵和 routed IQ2/Q2 专家，但还没有 DwarfStar 那样的完整 fused MoE/批量 verifier。
