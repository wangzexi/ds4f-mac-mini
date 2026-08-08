# ds4f-mini

面向 Apple M4 Mac mini（16GB 统一内存）的最小 DeepSeek V4 Flash 0731 推理实验。

目标被刻意限制为：只支持固定的量化 GGUF 权重，在 16GB 统一内存上尽可能完成连续生成；不引入 Ollama，也不把 DwarfStar 或 Kimi 工程作为运行时依赖。两个工程仅用于理解 GGUF、量化格式、KV 压缩和 DSpark 公式。

当前模型：

`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`

当前状态：

- 独立 GGUF 解析、按张量 `pread` 读取和 Q8/Q2/IQ2 计算模块已完成。
- 固定版 GPT2 byte-level BPE tokenizer 已完成；chat prompt 采用 DwarfStar 同样的 `<｜User｜>...<｜Assistant｜></think>` 结构，不额外发出 BOS。
- 已完成压缩 KV cache、位置编码、prefill、连续 greedy generation 和多 token 输出。
- 已加入可选 Metal Q8_0 dense matvec；`DS4F_METAL_CACHE_GIB` 控制统一内存权重缓存，默认 10 GiB，上限 13 GiB。
- routed IQ2/Q2 专家切片有独立 LRU 缓存，`DS4F_EXPERT_CACHE_GIB` 控制预算，默认 4 GiB，上限 8 GiB；Metal 版本已加入 IQ2/Q2 专家线程组 kernel 和 gate/up 批处理。
- DSpark support GGUF 已下载并通过结构验证（约 5.58 GiB）；尚未接入运行时，因此当前程序仍是 target-only decoding。

2026-08-08 在 Mini 上重新验证：DwarfStar 的 target-only SSD streaming 路径能够在 M4/16GB 上实际输出 token。对空 chat prompt（3 个 prompt token）生成 2 token 时，日志给出的 generation 速度为 **2.99 token/s**；其内存规划为 5.84 GiB（KV 0.61 GiB、常驻 embedding 0.99 GiB、SSD expert cache/reserve 4 GiB 等）。因此“16GB 机器上把这个固定模型跑起来”的目标已经有一条可用且有速度的参考运行路径。

自写运行器已完成独立 GGUF、tokenizer、原始/压缩 KV cache、连续 greedy decode 和 Metal Q8/IQ2/Q2 路径。对同一空 chat prompt 的 8-token 回归中，DwarfStar 与 ds4f-generate-metal 的 top-k 排名和每一步选中的 token 全部一致；第 0 步 top-1 logit 差为 +0.0120345。这证明当前数值路径可以稳定地连续生成，仍不代表长上下文、多种 prompt 和 DSpark 的完整覆盖。

自写路径仍是实验运行器：默认 10GiB Metal weight cache 下，短 prompt prefill 约 10 秒，decode 约 2.6–2.8 秒/token（约 0.37 token/s）。性能差距来自逐层 CPU/Metal 同步和专家切片调度；DwarfStar 的单 token GPU graph、GPU router 与 SSD expert cache 是当前实际推理的推荐路径。

DS4F_PROFILE=1 可输出 attention、FFN 与 head 的分项时间。10GiB 是自写路径的安全默认值；提高到 11GiB 以上会造成统一内存压力并使连续 decode 变慢，因此不把它设为默认值。

当前可用的 target-only 启动命令：

```sh
cd /Users/zexi/workspace/ds4f-mini/reference-ds4
./ds4 \
  -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 4GB \
  --temp 0 --nothink -n 16 -p '<｜User｜>你好<｜Assistant｜></think>'
```

--ssd-streaming 是在 16GB 机器上成立的关键；不要省略。4GB expert cache 是当前 Mini 上的最佳安全设置。6GB 在短请求上无收益；8GB 会因 macOS 无法锁住足够的专家页而退化到约 0.63 token/s。


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
