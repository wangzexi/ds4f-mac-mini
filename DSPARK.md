# DSpark 研究记录

## 已确认的 support 格式

`DeepSeek-V4-Flash-DSpark-support-0731.gguf` 与当前锁定的 Flash 0731 主模型匹配：

- 3 个 support stage；每轮最多提出 5 个 token。
- target hidden 来自主模型第 40、41、42 层，经 `mtp.0.main_proj`（12288 -> 4096）进入 support 链。
- 最后一阶段包含 Markov head（rank 256）和 confidence head。
- support 文件 81 个张量、约 5.58 GiB；结构校验 `ds4f-dspark-probe` 已通过。
- noise token 为 128799，不能把 support GGUF 当成独立模型运行。

## 16GB Mini 的现实限制

DwarfStar 的参考二进制在 M4 Mini 上报告：`--ssd-streaming is not compatible with --mtp yet`。因此它的 DSpark 运行路径要求主模型和 support 模型同时使用 graph/backend 的驻留方案，不能直接作为本机 16GB 方案。

我们自己的实现如果要在这台机器上真正获益，必须同时具备：

1. 主模型和 support 模型都可 SSD 流式读取；
2. support 三阶段的 5-token block 推理；
3. 主模型对 5 个候选 token 的批量 target verification；
4. IQ2/Q2 routed expert 的 Metal 或批量 CPU kernel。

只按当前 scalar `gen_forward` 逐 token 验证，虽然可以实现正确性实验，但不会带来 DSpark 加速，反而会增加 support 推理成本。因此在批量 verifier 完成前，DSpark 只做格式校验，不接入默认生成路径。
