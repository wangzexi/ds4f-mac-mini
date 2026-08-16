# ds4f-mini

面向 Apple M4 Mac mini 16 GiB 统一内存的 DeepSeek V4 Flash 0731 专用推理运行时。

当前目标固定为：

- 单会话；
- 最大 32K 上下文；
- 精确 CPU Router Top-6；
- Metal 推理；
- SSD 流式专家读取；
- 独立的 Prefill 和 Decode 路径；
- Q4 主干、IQ2/Q2 路由专家。

本项目不追求通用硬件兼容，也不支持 Ollama、DSpark/MTP、CUDA/ROCm 或多模型自动调参。

## 快速开始

所有操作流程都写在 Skill 中：

- [ds4f-mini-ops](agents/skills/ds4f-mini-ops/SKILL.md)：模型初始化、下载量化、启动、停止、重启、状态检查和回归测试。

在开发 Mac 上操作 Mini：

```sh
./agents/skills/ds4f-mini-ops/scripts/mini status
./agents/skills/ds4f-mini-ops/scripts/mini start
```

明确需要重启时：

```sh
./agents/skills/ds4f-mini-ops/scripts/mini restart
```

模型构建、磁盘要求和断点续传说明见
[ds4f-mini-ops Skill](agents/skills/ds4f-mini-ops/SKILL.md)。

## 当前模型

Mini 上的模型文件为：

```text
models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf
models/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin
```

当前部署模型是有损量化版本，不保证与 Q8 模型数值等价。

## 本地构建

```sh
make
```

构建结果：

- `ds4f-q4-speed`：固定运行器；
- `ds4f-server`：HTTP server。

## API

server 提供以下兼容接口：

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
POST /v1/responses
POST /v1/messages
```

默认监听 Mini 的 `8000` 端口。服务没有身份验证，只应在可信局域网或 Tailscale
网络中使用。

## 目录

```text
agents/skills/  可执行的模型和服务流程
src/runtime/    Metal 运行时和 HTTP server
src/            固定运行器适配层
agents/skills/  量化器源码、模型布局、专家打包和服务流程
models/         本地模型权重，已被 Git 忽略
```
