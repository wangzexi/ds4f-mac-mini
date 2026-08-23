# ds4f-mac-mini

面向 Apple Silicon 统一内存设备的 DeepSeek V4 Flash 0731 专用推理运行时；支持
16 GiB 及以上统一内存。

## 演示

![demo](docs/demo.mp4)

## 当前目标固定为：

- 单会话；
- 最大 32K 上下文；
- 精确 CPU Router Top-6；
- Metal 推理；
- SSD 流式专家读取；
- 独立的 Prefill 和 Decode 路径；
- Q4 主干、IQ2/Q2 路由专家。

## 快速开始

所有操作流程都写在 Skill 中：

- [ds4f-mac-mini-ops](agents/skills/ds4f-mac-mini-ops/SKILL.md)：模型初始化、下载量化、本地构建、启动和回归测试。

构建运行时：

```sh
make
```

启动本地服务时，通过环境变量指定模型：

```sh
DS4F_SERVER_MODEL=/path/to/DeepSeek-V4-Flash-0731-Q4.gguf \
./agents/skills/ds4f-mac-mini-ops/scripts/ds4f-cli server
```

模型构建、磁盘要求和断点续传说明见
[ds4f-mac-mini-ops Skill](agents/skills/ds4f-mac-mini-ops/SKILL.md)。

## 当前模型

模型文件示例：

```text
models/DeepSeek-V4-Flash-0731-Q4.gguf
```

当前部署模型是有损量化版本，不保证与 Q8 模型数值等价。

Metal 的 SSD 流式路径只在启动时用 `pread` 读取 GGUF 元数据；模型权重由运行时按
调度计划显式 `pread` 到 Metal buffer，不使用权重 payload 的 `mmap`、缺页读取或系统
预读。推荐始终通过 Skill 启动，它会自动带上 `--ssd-streaming`。

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

默认监听 `0.0.0.0:8000`。服务没有身份验证，公开部署前请自行配置网络隔离。

终端连续对话可直接运行
`python3 agents/skills/ds4f-mac-mini-ops/scripts/ds4f-chat.py --base-url http://mini:8000`。
每轮结束会显示 server 实测的 Prefill 与 Decode 两段 token/s；Prefill 只统计本轮未命中
KV、实际新增计算的 token。

## 目录

```text
agents/skills/  模型初始化、量化器和服务流程
src/            固定运行器与 Metal runtime/server
models/         本地模型权重，已被 Git 忽略
```
