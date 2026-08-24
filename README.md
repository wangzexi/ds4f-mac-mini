# ds4f-mac-mini

面向 Apple Silicon 统一内存设备的 DeepSeek V4 Flash 0731 专用推理运行时；
16 GiB 统一内存，M4 实测解码约 2.4 tok/s。

## 演示

https://github.com/user-attachments/assets/a43fbd4c-8c4c-4423-a5b9-76777de05b47

## 特性

- 单会话，最大 32K 上下文
- 精确 CPU Router Top-6，Metal 推理
- SSD 流式专家读取：权重按调度计划显式 `pread` 到 Metal buffer，
  不使用 `mmap`、缺页读取或系统预读
- 独立的 Prefill 与 Decode 路径
- Q4 主干、IQ2/Q2 路由专家（有损量化，不保证与 Q8 数值等价）

## 快速开始

```sh
make
DS4F_SERVER_MODEL=/path/to/DeepSeek-V4-Flash-0731-Q4.gguf \
./.agents/skills/ds4f-mac-mini-ops/scripts/ds4f-cli server
```

构建产物：`ds4f-server`（HTTP server）与 `ds4f-q4-speed`（固定运行器）。
模型下载量化、磁盘要求与断点续传见
[ds4f-mac-mini-ops Skill](.agents/skills/ds4f-mac-mini-ops/SKILL.md)。

终端连续对话（每轮结束显示 server 实测的 Prefill/Decode token/s，Prefill 只
统计本轮未命中 KV 的新增 token）：

```sh
python3 .agents/skills/ds4f-mac-mini-ops/scripts/ds4f-chat.py --base-url http://mini:8000
```

## API

`GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/completions`、
`POST /v1/responses`、`POST /v1/messages`。

默认监听 `0.0.0.0:8000`，无身份验证，公开部署前请自行配置网络隔离。

## 目录

```text
.agents/skills/  模型初始化、量化器和服务流程
src/            固定运行器与 Metal runtime/server
models/         本地模型权重，已被 Git 忽略
```
