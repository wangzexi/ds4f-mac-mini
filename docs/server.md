# 独立聊天服务器

`ds4f-server` 直接由仓库内 `runtime/` 源码构建，不需要
`reference-ds4`、`q4-speed-ds4`、Git 子模块或构建时源码复制。当前只面向
Apple M4 Mac mini 16GB 和固定的 DeepSeek V4 Flash 0731 Q4-trunk 模型。

## 构建与启动

模型文件固定放在 `models/`：

```text
models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf
models/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin
```

构建并启动：

```sh
make server
scripts/run-server.sh
```

默认监听 `0.0.0.0:8000`，使用 32K context、600 个专家缓存槽；固定的
`deepseek-v4-flash` 模型 ID 默认直接回答，不进入 thinking。服务保留一个常驻
session，并启用精确 CPU router、packed expert sidecar 和 Metal
attention/FFN workspace 复用。可通过以下变量覆盖：

```text
DS4F_SERVER_HOST
DS4F_SERVER_PORT
DS4F_SERVER_CONTEXT
DS4F_SERVER_TOKENS
DS4F_SERVER_CACHE_EXPERTS
```

服务器不提供身份验证；只应通过可信局域网或 Tailscale 暴露，不要直接映射到
公网。

## API

同一个进程提供：

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/responses
POST /v1/completions
POST /v1/messages
```

OpenAI Chat Completions：

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":32,"stream":true}'
```

Anthropic Messages：

```sh
curl http://127.0.0.1:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: local' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":32,"stream":true}'
```

## Open WebUI

在 Open WebUI 的 OpenAI-compatible connection 中填写：

```text
Base URL: http://<Mini 的局域网或 Tailscale 地址>:8000/v1
API Key: local
Model ID: deepseek-v4-flash
```

Open WebUI 每轮发送完整消息历史；server 使用公共 token 前缀继续已有 KV
session。当前 16GB 配置只保留一个常驻 session，因此并发请求会排队，切换到
另一段不共享前缀的会话时需要重新 prefill。
