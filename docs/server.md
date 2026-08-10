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
DS4F_SERVER_WORKING_SET_MIB
DS4F_SERVER_PINNED_MIB
DS4F_SERVER_MEMORY_RESERVE_MIB
```

## 按请求分配内存

服务启动时只建立模型 mmap、Metal pipeline 和 32K session 的虚拟 buffer；
模型的 77GiB 映射不会因此全部进入物理内存，专家 cache slab 也要等第一个请求
实际路由到专家后才创建。当前只在单驻留 session 下启用按请求内存规划，多 session
模式会自动关闭该功能，避免全局专家缓存被不同请求互相缩放。

每次请求在聊天模板渲染、分词和 KV 前缀匹配之后，使用以下真实数字制定本轮方案：

- `uncached = prompt_tokens - cached_tokens`，决定本轮真正触碰多少 prefill 行；
- `output = min(max_tokens, 32768 - prompt_tokens)`，决定本轮规划 context；
- 固定模型常驻页、预计 KV、精确 prefill graph 行数和 512MiB 安全余量先扣除；
- 剩余工作集与 4GiB 专家锁页预算共同决定专家槽上限，且不超过启动参数的
  600 槽。

预填充结束后，约 2.91GiB 的 batch-only Metal workspace 会通过 macOS
`MADV_FREE_REUSABLE` 标为可回收；tensor 地址和 KV 不变，下次 prefill 前再用
`MADV_FREE_REUSE` 重新取得。因此 decode 可以继续使用专家缓存，而不会把上一轮
预填充草稿永久压在物理内存中。实测 5-token 和 292-token prompt 完成后，server
RSS 均回到约 4.1GiB；600 槽没有出现 `mlock` 降级。

默认预算为 11.5GiB 进程工作集、4GiB 专家锁页和 512MiB 余量。空机单独测试
得到的 10.66GiB `mlock` 极限不能全给专家，因为 Metal/KV 工作区也占用系统
wire budget；曾尝试 1300 槽时只锁到约 4.03GiB，运行时反而降到 455 槽，所以
生产默认仍为 600。可用上面的三个 `*_MIB` 环境变量继续实验，但修改后必须检查
日志中是否出现 `could not mlock all buffers`。

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
