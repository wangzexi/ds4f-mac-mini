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

默认监听 `0.0.0.0:8000`，使用 32K context、最多 900 个专家缓存槽；固定的
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
DS4F_SERVER_KV_CACHE_DIR
DS4F_SERVER_KV_CACHE_MIB
DS4F_SERVER_KV_CACHE_MIN_TOKENS
```

默认还会在 `cache/kv/` 保留最多 10240MiB（10GiB）的磁盘 KV 前缀缓存。
每次成功 response 已经发送给客户端后，server 同步保存当前完整 token frontier
和对应 KV；下次请求会在磁盘条目中选择最长的文本前缀命中，只恢复该快照并
prefill 新增后缀。生产配置把最小保存长度设为 1 token，短对话也会持久化。
旧的冷 prompt 锚点和每 10K token 中间检查点在生产启动中关闭，因此正常路径
每轮只新增 response-end 快照；会话被替换或服务退出时仍会确保当前状态已保存，
相同 key 会直接复用已有文件而不重写 KV payload。

磁盘上限采用严格 LRU：缓存命中会更新 `last_used`；写入新快照导致总量超过
10GiB 时，持续删除最久未使用的文件直到重新低于上限。保存发生在网络响应
结束之后，但会占用单 session worker，因此紧接着到来的下一轮请求需要等待
本轮 SSD 写入完成。

## 按请求分配内存

服务启动时只建立模型 mmap、Metal pipeline 和 32K session 的虚拟 buffer；
模型的 77GiB 映射不会因此全部进入物理内存，专家 cache slab 也要等第一个请求
实际路由到专家后才创建。当前只在单驻留 session 下启用按请求内存规划，多 session
模式会自动关闭该功能，避免全局专家缓存被不同请求互相缩放。

每次请求在聊天模板渲染、分词和 KV 前缀匹配之后，使用以下真实数字制定本轮方案：

- `uncached = prompt_tokens - cached_tokens`，决定本轮真正触碰多少 prefill 行；
- `output = min(max_tokens, 32768 - prompt_tokens)`，决定本轮规划 context；
- 固定模型常驻页、预计 KV、精确 prefill graph 行数和 512MiB 安全余量先扣除；
- M4 实测 Metal prefill 物理驻留约为静态 graph 估计的 4 倍，因此 prefill
  规划按 `graph_bytes * 4` 计价，decode 仍按精确的单行 graph 计价；
- 剩余的 8GiB 工作集预算决定总专家槽数，最多 900 槽；4GiB 锁页预算只决定
  一级 wired cache（当前为 606 槽），额外槽构成可分页的二级缓存。

总槽数和锁页槽数已经解耦。macOS 拒绝继续 `mlock` 时只缩小 wired tier，不能再
把整个专家缓存从 1000 槽误降到 455 槽。planner-driven 缩容会逐槽 `munlock`，
并在释放旧 Metal slab 前设置 `MTLPurgeableStateEmpty`，避免驱动缓存旧物理后备、
再与新 slab 和 prefill workspace 叠加。

预填充结束后，约 2.91GiB 的 batch-only Metal workspace 会通过 macOS
`MADV_FREE_REUSABLE` 标为可回收；tensor 地址和 KV 不变，下次 prefill 前再用
`MADV_FREE_REUSE` 重新取得。因此 decode 可以继续使用专家缓存，而不会把上一轮
预填充草稿永久作为不可回收内存保留。

默认预算为 8GiB 动态工作集、4GiB专家锁页、512MiB余量和900总槽。这不是简单
追求最大的 footprint；实机存在明确的 VM 性能悬崖：

| 总槽 | 短提示 decode | `phys_footprint` | 结果 |
|---:|---:|---:|---|
| 600 | 约 2.32 t/s | 约 4.8GB | 原始稳定基线 |
| 900 | 2.54–2.58 t/s | 约 6.8GB | 当前最优稳定点 |
| 1000 | 约 2.52 t/s | 约 7.47GB | 无额外收益 |
| 1050 | 约 1.97 t/s | 约 7.81GB | 开始 VM 抖动 |
| 1200 | 约 0.31 t/s | 约 8.82GB | 不可用 |

阶段性规划实测：605-token prompt 使用844槽，prefill约36.2秒（16.7 t/s）；
1005-token prompt 使用682槽，prefill约98.2秒（10.2 t/s）。连续缩容/扩容没有
`mlock` 失败。修改 `*_WORKING_SET_MIB`、`*_PINNED_MIB` 或总槽上限后必须同时检查
速度、`phys_footprint`、memory pressure 和日志，不能只以占用更大作为优化成功。

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
