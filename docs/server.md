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

默认监听 `0.0.0.0:8000`，使用 32K context、最多 1000 个专家缓存槽；固定的
`deepseek-v4-flash` 模型 ID 默认直接回答，不进入 thinking。服务保留一个常驻
session，并启用精确 CPU router、packed expert sidecar 和 Metal
attention/FFN workspace 复用。可通过以下变量覆盖：

```text
DS4F_SERVER_HOST
DS4F_SERVER_PORT
DS4F_SERVER_CONTEXT
DS4F_SERVER_TOKENS
DS4F_SERVER_CACHE_EXPERTS
DS4F_SERVER_PREFILL_FULL_LAYER_PARALLEL_PREAD
DS4F_SERVER_KEEP_HASH_LAYER0
DS4F_SERVER_PREFILL_RELEASE_HASH_LAYER0_AFTER
DS4F_SERVER_PREAD_THREADS
DS4F_SERVER_DECODE_SPLIT_MIN_MISSES
DS4F_SERVER_WORKING_SET_MIB
DS4F_SERVER_PINNED_MIB
DS4F_SERVER_DECODE_PINNED_MIB
DS4F_SERVER_MEMORY_RESERVE_MIB
DS4F_SERVER_KV_CACHE_DIR
DS4F_SERVER_KV_CACHE_MIB
DS4F_SERVER_KV_CACHE_MIN_TOKENS
```

`DS4F_SERVER_KEEP_HASH_LAYER0` 默认是 `1`，保留 canonical 的 L0 预载与
Learned Router 数值路径。设为 `0` 的实验可以让后续 learned layer 连续整层预载，
在 M4 Mini 的 1K Prefill 上从约 185 秒降至约 96–98 秒；但已测得它从 L3 的
第一个 token 行开始改变 router 排序，因此在定位并消除该数值分歧前不得作为生产路径。

生产路径保留 L0 直到 `DS4F_SERVER_PREFILL_RELEASE_HASH_LAYER0_AFTER=2`：三个
hash-routed 层已经完整运行，随后在进入 learned stack 前归还 L0 的 256 个专家槽。
512-token 精确测试的 22,231 条 Prefill 选路、两轮 32-token greedy 轨迹和输出哈希
都与全程保留 L0 一致；Prefill 从约 89 秒降至 49–52 秒，Decode 从约 1.89 提升到
2.43–2.46 token/s。

Decode 保持全局 `lru`。在每层同时有命中和 miss 时，默认至少有 **2** 个 miss 才
拆开提交：先让 GPU 计算命中专家，同时读取其余专家。L2-after-L0 的 512-token 前缀、
32-token 输出交错三次 A/B 中，阈值 2 的中位数为 2.42 token/s；阈值 1 为 2.40，
阈值 3–6 为 2.38–2.41。所有配置的 greedy token trace 相同。

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
- 固定模型常驻页、按计划 context 计算的 KV、单行固定 graph 和 512MiB 系统
  安全余量先扣除；
- 已删除原来的 `graph_bytes * 4` 近似。M4实测在18行及以下走小路径，新增物理
  驻留最高约44.3MiB，规划取48MiB；从19行开始进入完整batch路径，直接读取
  本session的Metal owner buffer总量精确计价，当前固定值为3122479104字节
  （2.91GiB），不再随逻辑graph字节乘倍数；
- 剩余的 11.5GiB 工作集预算决定总专家槽数，最多1000槽。冷启动 prefill 先锁
  4GiB（约606槽），草稿清空后 decode 扩到6GiB并补锁约910槽，因此900个热点
  专家在系统wired余量充足时可以全部进入一级缓存，另约90槽作为可分页二级缓存；
  余量不足时则保留内核已确认锁定的槽，其余槽仍可驻留但允许分页。后续短 prefill
  不再机械地解锁910→606。当前固定配置即使在4093行prefill时，base也只有
  4.05GiB，仍保留全部1000槽；只有未来实际owner/KV超过预算时才会精确缩容。

总槽数和锁页槽数已经解耦。macOS 拒绝继续 `mlock` 时只缩小 wired tier，不能再
把整个专家缓存从 1000 槽误降到 455 槽。planner-driven 缩容会逐槽 `munlock`，
并在释放旧 Metal slab 前设置 `MTLPurgeableStateEmpty`，避免驱动缓存旧物理后备、
再与新 slab 和 prefill workspace 叠加。

预填充结束后，约 2.91GiB 的 batch-only Metal workspace 会先通过 macOS
`MADV_FREE_REUSABLE` 标为可回收，再把对应 Metal resource 设置为
`MTLPurgeableStateEmpty`，立即丢弃物理后备但保留 tensor 地址；下次 prefill 前
恢复为 `MTLPurgeableStateNonVolatile`。释放完成后，server 会把已有的可分页专家槽
补锁进解码 wired tier；反向进入下一轮 prefill 时则按预算逐槽 `munlock`。

默认预算为11.5GiB动态工作集、冷预填充4GiB/解码6GiB专家锁页、512MiB余量和1000总槽。这不是简单
追求最大的 footprint；实机存在明确的 VM 性能悬崖：

| 总槽 | 短提示 decode | `phys_footprint` | 结果 |
|---:|---:|---:|---|
| 600 | 约 2.32 t/s | 约 4.8GB | 原始稳定基线 |
| 900 | 2.33–2.58 t/s | 约 6.8GB | 最小高性能点；全锁受启动时系统 wired 压力影响 |
| 1000 | 2.19–2.52 t/s；热态实测2.38–2.49 | 约7.5GB | 当前生产点；实测wired tier为606–910槽 |
| 1050 | 约 1.97 t/s | 约 7.81GB | 开始 VM 抖动 |
| 1200 | purge前约0.31、purge后约1.23 t/s | 约8.8GB | 仍明显慢于1000槽 |

这里的11.5GiB是动态安全上限，不是要求 `phys_footprint` 永久显示11.5GiB。
decode 时被清空的2.91GiB草稿不应为了数字好看重新占满；未归属进程的余量会由
macOS文件缓存吸收专家sidecar读取，并保留系统要求的不可用户锁页空间。实测1000槽
时进程footprint约7.5GB，而整机空闲页已接近零，说明统一内存仍被有效利用。

旧的×4近似曾让605-token prompt只保留844槽、1005-token只保留682槽；实机逐点
测量覆盖14、18、19、23、31、32、33、39、71、135、256、512、1025、2048和
4095行后确认这是误判。新规划在4093行仍保留1000槽。设置
`DS4_SERVER_MEMORY_CALIBRATION=1` 可在日志记录每轮逻辑graph、owner字节以及prefill
前/峰值/清理后的 `phys_footprint`，用于同一固定设备重新标定。`mlock`仍受整机全局
wired压力约束，失败时只保留已经被内核确认锁定的一级缓存，不影响额外可分页槽。
修改 `*_WORKING_SET_MIB`、`*_PINNED_MIB` 或总槽上限后必须同时检查速度、
`phys_footprint`、memory pressure 和日志，不能只以占用更大作为优化成功。

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

### Decode 专家读取并发

2026-08-13 的实际单会话 server 基准（每档冷启动两次、L0 和静态 Decode 主干均预热、
`你好` → 32 个 exact greedy token）证实，PREAD 读者数是当前 Decode 的直接瓶颈之一：

| PREAD readers | Decode median | 32-token Decode median |
|---:|---:|---:|
| 1 | 1.50 t/s | 21.44 s |
| 2 | 1.80 t/s | 17.85 s |
| 3 | 1.82 t/s | 17.69 s |
| 4 | 1.91 t/s | 16.80 s |
| 6 | **1.94 t/s** | **16.53 s** |

所有十次运行的 token-ID trace 和响应字节完全一致。每一层 top-6 最多发出六个
连续 packed-expert `pread`，所以 6 是这个固定模型的并行上限；生产启动器固定使用
6 个 persistent reader（可用 `DS4F_SERVER_PREAD_THREADS` 仅用于复现实验）。
1→6 提升约 29%，但 4→6 只约 1.3%，说明下一阶段不应继续加 I/O 线程，而应提高
真实 expert cache 命中并减少约 24GiB/32-token 的 miss 读取量。

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
### KV-bound route heat

Each persisted Flash KV prefix carries a small route-heat trailer. It stores
the accumulated real Router weight for every layer/expert pair plus the decay
token clock. Heat decays continuously with a 16-token half-life; batched
Prefill weights older rows less than recent rows. A disk KV hit restores the
matching heat atomically with the graph payload. Loading an older checkpoint
without this trailer resets heat instead of reusing state from another prompt.
The 43 x 256 float32 matrix costs 44,032 bytes per checkpoint.
