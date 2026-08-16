---
name: ds4f-mini-ops
description: 管理 DeepSeek-V4-Flash-0731 模型初始化以及远程 M4 Mini server 的启动、检查、重启和停止
---

# ds4f-mini 运维

这是本工程唯一的模型和生产服务操作入口。默认目标是 SSH 别名 `mini` 上的
`/Users/zexi/workspace/ds4f-mini`。

## 模型初始化

这是本工程唯一的模型下载和转换流程。它只从官方 safetensors 快照开始，不下载或
依赖第三方完整量化权重。

前置条件：

- 已准备外部 DS4 `gguf-tools/quants.c` 量化后端；
- 已准备至少 170 GB 的构建磁盘空间；
- `agents/skills/ds4f-mini-ops/assets/deepseek-v4-flash-0731.layout.gguf` 存在；
- 不要把模型权重、源 shard 或中间状态提交到 Git。

构建量化器：

```sh
make quantizer QUANTS_DIR=/path/to/ds4/gguf-tools
```

下载并流式量化：

```sh
python3 agents/skills/ds4f-mini-ops/scripts/build_official_q4_stream.py \
  --quantizer build/deepseek4-quantize \
  --staging /path/with/free-space/ds4f-official-staging \
  --out /path/with/free-space/DeepSeek-V4-Flash-0731-Mini-Q4.gguf
```

构建器默认固定官方快照
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`。它一次只下载当前 shard 组，把量化结果
直接写入最终 GGUF 偏移，完成检查点后删除源 shard，最后生成专家包。构建中断后
直接重新运行同一命令，使用 `*.stream-state.json` 续传。

## 启动流程

在仓库根目录执行 Skill 内的运维脚本：

```sh
./agents/skills/ds4f-mini-ops/scripts/mini status
```

先看 API 是否已经健康：

- 如果显示 `API` 可用，不要再次启动第二个 server。
- 如果 server 已停止，执行：

```sh
./agents/skills/ds4f-mini-ops/scripts/mini start
```

启动后再次执行 `./agents/skills/ds4f-mini-ops/scripts/mini status`，确认进程存在并且
`/v1/models` 可访问。

## 其他操作

```sh
./agents/skills/ds4f-mini-ops/scripts/mini restart   # 明确需要重启时使用
./agents/skills/ds4f-mini-ops/scripts/mini stop      # 停止生产服务
./agents/skills/ds4f-mini-ops/scripts/mini logs      # 查看生产日志
./agents/skills/ds4f-mini-ops/scripts/mini shell     # 进入 Mini 上的工程目录
```

## 规则

- 只运行固定的 DeepSeek-V4-Flash-0731 server，不启动 Ollama、DSpark/MTP
  或其他模型服务。
- 不直接执行第二条 `ds4f-server` 命令；先通过
  `./agents/skills/ds4f-mini-ops/scripts/mini status` 检查已有进程。
- 不删除 `models/`、`cache/`、`/tmp/ds4f-mini/kv/` 或专家包。
- 只有明确要求重启、停止或部署时，才执行对应的有状态操作。
- 远程 server 的底层启动脚本是
  `agents/skills/ds4f-mini-ops/scripts/run-server.sh`；日常操作优先使用 Skill 内的
  `mini` 脚本，避免绕过进程检查和健康检查。
- 只使用 `deepseek-ai/DeepSeek-V4-Flash-0731` 官方来源；不执行 Q2 主干、模板转换
  或 DwarfStar 依赖流程。

## 验证

完成启动或重启后必须确认：

```sh
./agents/skills/ds4f-mini-ops/scripts/mini status
```

输出中应同时有 `server: running` 和成功的 `/v1/models` 响应。
