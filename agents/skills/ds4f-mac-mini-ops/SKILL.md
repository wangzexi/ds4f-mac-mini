---
name: ds4f-mac-mini-ops
description: 在本地初始化 DeepSeek-V4-Flash-0731 模型、构建运行时并启动服务
---

## 模型初始化

这是本工程唯一的模型下载和转换流程。它只从官方 safetensors 快照开始，不下载或
依赖第三方完整量化权重。

前置条件：

- Apple Silicon macOS 和 Xcode Command Line Tools（提供 `cc`、Metal framework）；
- 统一内存至少 16 GiB；16 GiB 的 M4 Mac mini 是当前实机验证基线，启动脚本会依据
  Metal 推荐工作集自动为更大内存配置更大的专家缓存；
- 其它 Apple Silicon 芯片理论上走同一 Metal 路径，但尚未逐型号验证性能；Intel、CUDA
  和 ROCm 不属于本项目支持范围；
- Python 3.10 或更高版本；
- 能访问 `huggingface.co` 和 `raw.githubusercontent.com` 的网络；如果 Hugging Face
  要求认证，先设置 `HF_TOKEN`；
- 已准备外部 DS4 `gguf-tools/quants.c` 量化后端；
- 建议准备约 100 GB 的构建盘空间；最终 GGUF 约 77 GB，剩余空间用于当前 shard、
  checkpoint 和安全余量；
- `agents/skills/ds4f-mac-mini-ops/assets/deepseek-v4-flash-0731.layout.gguf` 存在；
- 不要把模型权重、源 shard 或中间状态提交到 Git。

准备量化后端（只取构建所需文件）。为避免上游接口漂移，固定使用
`antirez/ds4` 当前已验证的提交 `84cc882352757baf628a1776badf7cc54d584e28`。
macOS 自带 `curl` 和 `shasum`，直接下载这两个源码文件：

```sh
mkdir -p reference-ds4/gguf-tools
curl -fL --retry 3 \
  -o reference-ds4/gguf-tools/quants.c \
  https://raw.githubusercontent.com/antirez/ds4/84cc882352757baf628a1776badf7cc54d584e28/gguf-tools/quants.c
curl -fL --retry 3 \
  -o reference-ds4/gguf-tools/quants.h \
  https://raw.githubusercontent.com/antirez/ds4/84cc882352757baf628a1776badf7cc54d584e28/gguf-tools/quants.h
shasum -a 256 reference-ds4/gguf-tools/quants.c reference-ds4/gguf-tools/quants.h
# expected:
# 80d52bcafcf4c4f8b53d91612267cd04429e628e5b28d5d845619a89e96e8b70  reference-ds4/gguf-tools/quants.c
# 4dfcb9889c95ba3c8f4946d92a7d75a91620a6905e30a66f65bad2c5b3c1d2e6  reference-ds4/gguf-tools/quants.h
```

构建量化器：

```sh
make quantizer QUANTS_DIR=reference-ds4/gguf-tools
```

下载并流式量化：

```sh
python3 agents/skills/ds4f-mac-mini-ops/scripts/build_official_q4_stream.py \
  --quantizer build/deepseek4-quantize \
  --staging /path/with/free-space/ds4f-official-staging \
  --out /path/with/free-space/DeepSeek-V4-Flash-0731-Q4.gguf
```

构建器默认固定官方快照
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`。它一次只下载当前 shard 组，把量化结果
直接写入最终 GGUF 偏移，完成检查点后删除源 shard。构建中断后直接重新运行同一命令，
使用 `*.stream-state.json` 续传。

磁盘峰值大致是：最终 GGUF + 当前正在处理的一个官方 shard 组 + 6 GiB 安全余量，
目标控制在 100 GB 内。最终部署只需要这个 GGUF 文件。

## 本地构建与启动

在仓库根目录构建运行器和 server：

```sh
make
```

用环境变量指定模型，然后在前台启动本地服务：

```sh
DS4F_SERVER_MODEL=/path/to/DeepSeek-V4-Flash-0731-Q4.gguf \
./agents/skills/ds4f-mac-mini-ops/scripts/ds4f-cli server
```

默认按 Metal 设备的推荐工作集自动规划内存，额外内存会用于扩大专家缓存；需要手工
限制预算时，可在启动前显式设置 `DS4F_SERVER_WORKING_SET_MIB` 和
`DS4F_SERVER_CACHE_EXPERTS`。

Skill 启动命令固定启用 `--ssd-streaming`：Metal 主路径只保留 GGUF 元数据，所有模型
权重由运行时按计划显式 `pread` 到 Metal buffer；不依赖权重文件 `mmap`、缺页读取或
系统预读。未启用 SSD 流式的底层兼容入口仍保留旧映射路径，不属于本项目推荐运行方式。

另一个终端检查服务：

```sh
curl -fsS http://127.0.0.1:8000/v1/models
```

在本机终端进行连续对话（只使用 Python 标准库，不注入默认 system prompt）：

```sh
python3 agents/skills/ds4f-mac-mini-ops/scripts/ds4f-chat.py \
  --base-url http://127.0.0.1:8000
```

如果 server 在局域网里的 Mini 上运行，把地址改成 Mini 的地址：

```sh
python3 agents/skills/ds4f-mac-mini-ops/scripts/ds4f-chat.py \
  --base-url http://mini:8000
```

支持 `/new` 开始新对话、`/stream on|off` 切换流式输出、`/max_tokens N` 调整输出上限，
`/quit` 退出。也可以用 `--prompt "你好"` 做一次性请求。

运行一次固定 runner：

```sh
DS4F_FAST_MODEL=/path/with/free-space/DeepSeek-V4-Flash-0731-Q4.gguf \
./agents/skills/ds4f-mac-mini-ops/scripts/ds4f-cli run "你好" 32
```

执行固定贪心回归：

```sh
./agents/skills/ds4f-mac-mini-ops/scripts/ds4f-cli regression /path/to/DeepSeek-V4-Flash-0731-Q4.gguf
```

服务在前台运行，使用 `Ctrl-C` 停止。需要后台运行时，由调用方自行使用标准的
进程管理方式；Skill 不假设特定操作系统服务管理器。

## 规则

- 只运行固定的 DeepSeek-V4-Flash-0731 server，不启动 Ollama、DSpark/MTP
  或其他模型服务。
- 不在没有确认端口占用的情况下启动第二个 server。
- 不删除 `models/`、`cache/` 或 `/tmp/ds4f-mac-mini/kv/`。
- 服务启动、停止和日志查看由调用方的本地进程管理方式负责。
- 只使用 `deepseek-ai/DeepSeek-V4-Flash-0731` 官方来源；不执行 Q2 主干、模板转换
  或 DwarfStar 依赖流程。

## 验证

完成启动后必须确认：

```sh
curl -fsS http://127.0.0.1:8000/v1/models
```

命令应返回固定的 `deepseek-v4-flash` 模型信息。
