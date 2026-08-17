#!/usr/bin/env python3
"""Small standard-library terminal client for the local OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat client for ds4f-server"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="OpenAI-compatible server base URL",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="wait for the complete response instead of printing it incrementally",
    )
    parser.add_argument("--api-key", default="", help="optional Bearer token")
    parser.add_argument("--prompt", help="send one prompt and exit")
    return parser.parse_args()


def request_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def error_text(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        body = exc.read().decode("utf-8", errors="replace").strip()
        return f"HTTP {exc.code}: {body or exc.reason}"
    if isinstance(exc, URLError):
        return f"连接失败: {exc.reason}"
    return str(exc)


class Spinner:
    """Show a small typing indicator until the first response text arrives."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        frames = (".  ", ".. ", "...")
        index = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r模型> {frames[index % len(frames)]}")
            sys.stdout.flush()
            index += 1
            self._stop.wait(0.35)

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def chat_once(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    stream: bool,
    api_key: str,
    on_first_text: Callable[[], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if stream:
        # ds4f-server emits a final usage-only SSE event for this option.
        payload["stream_options"] = {"include_usage": True}
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers(api_key),
        method="POST",
    )

    started = time.monotonic()
    if not stream:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        choices = result.get("choices") or []
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = result.get("usage") or {}
        usage["elapsed_seconds"] = time.monotonic() - started
        return content, usage

    parts: list[str] = []
    usage: dict[str, Any] = {}
    first_text_at: float | None = None
    with urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage.update(event["usage"])
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content") or ""
            if text:
                parts.append(text)
                if first_text_at is None:
                    first_text_at = time.monotonic()
                    if on_first_text is not None:
                        on_first_text()
                sys.stdout.write(text)
                sys.stdout.flush()
    usage["elapsed_seconds"] = time.monotonic() - started
    if first_text_at is not None:
        usage["first_token_seconds"] = first_text_at - started
    return "".join(parts), usage


def print_stats(usage: dict[str, Any]) -> None:
    tokens = usage.get("completion_tokens")
    elapsed = usage.get("elapsed_seconds")
    if not isinstance(tokens, int) or not isinstance(elapsed, (float, int)):
        return
    rate = tokens / elapsed if elapsed > 0 else 0.0
    first = usage.get("first_token_seconds")
    if isinstance(first, (float, int)) and tokens > 1 and elapsed > first:
        decode_rate = (tokens - 1) / (elapsed - first)
        print(
            f"\n[完成 {tokens} tokens, 总耗时 {elapsed:.2f}s, "
            f"请求平均 {rate:.2f} token/s, 首 token 后解码估算 {decode_rate:.2f} token/s]"
        )
    else:
        print(f"\n[完成 {tokens} tokens, {elapsed:.2f}s, {rate:.2f} token/s]")


def print_help() -> None:
    print("/new             清空当前对话")
    print("/model NAME      切换模型名")
    print("/max_tokens N    设置本轮最大输出 token 数")
    print("/stream on|off   开关流式输出")
    print("/history         显示当前消息数量")
    print("/quit            退出")
    print("本客户端不添加默认 system prompt。")


def configure_readline() -> None:
    """Keep Backspace and the macOS Delete key working at end-of-line."""
    try:
        import readline
    except ImportError:
        return
    if "libedit" in (readline.__doc__ or "").lower():
        bindings = (
            r'bind -e',
            r'bind "\e[3~" ed-delete-next-char',
            r'bind "^?" ed-delete-prev-char',
            r'bind "^H" ed-delete-prev-char',
        )
    else:
        bindings = (
            r'"\e[3~": delete-char',
            r'"\C-?": backward-delete-char',
            r'"\C-h": backward-delete-char',
        )
    for binding in bindings:
        try:
            readline.parse_and_bind(binding)
        except (ValueError, RuntimeError):
            pass


def run(args: argparse.Namespace) -> int:
    configure_readline()
    messages: list[dict[str, str]] = []
    stream = not args.no_stream

    def send(prompt: str) -> None:
        messages.append({"role": "user", "content": prompt})
        spinner = Spinner()

        def on_first_text() -> None:
            spinner.stop()
            sys.stdout.write("模型> ")
            sys.stdout.flush()

        spinner.start()
        try:
            content, usage = chat_once(
                args.base_url,
                args.model,
                messages,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.timeout,
                stream,
                args.api_key,
                on_first_text if stream else None,
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            messages.pop()
            spinner.stop()
            print(f"\n错误: {error_text(exc)}", file=sys.stderr)
            return
        finally:
            # If the server returns no text, the first-text callback never
            # stops the indicator.
            spinner.stop()
        if not stream:
            print(f"模型> {content}", end="")
        print()
        messages.append({"role": "assistant", "content": content})
        print_stats(usage)

    if args.prompt is not None:
        send(args.prompt)
        return 0

    print(f"连接: {args.base_url}  模型: {args.model}")
    print("输入 /help 查看命令；没有默认 system prompt。")
    while True:
        try:
            prompt = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        command, _, argument = prompt.partition(" ")
        if command in {"/quit", "/exit", "/q"}:
            return 0
        if command == "/help":
            print_help()
            continue
        if command == "/new":
            messages.clear()
            print("已开始新对话。")
            continue
        if command == "/history":
            print(f"当前对话消息: {len(messages)}")
            continue
        if command == "/model" and argument.strip():
            args.model = argument.strip()
            print(f"模型已切换为: {args.model}")
            continue
        if command == "/max_tokens" and argument.strip().isdigit():
            args.max_tokens = max(1, int(argument.strip()))
            print(f"max_tokens={args.max_tokens}")
            continue
        if command == "/stream" and argument.strip() in {"on", "off"}:
            stream = argument.strip() == "on"
            print(f"stream={'on' if stream else 'off'}")
            continue
        if command.startswith("/"):
            print("未知命令，输入 /help 查看帮助。")
            continue
        send(prompt)


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
