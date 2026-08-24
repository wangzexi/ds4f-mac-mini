#!/usr/bin/env python3
"""Small standard-library terminal client for the local OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL_NAME = "deepseek-v4-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat client for ds4f-server"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="OpenAI-compatible server base URL",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=7200.0)
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


class PrefillStatus:
    """Show elapsed waiting time until the first response text arrives."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = 0.0

    def start(self) -> None:
        self._started = time.monotonic()
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            elapsed = time.monotonic() - self._started
            sys.stdout.write(f"\r[Prefilling {elapsed:.1f}s]")
            sys.stdout.flush()
            self._stop.wait(0.1)

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def chat_once(
    base_url: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    think_enabled: bool,
    api_key: str,
    on_first_text: Callable[[], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "thinking": think_enabled,
    }
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers(api_key),
        method="POST",
    )

    started = time.monotonic()
    parts: list[str] = []
    usage: dict[str, Any] = {}
    first_text_at: float | None = None
    reasoning_started = False
    reasoning_needs_line_break = False
    content_started = False
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
            reasoning = delta.get("reasoning_content") or ""
            text = delta.get("content") or ""
            if reasoning:
                if first_text_at is None:
                    first_text_at = time.monotonic()
                    if on_first_text is not None:
                        on_first_text()
                if not reasoning_started:
                    sys.stdout.write("<think>\n")
                    reasoning_started = True
                sys.stdout.write(reasoning)
                reasoning_needs_line_break = not reasoning.endswith("\n")
                sys.stdout.flush()
            if text:
                parts.append(text)
                if first_text_at is None:
                    first_text_at = time.monotonic()
                    if on_first_text is not None:
                        on_first_text()
                if think_enabled and not content_started:
                    if reasoning_started:
                        if reasoning_needs_line_break:
                            sys.stdout.write("\n")
                        sys.stdout.write("</think>\n\n")
                    else:
                        sys.stdout.write("<think>\n</think>\n\n")
                content_started = True
                sys.stdout.write(text)
                sys.stdout.flush()
    if reasoning_started and not content_started:
        if reasoning_needs_line_break:
            sys.stdout.write("\n")
        sys.stdout.write("</think>\n\n")
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
    timing = usage.get("ds4_timing")
    if isinstance(timing, dict):
        prefill_seconds = timing.get("prefill_seconds")
        prefill_tokens = timing.get("prefill_tokens")
        decode_seconds = timing.get("decode_seconds")
        decode_tokens = timing.get("decode_tokens")
        if (isinstance(prefill_seconds, (float, int)) and prefill_seconds > 0 and
                isinstance(prefill_tokens, int) and
                isinstance(decode_seconds, (float, int)) and decode_seconds > 0 and
                isinstance(decode_tokens, int)):
            prefill_rate = prefill_tokens / prefill_seconds
            decode_rate = decode_tokens / decode_seconds if decode_tokens > 0 else 0.0
            print(f"\n[Prefill {prefill_rate:.2f} token/s, Decode {decode_rate:.2f} token/s]")
            return
    first = usage.get("first_token_seconds")
    if isinstance(first, (float, int)) and tokens > 1 and elapsed > first:
        decode_rate = (tokens - 1) / (elapsed - first)
        print(f"\n[Prefill N/A, Decode {decode_rate:.2f} token/s]")
    else:
        print(f"\n[Prefill N/A, Decode {rate:.2f} token/s]")


def print_help() -> None:
    print("/new             清空当前对话")
    print("/max_tokens N    设置本轮最大输出 token 数")
    print("/think on|off    开关思考模式")
    print("/quit            退出")


def _read_byte(fd: int) -> bytes:
    return os.read(fd, 1)


def _read_escape_sequence(fd: int) -> bytes:
    """Read a short terminal escape sequence after the initial ESC byte."""
    sequence = bytearray()
    known_endings = (b"[A", b"[B", b"[C", b"[D", b"[3~", b"[200~")
    deadline = time.monotonic() + 0.1
    while len(sequence) < 5:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            break
        sequence.extend(_read_byte(fd))
        current = bytes(sequence)
        if current in known_endings:
            break
    return bytes(sequence)


def _read_bracketed_paste(fd: int) -> str:
    """Read one terminal bracketed-paste block, including embedded newlines."""
    end_marker = b"\x1b[201~"
    data = bytearray()
    while True:
        data.extend(_read_byte(fd))
        if data.endswith(end_marker):
            del data[-len(end_marker):]
            break
    return bytes(data).replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode(
        "utf-8", errors="replace"
    )


def read_prompt(prompt: str) -> str | None:
    """Read a line and keep bracketed multi-line pastes as one prompt."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        line = sys.stdin.readline()
        if not line:
            return None
        return line.rstrip("\r\n")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buffer: list[str] = []
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\x1b[?2004h")
        sys.stdout.write(prompt)
        sys.stdout.flush()
        while True:
            byte = _read_byte(fd)
            if byte in {b"\r", b"\n"}:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "".join(buffer)
            if byte == b"\x03":
                raise KeyboardInterrupt
            if byte == b"\x04":
                if not buffer:
                    raise EOFError
                continue
            if byte in {b"\x7f", b"\x08"}:
                if buffer:
                    buffer.pop()
                    sys.stdout.write("\r\033[K" + prompt + "".join(buffer))
                    sys.stdout.flush()
                continue
            if byte == b"\x15":
                buffer.clear()
                sys.stdout.write("\r\033[K" + prompt)
                sys.stdout.flush()
                continue
            if byte == b"\x1b":
                sequence = _read_escape_sequence(fd)
                if sequence == b"[200~":
                    pasted = _read_bracketed_paste(fd)
                    buffer.extend(pasted)
                    sys.stdout.write(pasted)
                    sys.stdout.flush()
                elif sequence == b"[3~" and buffer:
                    buffer.pop()
                    sys.stdout.write("\r\033[K" + prompt + "".join(buffer))
                    sys.stdout.flush()
                continue
            if byte[0] < 0x20:
                continue
            pending = byte
            while True:
                try:
                    character = pending.decode("utf-8")
                    break
                except UnicodeDecodeError:
                    pending += _read_byte(fd)
            buffer.append(character)
            sys.stdout.write(character)
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()


def run(args: argparse.Namespace) -> int:
    messages: list[dict[str, str]] = []
    think_enabled = False

    def send(prompt: str) -> None:
        messages.append({"role": "user", "content": prompt})
        status = PrefillStatus()

        def on_first_text() -> None:
            status.stop()
            sys.stdout.flush()

        status.start()
        try:
            content, usage = chat_once(
                args.base_url,
                messages,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.timeout,
                think_enabled,
                args.api_key,
                on_first_text,
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            messages.pop()
            status.stop()
            print(f"\n错误: {error_text(exc)}", file=sys.stderr)
            return
        finally:
            # If the server returns no text, the first-text callback never
            # stops the indicator.
            status.stop()
        print()
        messages.append({"role": "assistant", "content": content})
        print_stats(usage)

    if args.prompt is not None:
        send(args.prompt)
        return 0

    print(f"{args.base_url.rstrip('/')}/v1/chat/completions")
    print("DeepSeek-V4-Flash-0731-Q4K-IQ2XXS-Q2K")
    print("帮助 /help 查看命令")
    while True:
        try:
            prompt = read_prompt("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt is None:
            print()
            return 0
        if not prompt.strip():
            continue
        command_line = prompt.strip()
        if "\n" in prompt:
            command = ""
            argument = ""
        else:
            command, _, argument = command_line.partition(" ")
        if command in {"/quit", "/exit", "/q"}:
            return 0
        if command == "/help":
            print_help()
            continue
        if command == "/new":
            messages.clear()
            print("已开始新对话。")
            continue
        if command == "/max_tokens" and argument.strip().isdigit():
            args.max_tokens = max(1, int(argument.strip()))
            print(f"max_tokens={args.max_tokens}")
            continue
        if command == "/think" and argument.strip() in {"on", "off"}:
            think_enabled = argument.strip() == "on"
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
