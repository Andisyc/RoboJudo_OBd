from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from queue import Full, Queue
from typing import BinaryIO, Optional


class TerminalKeyboardInput:
    """Non-blocking keyboard input for an interactive POSIX terminal."""

    _SPECIAL_KEYS = {
        b"\x1b[A": "Key.up",
        b"\x1b[B": "Key.down",
        b"\x1b[C": "Key.right",
        b"\x1b[D": "Key.left",
    }

    def __init__(
        self,
        event_queue: Queue,
        stream: Optional[BinaryIO] = None,
        escape_timeout: float = 0.03,
    ):
        self.event_queue = event_queue
        self.stream = sys.stdin if stream is None else stream
        self.fd = self.stream.fileno()
        if not os.isatty(self.fd):
            raise RuntimeError(
                "KeyboardCtrl terminal backend requires an interactive TTY; "
                "run RoboJuDo directly from an SSH or local terminal"
            )
        self.escape_timeout = escape_timeout
        self._original_settings = None
        self._buffer = bytearray()
        self._escape_started_at: Optional[float] = None
        self._closed = False

    def start(self):
        if self._original_settings is not None:
            return
        self._original_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def reset(self):
        self._buffer.clear()
        self._escape_started_at = None

    def _emit_key(self, name: str):
        timestamp = time.time()
        for pressed in (True, False):
            try:
                self.event_queue.put_nowait(
                    {
                        "type": "keyboard",
                        "name": name,
                        "pressed": pressed,
                        "timestamp": timestamp,
                    }
                )
            except Full:
                break

    def _consume_buffer(self, now: float):
        while self._buffer:
            if self._buffer[0] != 0x1B:
                value = self._buffer.pop(0)
                if value in (10, 13):
                    name = "Key.enter"
                elif value == 9:
                    name = "Key.tab"
                elif value in (8, 127):
                    name = "Key.backspace"
                elif 32 <= value <= 126:
                    name = chr(value).lower()
                else:
                    continue
                self._emit_key(name)
                continue

            matched = next(
                (sequence for sequence in self._SPECIAL_KEYS if self._buffer.startswith(sequence)),
                None,
            )
            if matched is not None:
                del self._buffer[: len(matched)]
                self._escape_started_at = None
                self._emit_key(self._SPECIAL_KEYS[matched])
                continue

            if len(self._buffer) >= 2 and self._buffer[1] in (ord("["), ord("O")):
                final_index = next(
                    (
                        index
                        for index, value in enumerate(self._buffer[2:], start=2)
                        if 0x40 <= value <= 0x7E
                    ),
                    None,
                )
                if final_index is not None:
                    del self._buffer[: final_index + 1]
                    self._escape_started_at = None
                    continue

            if self._escape_started_at is None:
                self._escape_started_at = now
            if now - self._escape_started_at < self.escape_timeout:
                return
            del self._buffer[0]
            self._escape_started_at = None
            self._emit_key("Key.esc")

    def poll(self):
        if self._closed:
            return
        while select.select([self.fd], [], [], 0.0)[0]:
            chunk = os.read(self.fd, 64)
            if not chunk:
                self._emit_key("Key.esc")
                self.close()
                return
            self._buffer.extend(chunk)
        self._consume_buffer(time.monotonic())

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._original_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._original_settings)
            self._original_settings = None
