from __future__ import annotations

import os
import pty
import termios
import time
import unittest
from queue import Queue
from unittest import mock

from robojudo.controller.keyboard_ctrl import KeyboardCtrl
from robojudo.controller.utils.terminal_keyboard import TerminalKeyboardInput


class TestTerminalKeyboardCtrl(unittest.TestCase):
    def setUp(self):
        self.master_fd, slave_fd = pty.openpty()
        self.stream = os.fdopen(os.dup(slave_fd), "rb", buffering=0)
        os.close(slave_fd)
        self.original_settings = termios.tcgetattr(self.stream.fileno())
        self.queue = Queue()
        self.keyboard = TerminalKeyboardInput(
            self.queue,
            stream=self.stream,
            escape_timeout=0.001,
        )
        self.keyboard.start()

    def tearDown(self):
        self.keyboard.close()
        self.stream.close()
        os.close(self.master_fd)

    def _events(self):
        events = []
        while not self.queue.empty():
            events.append(self.queue.get_nowait())
        return events

    def test_ascii_key_emits_existing_press_release_contract_and_restores_terminal(self):
        os.write(self.master_fd, b"W")
        self.keyboard.poll()
        events = self._events()
        self.assertEqual(
            [(event["name"], event["pressed"]) for event in events],
            [("w", True), ("w", False)],
        )

        self.keyboard.close()
        restored_settings = termios.tcgetattr(self.stream.fileno())
        terminal_mode_flags = termios.ICANON | termios.ECHO
        self.assertEqual(
            restored_settings[3] & terminal_mode_flags,
            self.original_settings[3] & terminal_mode_flags,
        )
        self.assertEqual(restored_settings[6], self.original_settings[6])

    def test_escape_is_delayed_but_arrow_key_does_not_become_shutdown(self):
        os.write(self.master_fd, b"\x1b[A")
        self.keyboard.poll()
        self.assertEqual(
            [(event["name"], event["pressed"]) for event in self._events()],
            [("Key.up", True), ("Key.up", False)],
        )

        os.write(self.master_fd, b"\x1b")
        self.keyboard.poll()
        self.assertEqual(self._events(), [])
        time.sleep(0.002)
        self.keyboard.poll()
        self.assertEqual(
            [(event["name"], event["pressed"]) for event in self._events()],
            [("Key.esc", True), ("Key.esc", False)],
        )

    def test_auto_backend_selects_terminal_only_for_headless_linux(self):
        with mock.patch("sys.platform", "linux"), mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(KeyboardCtrl._resolve_backend("auto"), "terminal")
        with mock.patch("sys.platform", "darwin"), mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(KeyboardCtrl._resolve_backend("auto"), "pynput")
        self.assertEqual(KeyboardCtrl._resolve_backend("terminal"), "terminal")


if __name__ == "__main__":
    unittest.main()
