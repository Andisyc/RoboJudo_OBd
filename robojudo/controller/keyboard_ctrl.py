import atexit
import os
import sys
import time
from queue import Empty, Queue
from typing import List, Optional

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg
from robojudo.tools.trajectory_trigger import request_trajectory_start


@ctrl_registry.register
class KeyboardCtrl(Controller):
    cfg_ctrl: KeyboardCtrlCfg

    def __init__(self, cfg_ctrl: KeyboardCtrlCfg, env=None, **kwargs):  # TODO
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, **kwargs)

        self.event_queue = Queue(maxsize=100)
        self.keyboard_input = self._create_keyboard_input(cfg_ctrl.backend)
        self.keyboard_input.start()
        atexit.register(self.close)

        self.reset()

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend != "auto":
            return backend
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            return "terminal"
        return "pynput"

    def _create_keyboard_input(self, backend: str):
        resolved_backend = self._resolve_backend(backend)
        if resolved_backend == "terminal":
            from robojudo.controller.utils.terminal_keyboard import TerminalKeyboardInput

            return TerminalKeyboardInput(self.event_queue)
        if resolved_backend == "pynput":
            try:
                from robojudo.controller.utils.keyboard import KeyboardThread
            except ImportError as exc:
                raise RuntimeError(
                    "KeyboardCtrl could not start the pynput backend; use "
                    "KeyboardCtrlCfg(backend='terminal') in an SSH terminal"
                ) from exc
            return KeyboardThread(self.event_queue)
        raise ValueError(f"Unknown KeyboardCtrl backend: {resolved_backend}")

    def reset(self):
        reset_input = getattr(self.keyboard_input, "reset", None)
        if reset_input is not None:
            reset_input()
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except Empty:
                break

    def get_events(self):
        poll_input = getattr(self.keyboard_input, "poll", None)
        if poll_input is not None:
            poll_input()
        events = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                events.append(event)
                if (
                    event.get("type") == "keyboard"
                    and event.get("name") == "w"
                    and event.get("pressed", False)
                ):
                    request_trajectory_start()
            except Empty:
                break
        return events

    def get_data(self):
        return {"keyboard_event": self.get_events()}

    def close(self):
        close_input = getattr(self, "keyboard_input", None)
        close = getattr(close_input, "close", None)
        if close is not None:
            close()

    def post_step_callback(self, commands: Optional[List[str]] = None):
        if commands and "[SHUTDOWN]" in commands:
            self.close()

    def process_triggers(self, ctrl_data):
        commands = []
        if len(self.triggers) == 0:
            return ctrl_data, commands

        for event in ctrl_data["keyboard_event"]:
            if event["type"] == "keyboard" and not event["pressed"]:  # trigger when key is released
                command = self.triggers.get(event["name"], None)
                if command is not None:
                    commands.append(command)
                    # remove event after triggered
                    ctrl_data["keyboard_event"].remove(event)

        return ctrl_data, commands


if __name__ == "__main__":
    kb_ctrl = KeyboardCtrl(
        cfg_ctrl=KeyboardCtrlCfg(
            triggers={
                "Key.space": "[TEST]",
                "\x01": "[CTRL_A]",
            }
        )
    )
    while True:
        data = kb_ctrl.get_data()
        ctrl_data, commands = kb_ctrl.process_triggers(data)
        if ctrl_data["keyboard_event"]:
            for e in ctrl_data["keyboard_event"]:
                print(e)
        if commands:
            print("Commands:", commands)
        time.sleep(0.1)
