from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import msgpack


SCHEMA_VERSION = 1
DEFAULT_TOPICS = {
    "lowstate": "rt/lowstate",
    "torso_imu": "rt/secondary_imu",
    "lowcmd": "rt/lowcmd",
}


def _field(obj: Any, name: str) -> Any:
    value = getattr(obj, name)
    return value() if callable(value) else value


def _floats(values: Any) -> list[float]:
    return [float(value) for value in values]


def _imu_payload(imu: Any) -> dict[str, list[float]]:
    return {
        "quaternion": _floats(_field(imu, "quaternion")),
        "gyroscope": _floats(_field(imu, "gyroscope")),
        "accelerometer": _floats(_field(imu, "accelerometer")),
        "rpy": _floats(_field(imu, "rpy")),
    }


def serialize_lowstate(message: Any, num_dofs: int = 29) -> dict[str, Any]:
    motors = list(_field(message, "motor_state"))[:num_dofs]
    if len(motors) != num_dofs:
        raise ValueError(f"LowState contains {len(motors)} motors, expected {num_dofs}")
    return {
        "tick": int(_field(message, "tick")),
        "mode_machine": int(_field(message, "mode_machine")),
        "motor": {
            "q": [float(_field(motor, "q")) for motor in motors],
            "dq": [float(_field(motor, "dq")) for motor in motors],
            "tau_est": [float(_field(motor, "tau_est")) for motor in motors],
        },
        "base_imu": _imu_payload(_field(message, "imu_state")),
    }


def serialize_torso_imu(message: Any) -> dict[str, Any]:
    return _imu_payload(message)


def serialize_lowcmd(message: Any, num_dofs: int = 29) -> dict[str, Any]:
    motors = list(_field(message, "motor_cmd"))[:num_dofs]
    if len(motors) != num_dofs:
        raise ValueError(f"LowCmd contains {len(motors)} motors, expected {num_dofs}")
    return {
        "mode_pr": int(_field(message, "mode_pr")),
        "mode_machine": int(_field(message, "mode_machine")),
        "motor": {
            "mode": [int(_field(motor, "mode")) for motor in motors],
            "q": [float(_field(motor, "q")) for motor in motors],
            "dq": [float(_field(motor, "dq")) for motor in motors],
            "kp": [float(_field(motor, "kp")) for motor in motors],
            "kd": [float(_field(motor, "kd")) for motor in motors],
            "tau": [float(_field(motor, "tau")) for motor in motors],
        },
    }


def default_output_path() -> Path:
    timestamp = time.strftime("%y%m%d-%H%M%S")
    return Path("logs") / f"g1_trajectory_{timestamp}.msgpack"


class MsgpackEventWriter:
    _STOP = object()

    def __init__(
        self,
        output_path: str | Path,
        *,
        queue_size: int = 10000,
        metadata: dict[str, Any] | None = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._counts = {kind: 0 for kind in DEFAULT_TOPICS}
        self._dropped = 0
        self._closed = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="g1-trajectory-writer")
        self._thread.start()
        self.record(
            "header",
            {
                "schema_version": SCHEMA_VERSION,
                "created_time_ns": time.time_ns(),
                "topics": DEFAULT_TOPICS.copy(),
                "metadata": dict(metadata or {}),
            },
            count=False,
        )

    def record(self, kind: str, payload: dict[str, Any], *, count: bool = True):
        if self._closed:
            return
        event = {
            "kind": kind,
            "receive_time_ns": time.time_ns(),
            "monotonic_time_ns": time.monotonic_ns(),
            "payload": payload,
        }
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return
        if count and kind in self._counts:
            with self._lock:
                self._counts[kind] += 1

    def _run(self):
        try:
            with self.output_path.open("wb") as stream:
                last_flush = time.monotonic()
                while True:
                    item = self._queue.get()
                    if item is self._STOP:
                        break
                    stream.write(msgpack.packb(item, use_bin_type=True))
                    if time.monotonic() - last_flush >= 0.5:
                        stream.flush()
                        last_flush = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException as exc:
            self._error = exc

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("trajectory writer failed") from self._error
        if self._dropped:
            raise RuntimeError(f"trajectory writer queue overflowed; dropped={self._dropped}")

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.summary()
        summary = self.summary()
        self.record("summary", summary, count=False)
        self._closed = True
        self._queue.put(self._STOP)
        self._thread.join()
        self.raise_if_failed()
        return summary

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {"counts": self._counts.copy(), "dropped": self._dropped}


@dataclass(frozen=True)
class UnitreeSubscriberBindings:
    initialize: Callable[[int, str], Any]
    subscriber_type: type
    lowstate_type: type
    torso_imu_type: type
    lowcmd_type: type


def load_unitree_subscriber_bindings() -> UnitreeSubscriberBindings:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

    return UnitreeSubscriberBindings(
        initialize=ChannelFactoryInitialize,
        subscriber_type=ChannelSubscriber,
        lowstate_type=LowState_,
        torso_imu_type=IMUState_,
        lowcmd_type=LowCmd_,
    )


class G1TrajectoryRecorder:
    def __init__(
        self,
        *,
        net_if: str,
        output_path: str | Path | None = None,
        num_dofs: int = 29,
        bindings: UnitreeSubscriberBindings | None = None,
    ):
        if num_dofs <= 0:
            raise ValueError("num_dofs must be positive")
        self.num_dofs = num_dofs
        sdk = bindings or load_unitree_subscriber_bindings()
        self.writer = MsgpackEventWriter(
            output_path or default_output_path(),
            metadata={"net_if": net_if, "num_dofs": num_dofs},
        )
        self._subscribers: list[Any] = []
        self._closed = False
        self._callback_error: BaseException | None = None
        try:
            sdk.initialize(0, net_if)
            specifications = (
                (DEFAULT_TOPICS["lowstate"], sdk.lowstate_type, self._on_lowstate),
                (DEFAULT_TOPICS["torso_imu"], sdk.torso_imu_type, self._on_torso_imu),
                (DEFAULT_TOPICS["lowcmd"], sdk.lowcmd_type, self._on_lowcmd),
            )
            for topic, message_type, callback in specifications:
                subscriber = sdk.subscriber_type(topic, message_type)
                subscriber.Init(callback, 10)
                self._subscribers.append(subscriber)
        except BaseException:
            self.writer.close()
            raise

    @property
    def output_path(self) -> Path:
        return self.writer.output_path

    def _on_lowstate(self, message: Any):
        try:
            self.writer.record("lowstate", serialize_lowstate(message, self.num_dofs))
        except BaseException as exc:
            self._callback_error = exc

    def _on_torso_imu(self, message: Any):
        try:
            self.writer.record("torso_imu", serialize_torso_imu(message))
        except BaseException as exc:
            self._callback_error = exc

    def _on_lowcmd(self, message: Any):
        try:
            self.writer.record("lowcmd", serialize_lowcmd(message, self.num_dofs))
        except BaseException as exc:
            self._callback_error = exc

    def raise_if_failed(self):
        self.writer.raise_if_failed()
        if self._callback_error is not None:
            raise RuntimeError("trajectory subscriber callback failed") from self._callback_error

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.writer.summary()
        self._closed = True
        for subscriber in self._subscribers:
            close = getattr(subscriber, "Close", None)
            if callable(close):
                close()
        summary = self.writer.close()
        self.raise_if_failed()
        return summary


def read_trajectory(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("rb") as stream:
        yield from msgpack.Unpacker(stream, raw=False)
