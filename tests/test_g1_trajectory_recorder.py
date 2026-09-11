from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "robojudo/tools/g1_trajectory_recorder.py"
SPEC = importlib.util.spec_from_file_location("g1_trajectory_recorder_test_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RECORDER_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECORDER_MODULE
SPEC.loader.exec_module(RECORDER_MODULE)

G1TrajectoryRecorder = RECORDER_MODULE.G1TrajectoryRecorder
UnitreeSubscriberBindings = RECORDER_MODULE.UnitreeSubscriberBindings
read_trajectory = RECORDER_MODULE.read_trajectory


def _imu():
    return SimpleNamespace(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[0.1, 0.2, 0.3],
        accelerometer=[0.0, 0.0, 9.81],
        rpy=[0.0, 0.0, 0.0],
    )


def _lowstate():
    motors = [SimpleNamespace(q=i, dq=i + 0.1, tau_est=i + 0.2) for i in range(29)]
    return SimpleNamespace(tick=123, mode_machine=5, motor_state=motors, imu_state=_imu())


def _lowcmd():
    motors = [
        SimpleNamespace(mode=1, q=i, dq=0.0, kp=20.0, kd=0.5, tau=0.0)
        for i in range(29)
    ]
    return SimpleNamespace(mode_pr=0, mode_machine=5, motor_cmd=motors)


class _FakeSubscriber:
    instances = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.callback = None
        self.closed = False
        self.instances.append(self)

    def Init(self, callback, _depth):
        self.callback = callback

    def Close(self):
        self.closed = True


class TestG1TrajectoryRecorder(unittest.TestCase):
    def test_subscriber_only_record_and_readback(self):
        _FakeSubscriber.instances = []
        initialized = []
        bindings = UnitreeSubscriberBindings(
            initialize=lambda domain, net_if: initialized.append((domain, net_if)),
            subscriber_type=_FakeSubscriber,
            lowstate_type=object,
            torso_imu_type=object,
            lowcmd_type=object,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectory.msgpack"
            recorder = G1TrajectoryRecorder(
                net_if="test0", output_path=path, bindings=bindings
            )
            callbacks = {subscriber.topic: subscriber.callback for subscriber in _FakeSubscriber.instances}
            callbacks["rt/lowstate"](_lowstate())
            callbacks["rt/secondary_imu"](_imu())
            callbacks["rt/lowcmd"](_lowcmd())
            summary = recorder.close()

            records = list(read_trajectory(path))

        self.assertEqual(initialized, [(0, "test0")])
        self.assertEqual(
            [subscriber.topic for subscriber in _FakeSubscriber.instances],
            ["rt/lowstate", "rt/secondary_imu", "rt/lowcmd"],
        )
        self.assertTrue(all(subscriber.closed for subscriber in _FakeSubscriber.instances))
        self.assertEqual(summary["counts"], {"lowstate": 1, "torso_imu": 1, "lowcmd": 1})
        self.assertEqual(summary["dropped"], 0)
        self.assertEqual([record["kind"] for record in records], [
            "header", "lowstate", "torso_imu", "lowcmd", "summary"
        ])
        self.assertEqual(records[0]["payload"]["metadata"], {"net_if": "test0", "num_dofs": 29})
        self.assertEqual(records[1]["payload"]["tick"], 123)
        self.assertEqual(len(records[1]["payload"]["motor"]["q"]), 29)
        self.assertEqual(len(records[3]["payload"]["motor"]["q"]), 29)

    def test_armed_recorder_ignores_pretrigger_data_and_names_duration(self):
        _FakeSubscriber.instances = []
        bindings = UnitreeSubscriberBindings(
            initialize=lambda _domain, _net_if: None,
            subscriber_type=_FakeSubscriber,
            lowstate_type=object,
            torso_imu_type=object,
            lowcmd_type=object,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            provisional_path = Path(tmpdir) / "trajectory.msgpack"
            with mock.patch.object(
                RECORDER_MODULE,
                "default_output_path",
                return_value=provisional_path,
            ):
                recorder = G1TrajectoryRecorder(
                    net_if="test0", bindings=bindings, armed=True
                )

            callbacks = {
                subscriber.topic: subscriber.callback
                for subscriber in _FakeSubscriber.instances
            }
            callbacks["rt/lowstate"](_lowstate())
            self.assertFalse(provisional_path.exists())

            self.assertTrue(recorder.start())
            callbacks["rt/lowstate"](_lowstate())
            callbacks["rt/secondary_imu"](_imu())
            callbacks["rt/lowcmd"](_lowcmd())
            summary = recorder.close()
            final_path = recorder.output_path
            records = list(read_trajectory(final_path))

        self.assertRegex(final_path.name, r"^trajectory_\d+\.\d+s\.msgpack$")
        self.assertEqual(summary["counts"], {"lowstate": 1, "torso_imu": 1, "lowcmd": 1})
        self.assertGreaterEqual(summary["duration_seconds"], 0.0)
        self.assertEqual(records[-1]["kind"], "summary")
        self.assertEqual(
            records[-1]["payload"]["duration_seconds"],
            summary["duration_seconds"],
        )


if __name__ == "__main__":
    unittest.main()
