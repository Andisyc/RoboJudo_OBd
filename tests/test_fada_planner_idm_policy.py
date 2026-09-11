from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from robojudo.config.g1.g1_cfg import (
    g1_fada_planner_idm,
    g1_real_fada_planner_idm,
    g1_real_unilab,
    g1_unilab,
    g1_unilab_distill,
)
from robojudo.config.g1.policy.g1_fada_policy_cfg import G1FADAPlannerIDMPolicyCfg
from robojudo.policy.fada.checkpoint import (
    _canonical_state_dict_sha256,
    load_fada_policy_checkpoint,
)
from robojudo.policy.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from robojudo.policy.fada.observation import FADATorsoImuProjector, project_fada_g1_state
from robojudo.policy.fada_policy import FADAPlannerIDMPolicyAdapter


class _CheckpointCfg(G1FADAPlannerIDMPolicyCfg):
    checkpoint_path: str

    @property
    def policy_file(self) -> str:
        return self.checkpoint_path


def _write_checkpoint(path: Path) -> None:
    config = FADAArchitectureConfig(
        obs_dim=66,
        action_dim=29,
        command_dim=3,
        observation_contract="g1_fada_state_v2",
        history_length=30,
        prediction_horizon=6,
        hidden_dim=16,
        num_heads=4,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=32,
    )
    policy = FADAPlannerIDMPolicy(config)
    payload = {
        "schema_version": 5,
        "architecture": asdict(config),
        "planner_state_dict": policy.planner.state_dict(),
        "idm_state_dict": policy.idm.state_dict(),
        "training_schedule": "alternating_idm_then_planner",
        "planner_optimizer_state_dict": {},
        "idm_optimizer_state_dict": {},
    }
    payload["idm_sha256"] = _canonical_state_dict_sha256(payload["idm_state_dict"])
    torch.save(payload, path)


class TestFADAPlannerIDMMigration(unittest.TestCase):
    def test_config_isolated_and_complete(self):
        self.assertEqual(g1_unilab().policy.policy_type, "UniLabPolicy")
        self.assertEqual(g1_unilab_distill().policy.policy_type, "UniLabDistillPolicy")
        cfg = g1_fada_planner_idm()
        self.assertEqual(cfg.policy.policy_type, "FADAPlannerIDMPolicyAdapter")
        self.assertEqual(cfg.policy.history_length, 30)
        self.assertEqual(cfg.policy.prediction_horizon, 6)
        self.assertEqual(cfg.policy.execution_action_scale, 1.0)
        self.assertTrue(cfg.policy.gait_phase_enabled)
        self.assertFalse(cfg.policy.freeze_phase_during_dry_run)
        self.assertFalse(cfg.policy.preserve_state_during_dry_run)
        np.testing.assert_allclose(cfg.policy.initial_gait_phase, [0.0, np.pi])
        np.testing.assert_allclose(cfg.policy.fixed_gait_phase, [0.0, np.pi])
        np.testing.assert_allclose(cfg.policy.keyboard_command_magnitudes, [0.6, 0.4, 0.8])
        self.assertEqual(cfg.policy.action_scale, 1.0)
        self.assertEqual(cfg.policy.action_beta, 1.0)
        self.assertIsNone(cfg.policy.action_clip)

        real_cfg = g1_real_fada_planner_idm()
        self.assertEqual(real_cfg.env.env_type, "UnitreeCppEnv")
        self.assertEqual(real_cfg.env.odometry_type, "NONE")
        self.assertFalse(real_cfg.env.unitree.enable_odometry)
        self.assertTrue(real_cfg.env.unitree.enable_torso_imu)
        self.assertEqual(real_cfg.env.unitree.torso_imu_topic, "rt/secondary_imu")
        self.assertTrue(
            real_cfg.policy.policy_file.endswith(
                "assets/models/g1/fada/planner_idm_v022/"
                "planner_idm_close_some_dr_v001.onnx"
            )
        )
        self.assertEqual(real_cfg.policy.policy_type, "FADAPlannerIDMPolicyAdapter")
        self.assertEqual(
            [type(ctrl).__name__ for ctrl in real_cfg.ctrl],
            ["KeyboardCtrlCfg"],
        )
        self.assertTrue(real_cfg.do_safety_check)

        self.assertEqual(g1_real_unilab().env.env_type, "UnitreeCppEnv")

    def test_real_fada_torso_imu_matches_sim_tilt_and_aligns_starting_yaw(self):
        def quat_mul(lhs, rhs):
            lw, lx, ly, lz = lhs
            rw, rx, ry, rz = rhs
            return np.asarray(
                [
                    lw * rw - lx * rx - ly * ry - lz * rz,
                    lw * rx + lx * rw + ly * rz - lz * ry,
                    lw * ry - lx * rz + ly * rw + lz * rx,
                    lw * rz + lx * ry - ly * rx + lz * rw,
                ]
            )

        yaw = np.deg2rad(70.0)
        q_yaw = np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        for tilt_deg in (30.0, -30.0):
            tilt = np.deg2rad(tilt_deg)
            sin_tilt = np.sin(tilt)
            cos_tilt = np.cos(tilt)
            cases = [
                (
                    np.asarray([np.cos(tilt / 2.0), np.sin(tilt / 2.0), 0.0, 0.0]),
                    [0.0, sin_tilt, -cos_tilt],
                ),
                (
                    np.asarray([np.cos(tilt / 2.0), 0.0, np.sin(tilt / 2.0), 0.0]),
                    [-sin_tilt, 0.0, -cos_tilt],
                ),
            ]
            for q_tilt, expected_gravity in cases:
                projector = FADATorsoImuProjector()
                projector.project(
                    SimpleNamespace(
                        quaternion=q_yaw,
                        gyroscope=[0.0, 0.0, 0.0],
                        accelerometer=[0.0, 0.0, 9.81],
                    )
                )
                gyro, gravity = projector.project(
                    SimpleNamespace(
                        quaternion=quat_mul(q_yaw, q_tilt),
                        gyroscope=[0.1, -0.2, 0.3],
                        accelerometer=[0.0, 0.0, 9.81],
                    )
                )
                np.testing.assert_allclose(gyro, [0.1, -0.2, 0.3])
                np.testing.assert_allclose(gravity, expected_gravity, atol=1e-6)

        with self.assertRaisesRegex(RuntimeError, "quaternion norm"):
            projector.project(
                SimpleNamespace(
                    quaternion=[0.0, 0.0, 0.0, 0.0],
                    gyroscope=[0.0, 0.0, 0.0],
                    accelerometer=[0.0, 0.0, 9.81],
                )
            )

        with self.assertRaisesRegex(RuntimeError, "has not received"):
            projector.project(
                SimpleNamespace(
                    quaternion=[1.0, 0.0, 0.0, 0.0],
                    gyroscope=[0.0, 0.0, 0.0],
                    accelerometer=[0.0, 0.0, 0.0],
                )
            )

    def test_projection_uses_exact_non_leaking_indices(self):
        raw = np.arange(98, dtype=np.float32)[None, :]
        projected = project_fada_g1_state(raw)
        np.testing.assert_array_equal(
            projected, np.concatenate((raw[:, :64], raw[:, 96:98]), axis=1)
        )
        with self.assertRaises(ValueError):
            project_fada_g1_state(np.zeros((1, 97), dtype=np.float32))

    def test_keyboard_commands_are_idempotent_and_report_changes(self):
        adapter = object.__new__(FADAPlannerIDMPolicyAdapter)
        adapter._held_motion_keys = set()
        adapter._keyboard_command = np.zeros(3, dtype=np.float32)
        adapter.keyboard_command_magnitudes = np.asarray([0.6, 0.4, 0.8], dtype=np.float32)

        def key_events(name):
            return {
                "KeyboardCtrl": {
                    "keyboard_event": [
                        {"type": "keyboard", "name": name, "pressed": True},
                        {"type": "keyboard", "name": name, "pressed": False},
                    ]
                }
            }

        with self.assertLogs("robojudo.policy.fada_policy", level="INFO") as captured:
            np.testing.assert_allclose(adapter._get_commands(key_events("w")), [0.6, 0.0, 0.0])
            np.testing.assert_allclose(adapter._get_commands(key_events("w")), [0.6, 0.0, 0.0])
            np.testing.assert_allclose(adapter._get_commands(key_events("a")), [0.6, 0.4, 0.0])
            np.testing.assert_allclose(adapter._get_commands(key_events("x")), [0.0, 0.0, 0.0])

        self.assertEqual(len(captured.records), 3)
        self.assertIn("key=w vx=0.600 vy=0.000 yaw=0.000", captured.output[0])
        self.assertIn("key=x vx=0.000 vy=0.000 yaw=0.000", captured.output[-1])

    def test_trajectory_recording_starts_once_on_positive_forward_command(self):
        adapter = object.__new__(FADAPlannerIDMPolicyAdapter)
        adapter._trajectory_recording_requested = False

        with mock.patch(
            "robojudo.policy.fada_policy.request_trajectory_start"
        ) as request_start:
            adapter._maybe_start_trajectory_recording(
                np.asarray([0.0, 0.4, 0.8], dtype=np.float32)
            )
            adapter._maybe_start_trajectory_recording(
                np.asarray([-0.6, 0.0, 0.0], dtype=np.float32)
            )
            adapter._maybe_start_trajectory_recording(
                np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
            )
            adapter._maybe_start_trajectory_recording(
                np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
            )

        request_start.assert_called_once_with()

    def test_checkpoint_identity_and_stateful_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "planner_idm.pt"
            _write_checkpoint(checkpoint)
            loaded = load_fada_policy_checkpoint(checkpoint)
            self.assertFalse(loaded.policy.training)

            cfg = _CheckpointCfg(checkpoint_path=str(checkpoint))
            adapter = FADAPlannerIDMPolicyAdapter(cfg, "cpu")
            env_data = SimpleNamespace(
                base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                base_ang_vel=np.zeros(3, dtype=np.float32),
                policy_gyro=np.zeros(3, dtype=np.float32),
                policy_gravity=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
                dof_pos=np.asarray(adapter.default_dof_pos, dtype=np.float32),
                dof_vel=np.zeros(29, dtype=np.float32),
            )
            obs, extras = adapter.get_observation(env_data, {})
            action = adapter.get_action(obs)
            self.assertEqual(obs.shape, (98,))
            self.assertEqual(action.shape, (29,))
            np.testing.assert_allclose(obs[-2:], [0.0, np.pi])
            self.assertEqual(extras["fada_projected_obs_dim"], 66)
            self.assertEqual(
                tuple(adapter.playback_controller._observation_history.shape), (1, 30, 66)
            )
            self.assertEqual(
                tuple(adapter.playback_controller._action_history.shape), (1, 30, 29)
            )
            np.testing.assert_allclose(action, adapter.last_action)
            np.testing.assert_allclose(
                adapter.playback_controller._action_history[0, -1].cpu().numpy(),
                adapter.last_action,
            )
            with self.assertRaises(RuntimeError):
                adapter.get_action(obs)
            adapter.post_step_callback([])
            phase_delta = 2.0 * np.pi * adapter.gait_frequency * adapter.dt
            np.testing.assert_allclose(
                adapter.gait_phase,
                np.asarray([phase_delta, np.pi + phase_delta]) % (2.0 * np.pi),
            )
            adapter.reset()
            self.assertIsNone(adapter.playback_controller._observation_history)

            payload = torch.load(checkpoint, weights_only=True)
            first = next(iter(payload["idm_state_dict"].values()))
            first.view(-1)[0] += 1.0
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "IDM identity mismatch"):
                load_fada_policy_checkpoint(checkpoint)

    def test_real_observation_uses_unitree_torso_imu(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "planner_idm.pt"
            _write_checkpoint(checkpoint)
            cfg = _CheckpointCfg(checkpoint_path=str(checkpoint))
            adapter = FADAPlannerIDMPolicyAdapter(cfg, "cpu")
            env_data = SimpleNamespace(
                base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                base_ang_vel=np.asarray([9.0, 9.0, 9.0], dtype=np.float32),
                torso_imu_state=SimpleNamespace(
                    quaternion=[1.0, 0.0, 0.0, 0.0],
                    gyroscope=[0.1, -0.2, 0.3],
                    accelerometer=[0.0, 0.0, 9.81],
                ),
                dof_pos=np.asarray(adapter.default_dof_pos, dtype=np.float32),
                dof_vel=np.zeros(29, dtype=np.float32),
            )
            obs, _ = adapter.get_observation(env_data, {})
            action = adapter.get_action(obs)

            snapshot = adapter.snapshot_state()
            obs_next, _ = adapter.get_observation(env_data, {})
            adapter.get_action(obs_next)
            adapter.restore_state(snapshot)
            restored = adapter.snapshot_state()
            torch.testing.assert_close(
                restored["playback"]["observation_history"],  # type: ignore[index]
                snapshot["playback"]["observation_history"],  # type: ignore[index]
            )
            torch.testing.assert_close(
                restored["playback"]["action_history"],  # type: ignore[index]
                snapshot["playback"]["action_history"],  # type: ignore[index]
            )
            np.testing.assert_array_equal(restored["last_action"], snapshot["last_action"])

            np.testing.assert_allclose(obs[:3], [0.025, -0.05, 0.075])
            np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0])
            self.assertEqual(obs.shape, (98,))
            self.assertEqual(action.shape, (29,))
            self.assertTrue(np.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
