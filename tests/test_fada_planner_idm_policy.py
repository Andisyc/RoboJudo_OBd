from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from robojudo.config.g1.g1_cfg import g1_fada_planner_idm, g1_unilab, g1_unilab_distill
from robojudo.config.g1.policy.g1_fada_policy_cfg import G1FADAPlannerIDMPolicyCfg
from robojudo.policy.fada.checkpoint import (
    _canonical_state_dict_sha256,
    load_fada_policy_checkpoint,
)
from robojudo.policy.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from robojudo.policy.fada.observation import project_fada_g1_state
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
        self.assertFalse(cfg.policy.gait_phase_enabled)
        np.testing.assert_allclose(cfg.policy.fixed_gait_phase, [0.0, 0.0])
        self.assertEqual(cfg.policy.action_scale, 1.0)
        self.assertEqual(cfg.policy.action_beta, 1.0)
        self.assertIsNone(cfg.policy.action_clip)

    def test_projection_uses_exact_non_leaking_indices(self):
        raw = np.arange(98, dtype=np.float32)[None, :]
        projected = project_fada_g1_state(raw)
        np.testing.assert_array_equal(
            projected, np.concatenate((raw[:, :64], raw[:, 96:98]), axis=1)
        )
        with self.assertRaises(ValueError):
            project_fada_g1_state(np.zeros((1, 97), dtype=np.float32))

    def test_keyboard_command_toggles_at_fada_walk_speed(self):
        adapter = object.__new__(FADAPlannerIDMPolicyAdapter)
        adapter._held_motion_keys = set()
        adapter._keyboard_command = np.zeros(3, dtype=np.float32)
        adapter.keyboard_command_magnitude = 0.4
        pressed = {
            "KeyboardCtrl": {
                "keyboard_event": [{"type": "keyboard", "name": "w", "pressed": True}]
            }
        }
        released = {
            "KeyboardCtrl": {
                "keyboard_event": [{"type": "keyboard", "name": "w", "pressed": False}]
            }
        }
        np.testing.assert_allclose(adapter._get_commands(pressed), [0.4, 0.0, 0.0])
        np.testing.assert_allclose(adapter._get_commands({"KeyboardCtrl": {}}), [0.4, 0.0, 0.0])
        np.testing.assert_allclose(adapter._get_commands(released), [0.4, 0.0, 0.0])
        np.testing.assert_allclose(adapter._get_commands(pressed), [0.0, 0.0, 0.0])

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
            np.testing.assert_allclose(obs[-2:], [0.0, 0.0])
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
            np.testing.assert_allclose(adapter.gait_phase, [0.0, 0.0])
            adapter.reset()
            self.assertIsNone(adapter.playback_controller._observation_history)

            payload = torch.load(checkpoint, weights_only=True)
            first = next(iter(payload["idm_state_dict"].values()))
            first.view(-1)[0] += 1.0
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "IDM identity mismatch"):
                load_fada_policy_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
