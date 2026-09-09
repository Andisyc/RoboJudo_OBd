from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robojudo.config.g1.env.g1_mujuco_env_cfg import G1UniLabMujocoEnvCfg
from robojudo.config.g1.policy.g1_unilab_policy_cfg import G1UniLabDistillPolicyCfg
from robojudo.environment.mujoco_env import MujocoEnv
from robojudo.pipeline.rl_pipeline import PolicyWrapper
from robojudo.policy.unilab_distill_policy import UniLabDistillPolicy


class _HeadlessViewer:
    def __init__(self, *args, **kwargs):
        self.cam = SimpleNamespace(distance=0.0, elevation=0.0, azimuth=0.0, lookat=None)
        self.is_alive = False

    def render(self):
        pass

    def close(self):
        pass


class TestUniLabDistillPolicy(unittest.TestCase):
    def test_distill_config_is_isolated_from_legacy_unilab_config(self):
        cfg = G1UniLabDistillPolicyCfg()

        self.assertEqual(cfg.policy_type, "UniLabDistillPolicy")
        self.assertEqual(cfg.expected_obs_dim, 99)
        self.assertEqual(cfg.expected_action_dim, 29)
        np.testing.assert_allclose(cfg.initial_gait_phase, [0.0, np.pi])
        self.assertEqual(cfg.action_dof.stiffness[0], 40.179)
        self.assertEqual(cfg.action_dof.damping[0], 2.558)
        self.assertEqual(cfg.action_dof.torque_limits[0], 88.0)
        self.assertTrue(
            cfg.policy_file.endswith(
                "/stand_height_walk_ordered_b_r4_dagger_iteration_1/policy.pt"
            )
        )

    def test_distill_observation_inserts_height_after_velocity_command(self):
        policy = object.__new__(UniLabDistillPolicy)
        policy.default_dof_pos = np.zeros(29, dtype=np.float32)
        policy.last_action = np.zeros(29, dtype=np.float32)
        policy.gait_phase = np.asarray([0.25, 0.75], dtype=np.float32)
        policy.expected_obs_dim = 99
        policy.target_height = 0.702
        policy.min_target_height = 0.650
        policy.max_target_height = 0.754
        policy.height_command_step = 0.010
        policy.height_up_key = "r"
        policy.height_down_key = "f"
        policy.height_command_map = [0.65, 0.754, 0.754]
        policy.command_xy_threshold = 0.05
        policy.command_yaw_threshold = 0.05
        policy.stand_gait_phase = np.asarray([np.pi, np.pi], dtype=np.float32)
        policy._last_obs = None

        env_data = SimpleNamespace(
            base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            dof_pos=np.zeros(29, dtype=np.float32),
            dof_vel=np.zeros(29, dtype=np.float32),
        )

        obs, extras = policy.get_observation(env_data, {})

        np.testing.assert_allclose(obs[93:99], [0.0, 0.0, 0.0, 0.702, np.pi, np.pi])
        self.assertEqual(extras["target_height"], np.float32(0.702))

    def test_keyboard_height_tracking_updates_and_clamps_target(self):
        policy = object.__new__(UniLabDistillPolicy)
        policy.target_height = 0.700
        policy.min_target_height = 0.650
        policy.max_target_height = 0.754
        policy.height_command_step = 0.010
        policy.height_up_key = "r"
        policy.height_down_key = "f"
        policy.height_command_map = [0.65, 0.754, 0.754]

        press_up = {
            "KeyboardCtrl": {
                "keyboard_event": [{"type": "keyboard", "name": "r", "pressed": True}]
            }
        }
        release_down = {
            "KeyboardCtrl": {
                "keyboard_event": [{"type": "keyboard", "name": "f", "pressed": False}]
            }
        }

        self.assertAlmostEqual(float(policy._get_height_command(press_up)), 0.710, places=6)
        self.assertAlmostEqual(float(policy._get_height_command(release_down)), 0.710, places=6)
        for _ in range(20):
            policy._get_height_command(press_up)
        self.assertEqual(policy._get_height_command({}), np.float32(0.754))

        press_down = {
            "KeyboardCtrl": {
                "keyboard_event": [{"type": "keyboard", "name": "f", "pressed": True}]
            }
        }
        for _ in range(20):
            policy._get_height_command(press_down)
        self.assertEqual(policy._get_height_command({}), np.float32(0.650))

    def test_keyboard_motion_command_persists_until_key_release(self):
        policy = object.__new__(UniLabDistillPolicy)
        policy._pressed_motion_keys = set()
        policy.keyboard_input_scale = 0.4
        policy.command_maps = [
            [-0.6, 0.0, 1.0],
            [0.4, 0.0, -0.4],
            [0.8, 0.0, -0.8],
        ]
        press_w = {
            "KeyboardCtrl": {
                "keyboard_event": [{"type": "keyboard", "name": "w", "pressed": True}]
            }
        }
        empty_frame = {"KeyboardCtrl": {"keyboard_event": []}}
        release_w = {
            "KeyboardCtrl": {
                "keyboard_event": [{"type": "keyboard", "name": "w", "pressed": False}]
            }
        }

        np.testing.assert_allclose(policy._get_commands(press_w), [0.4, 0.0, 0.0])
        np.testing.assert_allclose(policy._get_commands(empty_frame), [0.4, 0.0, 0.0])
        np.testing.assert_allclose(policy._get_commands(release_w), [0.0, 0.0, 0.0])

    def test_opposite_keyboard_motion_keys_cancel(self):
        policy = object.__new__(UniLabDistillPolicy)
        policy._pressed_motion_keys = set()
        policy.keyboard_input_scale = 0.4
        policy.command_maps = [
            [-0.6, 0.0, 1.0],
            [0.4, 0.0, -0.4],
            [0.8, 0.0, -0.8],
        ]
        press_opposites = {
            "KeyboardCtrl": {
                "keyboard_event": [
                    {"type": "keyboard", "name": "a", "pressed": True},
                    {"type": "keyboard", "name": "d", "pressed": True},
                ]
            }
        }

        np.testing.assert_allclose(policy._get_commands(press_opposites), [0.0, 0.0, 0.0])

    def test_headless_mujoco_step_uses_distill_policy(self):
        env_cfg = G1UniLabMujocoEnvCfg(visualize_extras=False)
        with patch("mujoco_viewer.MujocoViewer", _HeadlessViewer):
            env = MujocoEnv(cfg_env=env_cfg, device="cpu")
            policy = PolicyWrapper(
                cfg_policy=G1UniLabDistillPolicyCfg(),
                env_dof_cfg=env.dof_cfg,
                device="cpu",
            )
            env.update_dof_cfg(override_cfg=policy.cfg_action_dof)
            env.reset()
            policy.reset()

            for _ in range(100):
                obs, extras = policy.get_observation(env.get_data(), {})
                pd_target = policy.get_pd_target(obs)
                env.step(pd_target)
                policy.post_step_callback([])

        self.assertEqual(obs.shape, (99,))
        self.assertEqual(pd_target.shape, (29,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(np.isfinite(pd_target).all())
        self.assertEqual(extras["target_height"], np.float32(0.754))
        self.assertGreater(float(env.data.qpos[2]), 0.65)
        self.assertLess(float(np.linalg.norm(env._base_rpy[:2])), 0.2)

    def test_headless_keyboard_forward_command_moves_without_falling(self):
        env_cfg = G1UniLabMujocoEnvCfg(visualize_extras=False)
        with patch("mujoco_viewer.MujocoViewer", _HeadlessViewer):
            env = MujocoEnv(cfg_env=env_cfg, device="cpu")
            policy = PolicyWrapper(
                cfg_policy=G1UniLabDistillPolicyCfg(),
                env_dof_cfg=env.dof_cfg,
                device="cpu",
            )
            env.update_dof_cfg(override_cfg=policy.cfg_action_dof)
            env.reset()
            policy.reset()
            start_x = float(env.data.qpos[0])

            for step in range(150):
                events = []
                if step == 0:
                    events.append({"type": "keyboard", "name": "w", "pressed": True})
                ctrl_data = {"KeyboardCtrl": {"keyboard_event": events}}
                obs, extras = policy.get_observation(env.get_data(), ctrl_data)
                pd_target = policy.get_pd_target(obs)
                env.step(pd_target)
                policy.post_step_callback([])

        self.assertAlmostEqual(float(extras["commands"][0]), 0.4, places=6)
        self.assertGreater(float(env.data.qpos[2]), 0.65)
        self.assertLess(float(np.linalg.norm(env._base_rpy[:2])), 0.2)
        self.assertGreater(float(env.data.qpos[0]) - start_x, 0.1)


if __name__ == "__main__":
    unittest.main()
