from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from robojudo.policy import policy_registry
from robojudo.utils.util_func import command_remap, get_gravity_orientation

from .unilab_policy import UniLabPolicy


@policy_registry.register
class UniLabDistillPolicy(UniLabPolicy):
    """RoboJuDo adapter for UniLab's G1 stand-height/walk DAgger student.

    The deployable actor observation is the 98-D G1 walk observation with a
    target-height command inserted after the three velocity commands.
    """

    def __init__(self, cfg_policy, device):
        self._pressed_motion_keys: set[str] = set()
        self.keyboard_input_scale = float(cfg_policy.keyboard_input_scale)
        if not 0.0 < self.keyboard_input_scale <= 1.0:
            raise ValueError("keyboard_input_scale must be in (0, 1]")
        self.command_xy_threshold = float(cfg_policy.command_xy_threshold)
        self.command_yaw_threshold = float(cfg_policy.command_yaw_threshold)
        self.stand_gait_phase = np.asarray(cfg_policy.stand_gait_phase, dtype=np.float32)
        if self.stand_gait_phase.shape != (2,):
            raise ValueError("stand_gait_phase must contain the left and right foot phases")
        self.target_height = float(cfg_policy.default_target_height)
        self.min_target_height = float(cfg_policy.min_target_height)
        self.max_target_height = float(cfg_policy.max_target_height)
        self.height_command_step = float(cfg_policy.height_command_step)
        self.height_up_key = str(cfg_policy.height_up_key)
        self.height_down_key = str(cfg_policy.height_down_key)
        if not self.min_target_height <= self.target_height <= self.max_target_height:
            raise ValueError("default_target_height must be within the configured height range")
        if self.height_command_step <= 0.0:
            raise ValueError("height_command_step must be positive")
        self.height_command_map = [float(value) for value in cfg_policy.height_command_map]
        if len(self.height_command_map) != 3:
            raise ValueError("height_command_map must contain [minimum, neutral, maximum]")
        super().__init__(cfg_policy=cfg_policy, device=device)

    def _check_runtime_contract(self):
        super()._check_runtime_contract()
        if self._runtime["kind"] != "torchscript":
            return
        probe = torch.zeros((1, self.expected_obs_dim), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self._runtime["model"](probe)
        if tuple(output.shape) != (1, self.expected_action_dim):
            raise ValueError(
                "UniLab distill TorchScript output shape mismatch: "
                f"expected (1, {self.expected_action_dim}), got {tuple(output.shape)}"
            )

    def reset(self):
        super().reset()
        self._pressed_motion_keys.clear()
        self.target_height = float(self.cfg_policy.default_target_height)

    def _get_commands(self, ctrl_data) -> np.ndarray:
        keyboard_data = ctrl_data.get("KeyboardCtrl")
        if keyboard_data is None:
            return super()._get_commands(ctrl_data)

        motion_keys = {"w", "s", "a", "d", "q", "e"}
        for event in keyboard_data.get("keyboard_event", []):
            if event.get("type") != "keyboard":
                continue
            name = event.get("name")
            if name not in motion_keys:
                continue
            if event.get("pressed", False):
                self._pressed_motion_keys.add(name)
            else:
                self._pressed_motion_keys.discard(name)

        forward = float("w" in self._pressed_motion_keys) - float(
            "s" in self._pressed_motion_keys
        )
        lateral = float("d" in self._pressed_motion_keys) - float(
            "a" in self._pressed_motion_keys
        )
        yaw = float("e" in self._pressed_motion_keys) - float(
            "q" in self._pressed_motion_keys
        )
        forward *= self.keyboard_input_scale
        lateral *= self.keyboard_input_scale
        yaw *= self.keyboard_input_scale
        return np.asarray(
            [
                command_remap(forward, self.command_maps[0]),
                command_remap(lateral, self.command_maps[1]),
                command_remap(yaw, self.command_maps[2]),
            ],
            dtype=np.float32,
        )

    def _get_height_command(self, ctrl_data) -> np.float32:
        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                if "RightY" in axes:
                    return np.float32(command_remap(axes["RightY"], self.height_command_map))
            if key == "KeyboardCtrl":
                for event in ctrl_data[key].get("keyboard_event", []):
                    if event.get("type") != "keyboard" or not event.get("pressed", False):
                        continue
                    if event.get("name") == self.height_up_key:
                        self.target_height += self.height_command_step
                    elif event.get("name") == self.height_down_key:
                        self.target_height -= self.height_command_step
                self.target_height = float(
                    np.clip(self.target_height, self.min_target_height, self.max_target_height)
                )
        return np.float32(self.target_height)

    def get_observation(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data)
        target_height = self._get_height_command(ctrl_data)
        command_active = (
            np.linalg.norm(commands[:2]) > self.command_xy_threshold
            or abs(float(commands[2])) > self.command_yaw_threshold
        )
        observation_gait_phase = self.gait_phase if command_active else self.stand_gait_phase
        policy_gravity = getattr(env_data, "policy_gravity", None)
        gravity = (
            get_gravity_orientation(env_data.base_quat).astype(np.float32)
            if policy_gravity is None
            else np.asarray(policy_gravity, dtype=np.float32)
        )
        dof_pos_rel = np.asarray(env_data.dof_pos - self.default_dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        policy_gyro = getattr(env_data, "policy_gyro", None)
        base_ang_vel = np.asarray(
            env_data.base_ang_vel if policy_gyro is None else policy_gyro,
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                base_ang_vel * 0.25,
                gravity,
                dof_pos_rel,
                dof_vel * 0.05,
                np.asarray(self.last_action, dtype=np.float32),
                commands,
                np.asarray([target_height], dtype=np.float32),
                observation_gait_phase.astype(np.float32),
            ],
        ).astype(np.float32)

        if obs.shape[0] != self.expected_obs_dim:
            raise ValueError(
                f"UniLab distill obs dim {obs.shape[0]} != expected {self.expected_obs_dim}"
            )
        self._last_obs: Optional[np.ndarray] = obs.copy()
        extras = {
            "commands": commands,
            "target_height": target_height,
            "gait_phase": observation_gait_phase.copy(),
            "unilab_obs_dim": obs.shape[0],
        }
        return obs, extras
