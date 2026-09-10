from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from robojudo.policy import policy_registry
from robojudo.policy.base_policy import Policy
from robojudo.utils.util_func import command_remap, get_gravity_orientation


@policy_registry.register
class UniLabPolicy(Policy):
    """Adapter for UniLab G1WalkFlat locomotion policies.

    UniLab G1WalkFlat actor obs order:
    gyro * 0.25, -gravity, dof_pos-default, dof_vel*0.05, action, command, raw gait_phase.
    """

    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)

        self.expected_obs_dim = int(getattr(cfg_policy, "expected_obs_dim", 98))
        self.expected_action_dim = int(getattr(cfg_policy, "expected_action_dim", 29))
        if self.num_actions != self.expected_action_dim:
            raise ValueError(
                f"UniLabPolicy action DoF mismatch: cfg has {self.num_actions}, "
                f"expected {self.expected_action_dim}."
            )

        self.gait_frequency = float(cfg_policy.gait_frequency)
        self.initial_gait_phase = np.asarray(
            getattr(cfg_policy, "initial_gait_phase", [0.0, np.pi]), dtype=np.float32
        )
        if self.initial_gait_phase.shape != (2,):
            raise ValueError(
                f"UniLabPolicy initial_gait_phase must have shape (2,), "
                f"got {self.initial_gait_phase.shape}."
            )
        self.gait_phase = self.initial_gait_phase.copy()
        self.freeze_phase_during_dry_run = bool(
            getattr(cfg_policy, "freeze_phase_during_dry_run", True)
        )
        self.use_torso_obs_source = bool(getattr(cfg_policy, "use_torso_obs_source", True))

        self.command_maps = [list(v) for v in cfg_policy.command_maps]
        self.debug_checks = bool(getattr(cfg_policy, "debug_checks", True))
        self._last_obs: Optional[np.ndarray] = None

        self._runtime = self._build_runtime(cfg_policy.policy_file)
        self._check_runtime_contract()

    def _build_runtime(self, policy_file: str) -> dict[str, Any]:
        if not os.path.isfile(policy_file):
            raise FileNotFoundError(f"UniLab policy file not found: {policy_file}")

        suffix = Path(policy_file).suffix.lower()
        if suffix == ".onnx":
            import onnxruntime as ort

            session = ort.InferenceSession(policy_file)
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1:
                raise ValueError(f"UniLab ONNX expects one input, got {len(inputs)}")
            if len(outputs) < 1:
                raise ValueError("UniLab ONNX has no outputs")
            return {
                "kind": "onnx",
                "session": session,
                "input_name": inputs[0].name,
                "output_name": outputs[0].name,
                "input_shape": inputs[0].shape,
                "output_shape": outputs[0].shape,
            }

        if suffix in {".pt", ".jit", ".torchscript"}:
            model = torch.jit.load(policy_file, map_location=self.device)
            model.eval()
            return {"kind": "torchscript", "model": model}

        raise ValueError(f"Unsupported UniLab policy format: {policy_file}")

    def _check_runtime_contract(self):
        if self._runtime["kind"] != "onnx":
            return

        in_dim = self._last_dim(self._runtime["input_shape"])
        out_dim = self._last_dim(self._runtime["output_shape"])
        if in_dim is not None and in_dim != self.expected_obs_dim:
            raise ValueError(
                f"UniLab ONNX input dim {in_dim} != expected {self.expected_obs_dim}"
            )
        if out_dim is not None and out_dim != self.expected_action_dim:
            raise ValueError(
                f"UniLab ONNX output dim {out_dim} != expected {self.expected_action_dim}"
            )

    @staticmethod
    def _last_dim(shape) -> Optional[int]:
        if not shape:
            return None
        dim = shape[-1]
        return int(dim) if isinstance(dim, int) else None

    def reset(self):
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.gait_phase = self.initial_gait_phase.copy()
        self._last_obs = None

    def snapshot_state(self) -> dict[str, np.ndarray | None]:
        return {
            "last_action": np.asarray(self.last_action, dtype=np.float32).copy(),
            "gait_phase": self.gait_phase.copy(),
            "last_obs": None if self._last_obs is None else self._last_obs.copy(),
        }

    def restore_state(self, state: dict[str, np.ndarray | None]):
        self.last_action = np.asarray(state["last_action"], dtype=np.float32).copy()
        self.gait_phase = np.asarray(state["gait_phase"], dtype=np.float32).copy()
        last_obs = state.get("last_obs")
        self._last_obs = None if last_obs is None else np.asarray(last_obs, dtype=np.float32).copy()

    def post_step_callback(self, commands: list[str] | None = None):
        if self.freeze_phase_during_dry_run and commands and "[UNILAB_FREEZE_PHASE]" in commands:
            return
        self.gait_phase = (
            self.gait_phase + 2.0 * np.pi * self.gait_frequency * self.dt
        ) % (2.0 * np.pi)

    def _get_commands(self, ctrl_data) -> np.ndarray:
        commands = np.zeros(3, dtype=np.float32)
        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                lx = axes["LeftX"]
                ly = axes["LeftY"]
                rx = axes["RightX"]
                commands[0] = command_remap(ly, self.command_maps[0])
                commands[1] = command_remap(lx, self.command_maps[1])
                commands[2] = command_remap(rx, self.command_maps[2])
                break

            if key in ["KeyboardCtrl"]:
                keys = ctrl_data[key]["keyboard_event"]
                for event in keys:
                    if event["type"] != "keyboard":
                        continue
                    value = event["pressed"] * 1.5
                    if event["name"] == "w":
                        commands[0] = command_remap(value, self.command_maps[0])
                    elif event["name"] == "s":
                        commands[0] = command_remap(-value, self.command_maps[0])
                    elif event["name"] == "a":
                        commands[1] = command_remap(-value, self.command_maps[1])
                    elif event["name"] == "d":
                        commands[1] = command_remap(value, self.command_maps[1])
                    elif event["name"] == "e":
                        commands[2] = command_remap(value, self.command_maps[2])
                    elif event["name"] == "q":
                        commands[2] = command_remap(-value, self.command_maps[2])
                break
        return commands

    def get_observation(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data)
        torso_ang_vel = getattr(env_data, "torso_ang_vel", None)
        torso_quat = getattr(env_data, "torso_quat", None)
        if self.use_torso_obs_source and torso_ang_vel is not None and torso_quat is not None:
            ang_vel = np.asarray(torso_ang_vel, dtype=np.float32)
            gravity = get_gravity_orientation(np.asarray(torso_quat, dtype=np.float32)).astype(np.float32)
            obs_source = "torso"
        else:
            ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)
            gravity = get_gravity_orientation(env_data.base_quat).astype(np.float32)
            obs_source = "base"
        dof_pos_rel = np.asarray(env_data.dof_pos - self.default_dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)

        obs = np.concatenate(
            [
                ang_vel * 0.25,
                gravity,
                dof_pos_rel,
                dof_vel * 0.05,
                np.asarray(self.last_action, dtype=np.float32),
                commands,
                self.gait_phase.astype(np.float32),
            ],
        ).astype(np.float32)

        if obs.shape[0] != self.expected_obs_dim:
            raise ValueError(
                f"UniLab obs dim {obs.shape[0]} != expected {self.expected_obs_dim}"
            )
        self._last_obs = obs.copy()
        extras = {
            "commands": commands,
            "gait_phase": self.gait_phase.copy(),
            "unilab_obs_dim": obs.shape[0],
            "obs_source": obs_source,
        }
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self._runtime["kind"] == "onnx":
            session = self._runtime["session"]
            ort_inputs = {
                self._runtime["input_name"]: np.expand_dims(obs, axis=0).astype(np.float32)
            }
            ort_outputs = session.run([self._runtime["output_name"]], ort_inputs)
            actions = np.asarray(ort_outputs[0]).squeeze().astype(np.float32)
        else:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                actions = self._runtime["model"](obs_tensor).detach().cpu().numpy().squeeze()
            actions = np.asarray(actions, dtype=np.float32)

        if actions.shape[0] != self.expected_action_dim:
            raise ValueError(
                f"UniLab action dim {actions.shape[0]} != expected {self.expected_action_dim}"
            )

        actions = (1.0 - self.action_beta) * np.asarray(self.last_action) + self.action_beta * actions
        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)
        actions = actions * self.action_scale
        self.last_action = actions.copy()
        return actions
