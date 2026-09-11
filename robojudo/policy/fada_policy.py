from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from robojudo.policy import policy_registry

from .fada.checkpoint import load_fada_policy_checkpoint
from .fada.onnx_runtime import FADAOnnxRuntime, FADATorchRuntime
from .fada.observation import (
    FADATorsoImuProjector,
    FADA_G1_ACTION_DIM,
    FADA_G1_ACTOR_OBS_DIM,
    FADA_G1_COMMAND_DIM,
    FADA_G1_STATE_DIM,
    FADA_G1_STATE_OBSERVATION_CONTRACT,
)
from .fada.playback import FADAPlaybackController
from .unilab_policy import UniLabPolicy
from robojudo.tools.trajectory_trigger import request_trajectory_start

logger = logging.getLogger(__name__)


@policy_registry.register
class FADAPlannerIDMPolicyAdapter(UniLabPolicy):
    """RoboJuDo runtime adapter for a native FADA Planner-IDM checkpoint."""

    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)
        self.gait_phase_enabled = bool(cfg_policy.gait_phase_enabled)
        self.fixed_gait_phase = np.asarray(cfg_policy.fixed_gait_phase, dtype=np.float32)
        if self.fixed_gait_phase.shape != (2,):
            raise ValueError("FADA fixed_gait_phase must contain two values")
        self.execution_action_scale = float(cfg_policy.execution_action_scale)
        if self.execution_action_scale <= 0.0:
            raise ValueError("FADA execution_action_scale must be positive")
        self.keyboard_command_magnitudes = np.asarray(
            cfg_policy.keyboard_command_magnitudes, dtype=np.float32
        )
        if (
            self.keyboard_command_magnitudes.shape != (3,)
            or not bool(np.isfinite(self.keyboard_command_magnitudes).all())
            or not bool((self.keyboard_command_magnitudes > 0.0).all())
        ):
            raise ValueError(
                "FADA keyboard_command_magnitudes must contain three positive values"
            )
        self._held_motion_keys: set[str] = set()
        self._keyboard_command = np.zeros(3, dtype=np.float32)
        self._trajectory_recording_requested = False
        self.playback_controller = FADAPlaybackController(
            self._runtime["runner"],
            device=self.device,
            history_length=int(cfg_policy.history_length),
            observation_contract=FADA_G1_STATE_OBSERVATION_CONTRACT,
            action_dim=FADA_G1_ACTION_DIM,
            command_dim=FADA_G1_COMMAND_DIM,
        )
        self._torso_imu_projector = FADATorsoImuProjector()
        self._pending_command: np.ndarray | None = None
        self._pending_observation: np.ndarray | None = None
        self.reset()

    def _build_runtime(self, policy_file: str):
        suffix = Path(policy_file).suffix.lower()
        expected_contract = (
            FADA_G1_STATE_DIM,
            FADA_G1_ACTION_DIM,
            FADA_G1_COMMAND_DIM,
            int(self.cfg_policy.history_length),
            int(self.cfg_policy.prediction_horizon),
            FADA_G1_STATE_OBSERVATION_CONTRACT,
        )
        if suffix == ".onnx":
            runner = FADAOnnxRuntime(
                policy_file,
                device=self.device,
                history_length=int(self.cfg_policy.history_length),
                observation_dim=FADA_G1_STATE_DIM,
                action_dim=FADA_G1_ACTION_DIM,
                command_dim=FADA_G1_COMMAND_DIM,
            )
            return {
                "kind": "fada_planner_idm_onnx",
                "runner": runner,
                "contract": expected_contract,
            }
        if suffix not in {".pt", ".pth"}:
            raise ValueError(f"Unsupported FADA Planner-IDM policy format: {policy_file}")
        loaded = load_fada_policy_checkpoint(policy_file, device=self.device)
        config = loaded.policy.config
        return {
            "kind": "fada_planner_idm_torch",
            "model": loaded.policy,
            "runner": FADATorchRuntime(loaded.policy),
            "checkpoint": loaded.checkpoint,
            "contract": (
                config.obs_dim,
                config.action_dim,
                config.command_dim,
                config.history_length,
                config.prediction_horizon,
                config.observation_contract,
            ),
        }

    def _check_runtime_contract(self):
        observed = self._runtime["contract"]
        expected = (
            FADA_G1_STATE_DIM,
            FADA_G1_ACTION_DIM,
            FADA_G1_COMMAND_DIM,
            int(self.cfg_policy.history_length),
            int(self.cfg_policy.prediction_horizon),
            FADA_G1_STATE_OBSERVATION_CONTRACT,
        )
        if observed != expected:
            raise ValueError(f"FADA runtime contract mismatch: expected={expected}, got={observed}")
        if self.expected_obs_dim != FADA_G1_ACTOR_OBS_DIM:
            raise ValueError("FADA adapter requires the 98-D G1 actor observation")
        if self.action_scale != 1.0 or self.action_beta != 1.0 or self.action_clip is not None:
            raise ValueError("FADA playback forbids action scaling, smoothing, and clipping")

    def reset(self):
        super().reset()
        if hasattr(self, "_held_motion_keys"):
            self._held_motion_keys.clear()
            self._keyboard_command.fill(0.0)
        if hasattr(self, "gait_phase_enabled") and not self.gait_phase_enabled:
            self.gait_phase = self.fixed_gait_phase.copy()
        if hasattr(self, "playback_controller"):
            self.playback_controller.reset()
        if hasattr(self, "_torso_imu_projector"):
            self._torso_imu_projector.reset()
        self._pending_command = None
        self._pending_observation = None

    def snapshot_state(self) -> dict[str, object]:
        state: dict[str, object] = dict(super().snapshot_state())
        state.update(
            {
                "held_motion_keys": set(self._held_motion_keys),
                "keyboard_command": self._keyboard_command.copy(),
                "pending_command": (
                    None if self._pending_command is None else self._pending_command.copy()
                ),
                "pending_observation": (
                    None
                    if self._pending_observation is None
                    else self._pending_observation.copy()
                ),
                "playback": self.playback_controller.snapshot_state(),
            }
        )
        return state

    def restore_state(self, state: dict[str, object]):
        super().restore_state(state)  # type: ignore[arg-type]
        self._held_motion_keys = set(state["held_motion_keys"])  # type: ignore[arg-type]
        self._keyboard_command = np.asarray(
            state["keyboard_command"], dtype=np.float32
        ).copy()
        pending_command = state.get("pending_command")
        self._pending_command = (
            None
            if pending_command is None
            else np.asarray(pending_command, dtype=np.float32).copy()
        )
        pending_observation = state.get("pending_observation")
        self._pending_observation = (
            None
            if pending_observation is None
            else np.asarray(pending_observation, dtype=np.float32).copy()
        )
        self.playback_controller.restore_state(state["playback"])  # type: ignore[arg-type]

    def post_step_callback(self, commands: list[str] | None = None):
        if self.gait_phase_enabled:
            super().post_step_callback(commands)

    def _get_commands(self, ctrl_data) -> np.ndarray:
        """Return persistent, in-distribution physical velocity commands.

        RoboJuDo keyboard input is delivered as press/release events.  A key
        press sets its axis to the corresponding trained command magnitude;
        release events do not stop motion because an SSH terminal cannot track
        key holds.  The x key clears all three command axes.
        """
        keyboard_data = ctrl_data.get("KeyboardCtrl")
        if keyboard_data is None:
            commands = np.asarray(super()._get_commands(ctrl_data), dtype=np.float32)
            self._maybe_start_trajectory_recording(commands)
            return commands

        key_axes: dict[str, tuple[int, float]] = {
            "w": (0, 1.0), "s": (0, -1.0),
            "a": (1, 1.0), "d": (1, -1.0),
            "q": (2, 1.0), "e": (2, -1.0),
        }
        for event in keyboard_data.get("keyboard_event", []):
            if event.get("type") != "keyboard":
                continue
            name = event.get("name")
            if name not in key_axes and name != "x":
                continue
            if not event.get("pressed", False):
                self._held_motion_keys.discard(name)
                continue

            self._held_motion_keys.add(name)
            next_command = self._keyboard_command.copy()
            if name == "x":
                next_command.fill(0.0)
            else:
                axis, sign = key_axes[name]
                next_command[axis] = sign * self.keyboard_command_magnitudes[axis]

            if np.array_equal(next_command, self._keyboard_command):
                continue
            self._keyboard_command = next_command
            logger.info(
                "[KEYBOARD] key=%s vx=%.3f vy=%.3f yaw=%.3f",
                name,
                self._keyboard_command[0],
                self._keyboard_command[1],
                self._keyboard_command[2],
            )
        commands = self._keyboard_command.copy()
        self._maybe_start_trajectory_recording(commands)
        return commands

    def _maybe_start_trajectory_recording(self, commands: np.ndarray):
        if getattr(self, "_trajectory_recording_requested", False) or float(commands[0]) <= 0.0:
            return
        self._trajectory_recording_requested = True
        request_trajectory_start()

    def get_observation(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data)
        policy_gyro = getattr(env_data, "policy_gyro", None)
        policy_gravity = getattr(env_data, "policy_gravity", None)
        if policy_gyro is None or policy_gravity is None:
            torso_imu_state = getattr(env_data, "torso_imu_state", None)
            if torso_imu_state is None:
                raise RuntimeError(
                    "FADA Planner-IDM requires MuJoCo policy IMU observations or "
                    "Unitree torso_imu_state"
                )
            policy_gyro, policy_gravity = self._torso_imu_projector.project(
                torso_imu_state
            )
        base_ang_vel = np.asarray(policy_gyro, dtype=np.float32)
        gravity = np.asarray(policy_gravity, dtype=np.float32)
        if (
            base_ang_vel.shape != (3,)
            or gravity.shape != (3,)
            or not bool(np.isfinite(base_ang_vel).all() and np.isfinite(gravity).all())
        ):
            raise RuntimeError("FADA Planner-IDM policy IMU observation must be finite 3-D vectors")
        gravity_norm = float(np.linalg.norm(gravity))
        if not 0.999 <= gravity_norm <= 1.001:
            raise RuntimeError(
                f"FADA Planner-IDM policy gravity norm is invalid: {gravity_norm:.6f}"
            )
        obs = np.concatenate(
            [
                base_ang_vel * 0.25,
                gravity,
                np.asarray(env_data.dof_pos - self.default_dof_pos, dtype=np.float32),
                np.asarray(env_data.dof_vel, dtype=np.float32) * 0.05,
                np.asarray(self.last_action, dtype=np.float32),
                commands,
                self.gait_phase.astype(np.float32),
            ]
        ).astype(np.float32)
        if obs.shape != (FADA_G1_ACTOR_OBS_DIM,) or not bool(np.isfinite(obs).all()):
            raise ValueError(f"FADA raw observation must be finite 98-D, got {obs.shape}")
        self._last_obs = obs.copy()
        self._pending_observation = obs.copy()
        self._pending_command = commands.copy()
        return obs, {
            "commands": commands,
            "gait_phase": self.gait_phase.copy(),
            "fada_raw_obs_dim": FADA_G1_ACTOR_OBS_DIM,
            "fada_projected_obs_dim": FADA_G1_STATE_DIM,
        }

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        candidate = np.asarray(obs, dtype=np.float32)
        if self._pending_observation is None or self._pending_command is None:
            raise RuntimeError("FADA get_action requires one fresh get_observation call")
        if candidate.shape != self._pending_observation.shape or not np.array_equal(
            candidate, self._pending_observation
        ):
            raise ValueError("FADA action observation does not match the pending control step")
        action = (
            self.playback_controller.act(candidate, self._pending_command)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        # The FADA actor observation and IDM history use raw policy actions.
        # Only the MuJoCo joint-position target uses UniLab's action scale.
        self.last_action = action.copy()
        self._pending_observation = None
        self._pending_command = None
        return action * self.execution_action_scale
