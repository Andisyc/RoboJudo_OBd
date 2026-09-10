from __future__ import annotations

import numpy as np

from robojudo.policy import policy_registry

from .fada.checkpoint import load_fada_policy_checkpoint
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
        self.keyboard_command_magnitude = float(cfg_policy.keyboard_command_magnitude)
        if self.keyboard_command_magnitude <= 0.0:
            raise ValueError("FADA keyboard_command_magnitude must be positive")
        self._held_motion_keys: set[str] = set()
        self._keyboard_command = np.zeros(3, dtype=np.float32)
        self.playback_controller = FADAPlaybackController(
            self._runtime["model"], device=self.device
        )
        self._torso_imu_projector = FADATorsoImuProjector()
        self._pending_command: np.ndarray | None = None
        self._pending_observation: np.ndarray | None = None
        self.reset()

    def _build_runtime(self, policy_file: str):
        loaded = load_fada_policy_checkpoint(policy_file, device=self.device)
        return {
            "kind": "fada_planner_idm",
            "model": loaded.policy,
            "checkpoint": loaded.checkpoint,
        }

    def _check_runtime_contract(self):
        config = self._runtime["model"].config
        observed = (
            config.obs_dim,
            config.action_dim,
            config.command_dim,
            config.history_length,
            config.prediction_horizon,
            config.observation_contract,
        )
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
        press toggles its axis between the FADA v022 nominal walking speed and
        zero; release events only clear debounce state.  This makes a tap
        start/stop walking while ignoring operating-system key-repeat events.
        """
        keyboard_data = ctrl_data.get("KeyboardCtrl")
        if keyboard_data is None:
            return super()._get_commands(ctrl_data)

        key_axes = {
            "w": (0, 1.0), "s": (0, -1.0),
            "a": (1, 1.0), "d": (1, -1.0),
            "q": (2, 1.0), "e": (2, -1.0),
        }
        for event in keyboard_data.get("keyboard_event", []):
            if event.get("type") != "keyboard":
                continue
            name = event.get("name")
            if name not in key_axes:
                continue
            if event.get("pressed", False):
                if name in self._held_motion_keys:
                    continue
                self._held_motion_keys.add(name)
                axis, sign = key_axes[name]
                target = sign * self.keyboard_command_magnitude
                self._keyboard_command[axis] = (
                    0.0 if np.isclose(self._keyboard_command[axis], target) else target
                )
            else:
                self._held_motion_keys.discard(name)
        return self._keyboard_command.copy()

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
