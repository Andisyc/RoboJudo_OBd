from __future__ import annotations

import numpy as np
import torch

FADA_G1_STATE_OBSERVATION_CONTRACT = "g1_fada_state_v2"
FADA_G1_ACTOR_OBS_DIM = 98
FADA_G1_STATE_DIM = 66
FADA_G1_ACTION_DIM = 29
FADA_G1_COMMAND_DIM = 3


class FADATorsoImuProjector:
    """Convert Unitree torso IMU samples to the FADA training convention."""

    def __init__(self):
        self._reference_yaw: float | None = None

    def reset(self):
        self._reference_yaw = None

    @staticmethod
    def _normalized_quaternion(quat_wxyz) -> np.ndarray:
        quat = np.asarray(quat_wxyz, dtype=np.float64)
        if quat.shape != (4,) or not bool(np.isfinite(quat).all()):
            raise RuntimeError(
                "Planner-IDM torso IMU quaternion must be finite with shape (4,)"
            )
        norm = float(np.linalg.norm(quat))
        if not 0.9 <= norm <= 1.1:
            raise RuntimeError(
                f"Planner-IDM torso IMU quaternion norm is invalid: {norm:.6f}"
            )
        return quat / norm

    @staticmethod
    def _yaw(quat_wxyz: np.ndarray) -> float:
        w, x, y, z = quat_wxyz
        return float(
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        )

    @staticmethod
    def _check_sample(imu_state):
        accelerometer = np.asarray(imu_state.accelerometer, dtype=np.float32)
        if accelerometer.shape != (3,) or not bool(np.isfinite(accelerometer).all()):
            raise RuntimeError(
                "Planner-IDM torso IMU accelerometer must be finite with shape (3,)"
            )
        if float(np.linalg.norm(accelerometer)) < 1.0e-3:
            raise RuntimeError("Planner-IDM has not received a valid torso IMU sample")

    def project(self, imu_state) -> tuple[np.ndarray, np.ndarray]:
        self._check_sample(imu_state)
        quat = self._normalized_quaternion(imu_state.quaternion)
        gyro = np.asarray(imu_state.gyroscope, dtype=np.float32)
        if gyro.shape != (3,) or not bool(np.isfinite(gyro).all()):
            raise RuntimeError(
                "Planner-IDM torso IMU gyroscope must be finite with shape (3,)"
            )

        if self._reference_yaw is None:
            self._reference_yaw = self._yaw(quat)

        w, x, y, z = quat
        torso_up_world = np.asarray(
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=np.float64,
        )
        c = float(np.cos(self._reference_yaw))
        s = float(np.sin(self._reference_yaw))
        torso_up_aligned = np.asarray(
            [
                c * torso_up_world[0] + s * torso_up_world[1],
                -s * torso_up_world[0] + c * torso_up_world[1],
                torso_up_world[2],
            ],
            dtype=np.float32,
        )
        return gyro.copy(), -torso_up_aligned


def project_fada_g1_state(source_obs: np.ndarray) -> np.ndarray:
    source = np.asarray(source_obs, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != FADA_G1_ACTOR_OBS_DIM:
        raise ValueError(
            "g1_fada_state_v2 requires finite rank-2 actor observations with width "
            f"{FADA_G1_ACTOR_OBS_DIM}, got {source.shape}"
        )
    if not bool(np.all(np.isfinite(source))):
        raise ValueError("g1_fada_state_v2 requires finite actor observations")
    return np.concatenate([source[:, :64], source[:, 96:98]], axis=1, dtype=np.float32)


def project_fada_observation_tensor(
    observation: torch.Tensor,
    *,
    observation_contract: str,
) -> torch.Tensor:
    if observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
        raise ValueError(f"unsupported FADA observation_contract: {observation_contract!r}")
    if observation.ndim != 2 or observation.shape[1] != FADA_G1_ACTOR_OBS_DIM:
        raise ValueError(
            "g1_fada_state_v2 playback requires rank-2 actor observations with width "
            f"{FADA_G1_ACTOR_OBS_DIM}, got {tuple(observation.shape)}"
        )
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("g1_fada_state_v2 requires finite actor observations")
    return torch.cat((observation[:, :64], observation[:, 96:98]), dim=1)
