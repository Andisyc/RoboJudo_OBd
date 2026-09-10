from __future__ import annotations

import numpy as np

from robojudo.environment import env_registry

from .unitree_cpp_env import UnitreeCppEnv


class FADATorsoImuProjector:
    """Project Unitree torso IMU data into the FADA training convention."""

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
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def project(self, imu_state) -> tuple[np.ndarray, np.ndarray]:
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


@env_registry.register
class FADAUnitreeCppEnv(UnitreeCppEnv):
    """Unitree environment exposing only FADA's torso-IMU observation contract."""

    def __init__(self, cfg_env, device="cpu"):
        self._fada_imu = FADATorsoImuProjector()
        self._policy_gyro: np.ndarray | None = None
        self._policy_gravity: np.ndarray | None = None
        super().__init__(cfg_env=cfg_env, device=device)

    def reset(self):
        self._fada_imu.reset()
        super().reset()

    @staticmethod
    def _check_torso_imu_sample(imu_state):
        accelerometer = np.asarray(imu_state.accelerometer, dtype=np.float32)
        if accelerometer.shape != (3,) or not bool(np.isfinite(accelerometer).all()):
            raise RuntimeError(
                "Planner-IDM torso IMU accelerometer must be finite with shape (3,)"
            )
        if float(np.linalg.norm(accelerometer)) < 1.0e-3:
            raise RuntimeError("Planner-IDM has not received a valid torso IMU sample")

    def update(self):
        super().update()
        torso_imu_state = getattr(self.robot_state, "torso_imu_state", None)
        if torso_imu_state is None:
            raise RuntimeError(
                "Planner-IDM torso IMU requires unitree_cpp>=1.0.4 "
                "with RobotState.torso_imu_state"
            )
        self._check_torso_imu_sample(torso_imu_state)
        self._policy_gyro, self._policy_gravity = self._fada_imu.project(torso_imu_state)

    def get_data(self):
        env_data = super().get_data()
        env_data["policy_gyro"] = self._policy_gyro
        env_data["policy_gravity"] = self._policy_gravity
        return env_data
