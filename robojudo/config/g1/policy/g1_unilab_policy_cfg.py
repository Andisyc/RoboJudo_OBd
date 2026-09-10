from __future__ import annotations

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1UniLabDoF(DoFConfig):
    joint_names: list[str] = [
        *[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ],
        *[
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        *["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        *[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        *[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    ]

    default_pos: list[float] | None = [
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],
        *[0.0, 0.0, 0.0],
        *[0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
        *[0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
    ]


class G1UniLabPolicyCfg(PolicyCfg):
    robot: str = "g1"
    policy_type: str = "UniLabPolicy"
    policy_name: str = "g1_walk_flat"
    disable_autoload: bool = True

    obs_dof: DoFConfig = G1UniLabDoF()
    action_dof: DoFConfig = obs_dof

    freq: int = 50
    action_scale: float = 1.0
    action_clip: float | None = None
    action_beta: float = 1.0

    expected_obs_dim: int = 98
    expected_action_dim: int = 29
    gait_frequency: float = 1.5
    initial_gait_phase: list[float] = [0.0, 3.141592653589793]
    use_torso_obs_source: bool = True

    # Joystick axes are remapped to UniLab's physical command ranges.
    command_maps: list[list[float]] = [
        [-0.6, 0.0, 1.0],
        [0.4, 0.0, -0.4],
        [0.8, 0.0, -0.8],
    ]
    freeze_phase_during_dry_run: bool = True
    debug_checks: bool = True

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/unilab/{self.policy_name}/policy.onnx"
        return policy_file.as_posix()
