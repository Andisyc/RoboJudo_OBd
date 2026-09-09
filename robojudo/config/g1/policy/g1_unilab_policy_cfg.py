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


class G1UniLabDistillDoF(G1UniLabDoF):
    """Control gains from the UniLab G1 MuJoCo training actuator model."""

    stiffness: list[float] = [
        *[40.179, 99.098, 40.179, 99.098, 28.501, 28.501],
        *[40.179, 99.098, 40.179, 99.098, 28.501, 28.501],
        *[40.179, 28.501, 28.501],
        *[14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778],
        *[14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778],
    ]
    damping: list[float] = [
        *[2.558, 6.309, 2.558, 6.309, 1.814, 1.814],
        *[2.558, 6.309, 2.558, 6.309, 1.814, 1.814],
        *[2.558, 1.814, 1.814],
        *[0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068],
        *[0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068],
    ]
    torque_limits: list[float] = [
        *[88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
        *[88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
        *[88.0, 50.0, 50.0],
        *[25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0],
        *[25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0],
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
    # UniLab's offset_phase reset contract starts the two feet half a cycle
    # apart. Starting both phases at zero makes the walking expert command both
    # legs into the same gait phase.
    initial_gait_phase: list[float] = [0.0, 3.141592653589793]

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


class G1UniLabDistillPolicyCfg(G1UniLabPolicyCfg):
    """UniLab stand-height/walk DAgger student deployment contract."""

    policy_type: str = "UniLabDistillPolicy"
    policy_name: str = "stand_height_walk_ordered_b_r4_dagger_iteration_1"

    expected_obs_dim: int = 99
    expected_action_dim: int = 29

    # A full-scale keyboard press previously requested 1.0 m/s forward, which
    # is outside the 0.4 command magnitude used by this distilled checkpoint's
    # DAgger transition data. Keep joystick behavior unchanged and scale only
    # the digital keyboard input.
    keyboard_input_scale: float = 0.4
    command_xy_threshold: float = 0.05
    command_yaw_threshold: float = 0.05
    stand_gait_phase: list[float] = [3.141592653589793, 3.141592653589793]

    obs_dof: DoFConfig = G1UniLabDistillDoF()
    action_dof: DoFConfig = obs_dof

    default_target_height: float = 0.754
    min_target_height: float = 0.650
    max_target_height: float = 0.754
    height_command_step: float = 0.010
    height_up_key: str = "r"
    height_down_key: str = "f"
    # RightY down lowers the target; neutral/up retains the nominal height.
    height_command_map: list[float] = [0.65, 0.754, 0.754]

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/unilab/{self.policy_name}/policy.pt"
        return policy_file.as_posix()
