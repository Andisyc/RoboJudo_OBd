from __future__ import annotations

from robojudo.config import ASSETS_DIR
from robojudo.tools.tool_cfgs import DoFConfig

from .g1_unilab_policy_cfg import G1UniLabDistillDoF, G1UniLabPolicyCfg


class G1FADAPlannerIDMPolicyCfg(G1UniLabPolicyCfg):
    """Native FADA Planner-IDM deployment contract."""

    policy_type: str = "FADAPlannerIDMPolicyAdapter"
    policy_name: str = "planner_idm_v022"
    checkpoint_filename: str = "planner_idm_close_some_dr_v001.onnx"
    expected_obs_dim: int = 98
    expected_action_dim: int = 29
    history_length: int = 30
    prediction_horizon: int = 6
    observation_contract: str = "g1_fada_state_v2"
    gait_phase_enabled: bool = True
    fixed_gait_phase: list[float] = [0.0, 3.141592653589793]
    freeze_phase_during_dry_run: bool = False
    preserve_state_during_dry_run: bool = False
    # Keep keyboard commands inside the close_some_dr_v001 training ranges.
    # Forward uses the curriculum's walking command; lateral and yaw use the
    # source task limits.
    keyboard_command_magnitudes: list[float] = [0.6, 0.4, 0.8]
    # v022 inherits the FADA distill MuJoCo control contract, which maps raw
    # action chunks directly to joint-position offsets.
    execution_action_scale: float = 1.0
    action_scale: float = 1.0
    action_beta: float = 1.0
    action_clip: float | None = None
    obs_dof: DoFConfig = G1UniLabDistillDoF()
    action_dof: DoFConfig = obs_dof

    @property
    def policy_file(self) -> str:
        return (
            ASSETS_DIR
            / f"models/{self.robot}/fada/{self.policy_name}/{self.checkpoint_filename}"
        ).as_posix()
