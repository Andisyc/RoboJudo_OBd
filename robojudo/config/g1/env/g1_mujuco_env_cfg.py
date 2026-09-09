from typing import Literal

from robojudo.config import ASSETS_DIR
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.tools.tool_cfgs import ForwardKinematicCfg

from .g1_env_cfg import G1_12EnvCfg, G1_23EnvCfg, G1EnvCfg


class G1MujocoEnvCfg(G1EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======

    update_with_fk: bool = True


class G1UniLabMujocoEnvCfg(G1MujocoEnvCfg):
    """MuJoCo contract matching the G1 model used by UniLab training."""

    xml: str = (ASSETS_DIR / "robots/g1/scene_unilab_flat.xml").as_posix()
    sim_dt: float = 0.02 / 3.0
    sim_decimation: int = 3
    reset_keyframe: int | None = 0
    actuator_control_mode: Literal["position"] = "position"
    policy_gyro_sensor: str | None = "torso_gyro"
    policy_upvector_sensor: str | None = "torso_upvector"
    # The policy consumes the MuJoCo torso sensors and joint state directly.
    forward_kinematic: ForwardKinematicCfg | None = None
    update_with_fk: bool = False


class G1_23MujocoEnvCfg(G1_23EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = True


class G1_12MujocoEnvCfg(G1_12EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = False
