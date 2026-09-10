from __future__ import annotations

import numpy as np
import torch

FADA_G1_STATE_OBSERVATION_CONTRACT = "g1_fada_state_v2"
FADA_G1_ACTOR_OBS_DIM = 98
FADA_G1_STATE_DIM = 66
FADA_G1_ACTION_DIM = 29
FADA_G1_COMMAND_DIM = 3


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

