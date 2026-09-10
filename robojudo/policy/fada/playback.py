from __future__ import annotations

from typing import Any

import torch

from .model import FADAPlannerIDMPolicy
from .observation import project_fada_observation_tensor


class FADAPlaybackController:
    def __init__(self, policy: FADAPlannerIDMPolicy, *, device: str | torch.device) -> None:
        self.policy = policy
        self.config = policy.config
        self.device = torch.device(device)
        self._observation_history: torch.Tensor | None = None
        self._action_history: torch.Tensor | None = None

    def reset(self) -> None:
        self._observation_history = None
        self._action_history = None

    def snapshot_state(self) -> dict[str, torch.Tensor | None]:
        return {
            "observation_history": (
                None
                if self._observation_history is None
                else self._observation_history.detach().clone()
            ),
            "action_history": (
                None if self._action_history is None else self._action_history.detach().clone()
            ),
        }

    def restore_state(self, state: dict[str, torch.Tensor | None]) -> None:
        observation_history = state.get("observation_history")
        action_history = state.get("action_history")
        self._observation_history = (
            None if observation_history is None else observation_history.detach().clone()
        )
        self._action_history = None if action_history is None else action_history.detach().clone()

    @torch.no_grad()
    def act(self, observation: Any, command: Any) -> torch.Tensor:
        raw = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        obs = project_fada_observation_tensor(
            raw, observation_contract=self.config.observation_contract
        )
        cmd = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if cmd.ndim == 1:
            cmd = cmd.unsqueeze(0)
        expected_command_shape = (obs.shape[0], self.config.command_dim)
        if tuple(cmd.shape) != expected_command_shape or not bool(torch.isfinite(cmd).all()):
            raise ValueError(
                "FADA playback command must be finite with shape "
                f"{expected_command_shape}, got {tuple(cmd.shape)}"
            )
        self._advance_observation_history(obs)
        assert self._observation_history is not None
        assert self._action_history is not None
        action = self.policy(self._observation_history, self._action_history, cmd).action.detach()
        expected_action_shape = (obs.shape[0], self.config.action_dim)
        if tuple(action.shape) != expected_action_shape or not bool(torch.isfinite(action).all()):
            raise ValueError(
                "FADA playback action must be finite with shape "
                f"{expected_action_shape}, got {tuple(action.shape)}"
            )
        self._action_history = torch.cat(
            (self._action_history[:, 1:], action.unsqueeze(1)), dim=1
        )
        return action

    def _advance_observation_history(self, obs: torch.Tensor) -> None:
        batch_size = int(obs.shape[0])
        if self._observation_history is None:
            self._observation_history = obs.unsqueeze(1).repeat(
                1, self.config.history_length, 1
            )
            self._action_history = torch.zeros(
                batch_size,
                self.config.history_length,
                self.config.action_dim,
                device=self.device,
                dtype=obs.dtype,
            )
            return
        if self._observation_history.shape[0] != batch_size:
            raise ValueError("FADA playback batch changed without reset")
        self._observation_history = torch.cat(
            (self._observation_history[:, 1:], obs.unsqueeze(1)), dim=1
        )
