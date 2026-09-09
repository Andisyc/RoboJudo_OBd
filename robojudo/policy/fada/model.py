from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .observation import (
    FADA_G1_ACTION_DIM,
    FADA_G1_COMMAND_DIM,
    FADA_G1_STATE_DIM,
    FADA_G1_STATE_OBSERVATION_CONTRACT,
)


@dataclass(frozen=True)
class FADAArchitectureConfig:
    obs_dim: int
    action_dim: int
    command_dim: int
    observation_contract: str
    history_length: int = 30
    prediction_horizon: int = 6
    hidden_dim: int = 128
    num_heads: int = 4
    planner_layers: int = 3
    idm_encoder_layers: int = 3
    idm_decoder_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "command_dim": self.command_dim,
            "history_length": self.history_length,
            "prediction_horizon": self.prediction_horizon,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "planner_layers": self.planner_layers,
            "idm_encoder_layers": self.idm_encoder_layers,
            "idm_decoder_layers": self.idm_decoder_layers,
            "feedforward_dim": self.feedforward_dim,
        }
        for name, value in integer_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        expected = (FADA_G1_STATE_DIM, FADA_G1_ACTION_DIM, FADA_G1_COMMAND_DIM)
        observed = (int(self.obs_dim), int(self.action_dim), int(self.command_dim))
        if self.observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
            raise ValueError(
                f"unsupported FADA observation_contract: {self.observation_contract!r}"
            )
        if observed != expected:
            raise ValueError(f"g1_fada_state_v2 requires dimensions {expected}, got {observed}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class PlannerIDMOutput:
    predicted_future: torch.Tensor
    action_chunk: torch.Tensor
    action: torch.Tensor


def _validate_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_sequence(
    name: str,
    tensor: torch.Tensor,
    *,
    length: int,
    feature_dim: int,
) -> None:
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (length, feature_dim):
        raise ValueError(
            f"{name} shape mismatch: expected [batch, {length}, {feature_dim}], "
            f"got {tuple(tensor.shape)}"
        )
    _validate_finite(name, tensor)


def _validate_matrix(name: str, tensor: torch.Tensor, *, feature_dim: int) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != feature_dim:
        raise ValueError(
            f"{name} shape mismatch: expected [batch, {feature_dim}], got {tuple(tensor.shape)}"
        )
    _validate_finite(name, tensor)


class _LearnedPositionalEncoding(nn.Module):
    def __init__(self, *, length: int, hidden_dim: int) -> None:
        super().__init__()
        self.length = int(length)
        self.embedding = nn.Parameter(torch.empty(1, self.length, int(hidden_dim)))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != self.length:
            raise ValueError(
                f"positional length mismatch: expected {self.length}, got {tokens.shape[1]}"
            )
        return tokens + self.embedding.to(dtype=tokens.dtype)


def _encoder_layer(config: FADAArchitectureConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.hidden_dim,
        nhead=config.num_heads,
        dim_feedforward=config.feedforward_dim,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
    )


class FADAPlanner(nn.Module):
    def __init__(self, config: FADAArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_embedding = nn.Linear(config.obs_dim, config.hidden_dim)
        self.command_embedding = nn.Linear(config.command_dim, config.hidden_dim)
        self.position = _LearnedPositionalEncoding(
            length=config.history_length, hidden_dim=config.hidden_dim
        )
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(config),
            num_layers=config.planner_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.future_head = nn.Linear(
            config.hidden_dim, config.prediction_horizon * config.obs_dim
        )

    def forward(self, observation_history: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        _validate_sequence(
            "observation_history",
            observation_history,
            length=self.config.history_length,
            feature_dim=self.config.obs_dim,
        )
        _validate_matrix("command", command, feature_dim=self.config.command_dim)
        if observation_history.shape[0] != command.shape[0]:
            raise ValueError("Planner observation_history and command batch sizes must match")
        command_token = self.command_embedding(command).unsqueeze(1)
        tokens = self.observation_embedding(observation_history) + command_token
        encoded = self.encoder(self.position(tokens))
        residual = self.future_head(encoded[:, -1]).reshape(
            observation_history.shape[0],
            self.config.prediction_horizon,
            self.config.obs_dim,
        )
        return observation_history[:, -1:].expand(-1, self.config.prediction_horizon, -1) + residual


class FADAInverseDynamicsModel(nn.Module):
    def __init__(self, config: FADAArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_embedding = nn.Linear(config.obs_dim, config.hidden_dim)
        self.action_embedding = nn.Linear(config.action_dim, config.hidden_dim)
        self.history_position = _LearnedPositionalEncoding(
            length=config.history_length, hidden_dim=config.hidden_dim
        )
        self.history_encoder = nn.TransformerEncoder(
            _encoder_layer(config),
            num_layers=config.idm_encoder_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.future_embedding = nn.Linear(config.obs_dim, config.hidden_dim)
        self.future_position = _LearnedPositionalEncoding(
            length=config.prediction_horizon, hidden_dim=config.hidden_dim
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.future_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.idm_decoder_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

    def encode_latent(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        _validate_sequence(
            "observation_history",
            observation_history,
            length=self.config.history_length,
            feature_dim=self.config.obs_dim,
        )
        _validate_sequence(
            "action_history",
            action_history,
            length=self.config.history_length,
            feature_dim=self.config.action_dim,
        )
        _validate_sequence(
            "future",
            future,
            length=self.config.prediction_horizon,
            feature_dim=self.config.obs_dim,
        )
        if len({observation_history.shape[0], action_history.shape[0], future.shape[0]}) != 1:
            raise ValueError("IDM observation, action, and future batch sizes must match")
        history_tokens = self.observation_embedding(observation_history)
        history_tokens = history_tokens + self.action_embedding(action_history)
        memory = self.history_encoder(self.history_position(history_tokens))
        future_tokens = self.future_position(self.future_embedding(future))
        return self.future_decoder(tgt=future_tokens, memory=memory)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        _validate_sequence(
            "latent",
            latent,
            length=self.config.prediction_horizon,
            feature_dim=self.config.hidden_dim,
        )
        return self.action_head(latent)

    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode_latent(self.encode_latent(observation_history, action_history, future))


class FADAPlannerIDMPolicy(nn.Module):
    def __init__(self, config: FADAArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.planner = FADAPlanner(config)
        self.idm = FADAInverseDynamicsModel(config)

    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> PlannerIDMOutput:
        predicted_future = self.planner(observation_history, command)
        action_chunk = self.idm(observation_history, action_history, predicted_future)
        return PlannerIDMOutput(
            predicted_future=predicted_future,
            action_chunk=action_chunk,
            action=action_chunk[:, 0],
        )

    @torch.no_grad()
    def explore(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        del deterministic
        return self(observation_history, action_history, command).action

