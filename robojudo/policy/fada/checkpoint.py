from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .model import FADAArchitectureConfig, FADAPlannerIDMPolicy

FADA_CHECKPOINT_SCHEMA_VERSION = 5
FADA_TRAINING_SCHEDULES = {
    "idm_pretrain",
    "alternating_idm_then_planner",
    "planner_from_idm",
}


@dataclass(frozen=True)
class LoadedFADAPlannerIDMPolicy:
    policy: FADAPlannerIDMPolicy
    checkpoint: Mapping[str, Any]


def _canonical_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to(device="cpu").contiguous()
        identity = f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode("ascii")
        digest.update(len(identity).to_bytes(8, "big"))
        digest.update(identity)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _load_architecture_config(payload: Mapping[str, Any]) -> FADAArchitectureConfig:
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("FADA checkpoint architecture must be a mapping")
    if payload.get("schema_version") == FADA_CHECKPOINT_SCHEMA_VERSION:
        if "observation_contract" not in architecture:
            raise ValueError("FADA checkpoint architecture must contain observation_contract")
    try:
        return FADAArchitectureConfig(**architecture)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid FADA checkpoint architecture: {architecture}") from exc


def _validate_payload(payload: Any) -> tuple[Mapping[str, Any], FADAArchitectureConfig]:
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2, 3, 4, 5}:
        raise ValueError("unsupported or malformed FADA checkpoint schema")
    config = _load_architecture_config(payload)
    planner_state = payload.get("planner_state_dict")
    idm_state = payload.get("idm_state_dict")
    if not isinstance(planner_state, dict) or not isinstance(idm_state, dict):
        raise ValueError("FADA checkpoint must contain Planner and IDM state dictionaries")
    if payload.get("schema_version") in {4, 5}:
        if payload.get("idm_sha256") != _canonical_state_dict_sha256(idm_state):
            raise ValueError("FADA checkpoint IDM identity mismatch")
    if payload.get("schema_version") == 5:
        if str(payload.get("training_schedule")) not in FADA_TRAINING_SCHEDULES:
            raise ValueError("schema-5 FADA checkpoint has an invalid training schedule")
        if any(
            not isinstance(payload.get(name), dict)
            for name in ("idm_optimizer_state_dict", "planner_optimizer_state_dict")
        ):
            raise ValueError("schema-5 FADA checkpoint requires both optimizer states")
    return payload, config


def load_fada_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedFADAPlannerIDMPolicy:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    payload, config = _validate_payload(payload)
    policy = FADAPlannerIDMPolicy(config).to(device)
    policy.planner.load_state_dict(payload["planner_state_dict"], strict=True)
    policy.idm.load_state_dict(payload["idm_state_dict"], strict=True)
    policy.eval()
    return LoadedFADAPlannerIDMPolicy(policy=policy, checkpoint=payload)

