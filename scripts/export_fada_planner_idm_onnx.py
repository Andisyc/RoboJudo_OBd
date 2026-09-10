from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robojudo.config import ASSETS_DIR
from robojudo.policy.fada.checkpoint import load_fada_policy_checkpoint
from robojudo.policy.fada.onnx_runtime import FADAOnnxRuntime


class ActionOnlyPlannerIDM(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        # Export the exact forward graph without the Python-only shape guards in
        # the training modules, which cannot be represented by ONNX tracing.
        planner = self.policy.planner
        command_token = planner.command_embedding(command).unsqueeze(1)
        planner_tokens = planner.observation_embedding(observation_history) + command_token
        planner_tokens = planner_tokens + planner.position.embedding
        encoded = planner.encoder(planner_tokens)
        residual = planner.future_head(encoded[:, -1]).reshape(
            -1, planner.config.prediction_horizon, planner.config.obs_dim
        )
        future = observation_history[:, -1:].expand(
            -1, planner.config.prediction_horizon, -1
        ) + residual

        idm = self.policy.idm
        history_tokens = idm.observation_embedding(observation_history)
        history_tokens = history_tokens + idm.action_embedding(action_history)
        history_tokens = history_tokens + idm.history_position.embedding
        memory = idm.history_encoder(history_tokens)
        future_tokens = idm.future_embedding(future) + idm.future_position.embedding
        latent = idm.future_decoder(tgt=future_tokens, memory=memory)
        return idm.action_head(latent)[:, 0]


def parse_args():
    model_dir = ASSETS_DIR / "models/g1/fada/planner_idm_v022"
    parser = argparse.ArgumentParser(description="Export FADA Planner-IDM to ONNX")
    parser.add_argument(
        "--input",
        type=Path,
        default=model_dir / "planner_idm_close_some_dr_v001.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=model_dir / "planner_idm_close_some_dr_v001.onnx",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}; pass --force to replace it")

    loaded = load_fada_policy_checkpoint(args.input, device="cpu")
    policy = loaded.policy.eval()
    config = policy.config
    wrapper = ActionOnlyPlannerIDM(policy).eval()
    torch.backends.mha.set_fastpath_enabled(False)

    generator = torch.Generator().manual_seed(0)
    observation_history = torch.randn(
        1, config.history_length, config.obs_dim, generator=generator
    )
    action_history = torch.randn(
        1, config.history_length, config.action_dim, generator=generator
    )
    command = torch.randn(1, config.command_dim, generator=generator)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch_action = wrapper(observation_history, action_history, command)
        torch.onnx.export(
            wrapper,
            (observation_history, action_history, command),
            args.output.as_posix(),
            input_names=list(FADAOnnxRuntime.INPUT_NAMES),
            output_names=[FADAOnnxRuntime.OUTPUT_NAME],
            opset_version=17,
            do_constant_folding=True,
        )

    onnx_runtime = FADAOnnxRuntime(
        args.output,
        device="cpu",
        history_length=config.history_length,
        observation_dim=config.obs_dim,
        action_dim=config.action_dim,
        command_dim=config.command_dim,
    )
    onnx_action = onnx_runtime.infer(
        observation_history, action_history, command
    )
    np.testing.assert_allclose(
        onnx_action.numpy(), torch_action.numpy(), rtol=1.0e-4, atol=1.0e-4
    )
    print(f"Exported and verified: {args.output}")


if __name__ == "__main__":
    main()
