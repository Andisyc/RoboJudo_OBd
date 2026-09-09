"""Export a UniLab distillation checkpoint as a RoboJuDo TorchScript policy.

Run this script in the UniLab environment so its model definitions are
available. The exported module contains the deployment-time command-intent
routing used by UniLab interactive playback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from unilab.algos.torch.distill import load_distillation_student_policy


class CommandRoutedStudent(nn.Module):
    def __init__(self, student: nn.Module, *, command_start: int = 93):
        super().__init__()
        if not hasattr(student, "experts") or len(student.experts) != 2:
            raise ValueError("RoboJuDo export requires a two-expert UniLab MoE student")
        self.expert_active = student.experts[0]
        self.expert_inactive = student.experts[1]
        self.command_start = int(command_start)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        commands = obs[:, self.command_start : self.command_start + 3]
        active = (torch.linalg.vector_norm(commands[:, :2], dim=1) > 0.05) | (
            torch.abs(commands[:, 2]) > 0.05
        )
        active_action = self.expert_active(obs)
        inactive_action = self.expert_inactive(obs)
        return torch.where(active[:, None], active_action, inactive_action)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    loaded = load_distillation_student_policy(args.checkpoint, device="cpu")
    if (loaded.obs_dim, loaded.action_dim) != (99, 29):
        raise ValueError(
            "Expected the G1 stand-height/walk 99x29 contract, got "
            f"{loaded.obs_dim}x{loaded.action_dim}"
        )

    module = CommandRoutedStudent(loaded.policy).eval()
    example = torch.zeros((1, loaded.obs_dim), dtype=torch.float32)
    traced = torch.jit.trace(module, example, check_trace=False)

    checks = [example, example.clone()]
    checks[1][:, 93] = 0.4
    with torch.no_grad():
        for check in checks:
            torch.testing.assert_close(traced(check), module(check))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    print(f"Exported {args.checkpoint} -> {args.output}")


if __name__ == "__main__":
    main()
