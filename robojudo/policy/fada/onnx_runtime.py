from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class FADAInferenceRuntime(Protocol):
    def infer(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor: ...


class FADATorchRuntime:
    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def infer(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy(observation_history, action_history, command).action.detach()


class FADAOnnxRuntime:
    INPUT_NAMES = ("observation_history", "action_history", "command")
    OUTPUT_NAME = "action"

    def __init__(
        self,
        policy_file: str | Path,
        *,
        device: str | torch.device,
        history_length: int,
        observation_dim: int,
        action_dim: int,
        command_dim: int,
    ):
        path = Path(policy_file)
        if not path.is_file():
            raise FileNotFoundError(f"FADA ONNX policy file not found: {path}")

        import onnxruntime as ort

        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            path.as_posix(),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        inputs = {entry.name: entry for entry in self.session.get_inputs()}
        outputs = {entry.name: entry for entry in self.session.get_outputs()}
        if tuple(inputs) != self.INPUT_NAMES or tuple(outputs) != (self.OUTPUT_NAME,):
            raise ValueError(
                "FADA ONNX names mismatch: "
                f"inputs={tuple(inputs)}, outputs={tuple(outputs)}"
            )

        expected_tails = {
            "observation_history": (int(history_length), int(observation_dim)),
            "action_history": (int(history_length), int(action_dim)),
            "command": (int(command_dim),),
        }
        for name, tail in expected_tails.items():
            observed_tail = tuple(inputs[name].shape[1:])
            if observed_tail != tail:
                raise ValueError(
                    f"FADA ONNX input {name} shape mismatch: expected [batch, {tail}], "
                    f"got {tuple(inputs[name].shape)}"
                )

    def infer(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        feed = {
            "observation_history": observation_history.detach().cpu().numpy(),
            "action_history": action_history.detach().cpu().numpy(),
            "command": command.detach().cpu().numpy(),
        }
        result = self.session.run([self.OUTPUT_NAME], feed)
        action = np.asarray(result[0], dtype=np.float32)
        expected_shape = (observation_history.shape[0], self.action_dim)
        if action.shape != expected_shape or not bool(np.isfinite(action).all()):
            raise ValueError(
                f"FADA ONNX action must be finite with shape {expected_shape}, got {action.shape}"
            )
        return torch.from_numpy(action).to(self.device)
