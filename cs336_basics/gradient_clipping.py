from typing import Iterable
import torch


def clip_gradients_(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, epsilon: float = 1e-6
) -> None:
    l2_norm = sum(
        p.grad.square().sum() for p in parameters if p.grad is not None
    ).sqrt()
    if l2_norm.item() > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad *= max_l2_norm / (l2_norm + epsilon)
