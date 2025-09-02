import torch
from einops import reduce, rearrange
from typing import Iterable, Optional, Callable
import math


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits -= reduce(logits, "... d -> ... 1", "max")
    sum_of_exp = reduce(torch.exp(logits), "... d -> ... 1", "sum")
    targets = rearrange(targets, "... -> ... 1")
    loss = torch.log(sum_of_exp) - torch.gather(logits, dim=-1, index=targets)
    return torch.mean(loss)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "epsilon": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                m1 = state.get("first_moment", 0)
                m2 = state.get("second_moment", 0)
                t = state.get("t", 1)

                m1 = m1 * beta_1 + grad * (1 - beta_1)
                m2 = m2 * beta_2 + grad**2 * (1 - beta_2)

                adjusted_lr = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)
                p.data -= adjusted_lr * m1 / torch.sqrt(m2 + eps)
                p.data *= 1 - lr * weight_decay

                state["first_moment"] = m1
                state["second_moment"] = m2
                state["t"] = t + 1
        return loss
