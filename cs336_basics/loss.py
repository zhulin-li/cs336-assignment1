import torch
from einops import reduce, rearrange


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits -= reduce(logits, "... d -> ... 1", "max")
    sum_of_exp = reduce(torch.exp(logits), "... d -> ... 1", "sum")
    targets = rearrange(targets, "... -> ... 1")
    loss = torch.log(sum_of_exp) - torch.gather(logits, dim=-1, index=targets)
    return torch.mean(loss)
