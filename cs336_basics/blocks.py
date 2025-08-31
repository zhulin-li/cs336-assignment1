from torch import nn
import torch
from einops import einsum, reduce, rearrange
import math


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            data=torch.empty(size=(out_features, in_features), **factory_kwargs)
        )
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in, out in -> ... out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            data=torch.empty(size=(num_embeddings, embedding_dim), **factory_kwargs)
        )
        nn.init.trunc_normal_(self.weight, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gain = nn.Parameter(torch.empty(d_model, **factory_kwargs))
        nn.init.ones_(self.gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (reduce(x.square(), "b s d -> b s", "mean") + self.eps).sqrt()
        x = x / rearrange(rms, "b s -> b s 1") * rearrange(self.gain, "d -> 1 1 d")
        return x.to(x_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if d_ff is None:
            d_ff = int(d_model * 8 / 3 / 64) * 64
        self.W1 = Linear(d_model, d_ff, **factory_kwargs)
        self.W2 = Linear(d_ff, d_model, **factory_kwargs)
        self.W3 = Linear(d_model, d_ff, **factory_kwargs)

        self.silu = lambda x: x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.silu(self.W1(x)) * self.W3(x))
