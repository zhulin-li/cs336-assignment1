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


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device}

        assert d_k % 2 == 0
        exponent = torch.arange(0, d_k, 2, **factory_kwargs) / d_k
        numerator = rearrange(
            torch.arange(max_seq_len, **factory_kwargs), "pos -> pos 1"
        )
        denominator = rearrange(theta**exponent, "d_over_2 -> 1 d_over_2")
        angle = numerator / denominator

        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: ..., seq_len, d_k
        # token_positions: ..., seq_len
        x = rearrange(
            x, "... seq_len (d_over_2 pair) -> ... seq_len d_over_2 pair", pair=2
        )
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        even = einsum(
            x[..., 0],
            cos,
            "... seq_len d_over_2, ... seq_len d_over_2 -> ... seq_len d_over_2",
        ) - einsum(
            x[..., 1],
            sin,
            "... seq_len d_over_2, ... seq_len d_over_2 -> ... seq_len d_over_2",
        )
        odd = einsum(
            x[..., 0],
            sin,
            "... seq_len d_over_2, ... seq_len d_over_2 -> ... seq_len d_over_2",
        ) + einsum(
            x[..., 1],
            cos,
            "... seq_len d_over_2, ... seq_len d_over_2 -> ... seq_len d_over_2",
        )
        x = rearrange(
            [even, odd],
            "pair ... seq_len d_over_2 -> ... seq_len (d_over_2 pair)",
            pair=2,
        )
        return x


class SoftMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        x -= torch.max(x, dim=dim, keepdim=True).values
        x = torch.exp(x)
        x /= torch.sum(x, dim=dim, keepdim=True)
        return x
