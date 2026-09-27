from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tw import FACTOR_NAMES, FactorState, factor_to_signal


class CoreMLP(nn.Module):
    """Architecture-compatible base network for the pretrained core model."""

    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1000, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + 128, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(F.one_hot(t, 1000).float())

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embedding = self.time_embedding(t)
        return self.net(torch.cat([x, embedding], dim=-1))


class FactorTokenizer(nn.Module):
    """Encode one TW factor into a fixed number of attention tokens."""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_tokens: int,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.type_embedding = nn.Parameter(torch.zeros(1, num_tokens, d_model))

    def forward(self, factor: torch.Tensor) -> torch.Tensor:
        tokens = self.pool(self.proj(factor))
        return tokens.permute(0, 2, 1) + self.type_embedding


class PairInteractionTokenizer(nn.Module):
    """Fuse two factor-token streams while preserving their joint state."""

    def __init__(self, d_model: int, num_tokens: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.type_embedding = nn.Parameter(torch.zeros(1, num_tokens, d_model))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        joint = torch.cat(
            [left, right, left * right, torch.abs(left - right)], dim=-1
        )
        return self.net(joint) + self.type_embedding


class CrossAttentionCoreDenoiser(nn.Module):
    """Core denoiser conditioned on G1/G2/G3 through cross-attention.

    The original CoreMLP is kept as ``base``. The cross-attention branch is
    residual and its gate is initialized to zero, so an untrained adapter
    starts from the existing pretrained core prior.
    """

    def __init__(
        self,
        core_dim: int,
        factor_shapes: Dict[str, Sequence[int]],
        *,
        d_model: int = 128,
        num_heads: int = 4,
        tokens_per_factor: int = 4,
        gate_init: float = 0.0,
        interaction_tokens: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.core_dim = core_dim
        self.interaction_tokens = bool(interaction_tokens)
        self.factor_shapes = {
            name: tuple(factor_shapes[name])
            for name in FACTOR_NAMES
        }
        self.base = CoreMLP(core_dim)
        self.core_proj = nn.Linear(core_dim, d_model)
        self.time_proj = nn.Linear(128, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.factor_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, core_dim),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        self.tokenizers = nn.ModuleDict()
        for name in FACTOR_NAMES:
            shape = self.factor_shapes[name]
            if name == "G1":
                in_channels = shape[0] * shape[2] * shape[3]
            elif name == "G2":
                in_channels = shape[0] * shape[2] * shape[3]
            else:
                in_channels = shape[0] * shape[2] * shape[3]
            self.tokenizers[name] = FactorTokenizer(
                in_channels=in_channels,
                d_model=d_model,
                num_tokens=tokens_per_factor,
            )
        self.pair_tokenizers = nn.ModuleDict()
        if self.interaction_tokens:
            for pair_name in ("G1_G2", "G2_G3", "G3_G1"):
                self.pair_tokenizers[pair_name] = PairInteractionTokenizer(
                    d_model=d_model,
                    num_tokens=tokens_per_factor,
                )

    def factor_tokens(self, factors: FactorState) -> torch.Tensor:
        tokens = []
        for index, name in enumerate(FACTOR_NAMES):
            signal = factor_to_signal(factors[index], name)
            tokens.append(self.tokenizers[name](signal))
        if self.interaction_tokens:
            tokens.extend(
                [
                    self.pair_tokenizers["G1_G2"](tokens[0], tokens[1]),
                    self.pair_tokenizers["G2_G3"](tokens[1], tokens[2]),
                    self.pair_tokenizers["G3_G1"](tokens[2], tokens[0]),
                ]
            )
        return torch.cat(tokens, dim=1)

    def forward(
        self,
        core: torch.Tensor,
        t: torch.Tensor,
        factors: FactorState,
    ) -> torch.Tensor:
        base_prediction = self.base(core, t)
        time_embedding = self.base.time_embedding(t)
        query = self.core_proj(core) + self.time_proj(time_embedding)
        query = self.query_norm(query).unsqueeze(1)
        keys = self.factor_norm(self.factor_tokens(factors))
        attended, _ = self.cross_attention(query, keys, keys)
        residual = self.cross_out(attended.squeeze(1))
        return base_prediction + torch.tanh(self.gate) * residual

    def load_base_checkpoint(self, path: str, map_location="cpu") -> None:
        state = torch.load(path, map_location=map_location)
        if all(not key.startswith("base.") for key in state):
            self.base.load_state_dict(state)
            return
        base_state = {
            key[len("base.") :]: value
            for key, value in state.items()
            if key.startswith("base.")
        }
        self.base.load_state_dict(base_state)

    def load_adapter_checkpoint(self, path: str, map_location="cpu") -> None:
        state = torch.load(path, map_location=map_location)
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        self.load_state_dict(state, strict=False)

    def trainable_parameters(self):
        return (
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("base.")
        )
