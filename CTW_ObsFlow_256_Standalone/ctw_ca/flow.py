from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from .model import FactorTokenizer, PairInteractionTokenizer
from .tw import FACTOR_NAMES, FactorState, factor_to_signal


class ConditionalCoreFlow(nn.Module):
    """Rectified-flow velocity field for a TW core conditioned on its factors."""

    def __init__(
        self,
        core_dim: int,
        factor_shapes: Dict[str, Sequence[int]],
        *,
        d_model: int = 64,
        num_heads: int = 4,
        tokens_per_factor: int = 4,
        interaction_tokens: bool = True,
        observation_dim: int = 0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.core_dim = int(core_dim)
        self.interaction_tokens = bool(interaction_tokens)
        self.observation_dim = int(observation_dim)
        self.factor_shapes = {
            name: tuple(factor_shapes[name]) for name in FACTOR_NAMES
        }
        self.core_proj = nn.Linear(core_dim, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.observation_mlp = None
        if self.observation_dim > 0:
            self.observation_mlp = nn.Sequential(
                nn.Linear(self.observation_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        self.query_norm = nn.LayerNorm(d_model)
        self.factor_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=True,
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, core_dim),
        )
        self.tokenizers = nn.ModuleDict()
        for name in FACTOR_NAMES:
            shape = self.factor_shapes[name]
            in_channels = shape[0] * shape[2] * shape[3]
            self.tokenizers[name] = FactorTokenizer(
                in_channels,
                d_model,
                tokens_per_factor,
            )
        self.pair_tokenizers = nn.ModuleDict()
        if self.interaction_tokens:
            for pair_name in ("G1_G2", "G2_G3", "G3_G1"):
                self.pair_tokenizers[pair_name] = PairInteractionTokenizer(
                    d_model,
                    tokens_per_factor,
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
        core_t: torch.Tensor,
        time: torch.Tensor,
        factors: FactorState,
        observation_features: torch.Tensor = None,
    ) -> torch.Tensor:
        time = time.to(dtype=core_t.dtype).reshape(-1, 1)
        time_features = torch.cat(
            [time, torch.sin(torch.pi * time), torch.cos(torch.pi * time)],
            dim=1,
        )
        query = self.core_proj(core_t) + self.time_mlp(time_features)
        if self.observation_mlp is not None:
            if observation_features is None:
                observation_features = core_t.new_zeros(
                    core_t.shape[0], self.observation_dim
                )
            query = query + self.observation_mlp(observation_features)
        query = self.query_norm(query)
        keys = self.factor_norm(self.factor_tokens(factors))
        attended, _ = self.cross_attention(
            query.unsqueeze(1),
            keys,
            keys,
        )
        return self.velocity_head(torch.cat([query, attended[:, 0]], dim=1))

    def load_checkpoint(self, path: str, map_location="cpu") -> dict:
        checkpoint = torch.load(path, map_location=map_location)
        state = checkpoint.get("model", checkpoint)
        self.load_state_dict(state)
        return checkpoint
