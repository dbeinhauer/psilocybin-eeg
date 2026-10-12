"""
Window encoders for the stage-08 contrastive synchrony model.

An encoder maps one window ``(n_features, n_frames)`` to an embedding vector. To add
one, decorate its class with :func:`register_encoder`; the config then selects it by
name (``model.name``) and passes ``model.encoders.<name>`` to its constructor. Every
encoder takes ``n_features``, ``n_frames`` and ``embedding_dim`` first, whether it
needs all three or not, so the training code never has to know which one it holds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

#: Every available encoder, by the name the config uses for it.
ENCODERS: dict[str, type[nn.Module]] = {}


def register_encoder(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in ENCODERS:
            raise ValueError(f"An encoder named {name!r} is already registered.")
        ENCODERS[name] = cls
        return cls

    return decorator


@register_encoder("conv1d")
class Conv1dEncoder(nn.Module):
    """Temporal CNN over the window.

    A 1x1 convolution first mixes all channel x frequency features into ``hidden``
    learned combinations — a spatial-spectral filter bank — and ``n_layers`` temporal
    convolutions then read their dynamics. Global average pooling over time makes the
    embedding insensitive to where inside the window an event falls.
    """

    def __init__(
        self,
        n_features: int,
        n_frames: int,
        embedding_dim: int,
        hidden: int = 128,
        n_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(n_features, hidden, kernel_size=1)]
        for _ in range(n_layers):
            layers += [
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x).mean(dim=-1))


@register_encoder("mlp")
class MlpEncoder(nn.Module):
    """Flattens the window and maps it through one hidden layer.

    Keeps the position of every frame, unlike :class:`Conv1dEncoder`, at the price of a
    first layer with ``n_features * n_frames * hidden`` weights.
    """

    def __init__(
        self,
        n_features: int,
        n_frames: int,
        embedding_dim: int,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * n_frames, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_encoder(
    name: str,
    n_features: int,
    n_frames: int,
    embedding_dim: int,
    params: dict[str, Any],
) -> nn.Module:
    """Instantiate the registered encoder *name*.

    :raises ValueError: If no encoder of that name is registered.
    :raises TypeError: If *params* holds an argument the encoder does not take.
    """
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder {name!r}; registered: {sorted(ENCODERS)}.")
    return ENCODERS[name](n_features, n_frames, embedding_dim, **params)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
