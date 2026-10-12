"""
Contrastive training of a window encoder, and embedding of whole tracks with it.

The self-supervision: windows recorded at the same moment of the same track by
different subjects are **positives**; every other window in the batch is a
**negative** — including the same subject at another moment, so subject identity alone
can never solve the task. What the encoder is pushed to keep is therefore whatever the
subjects share at a given moment and nowhere else, which is the definition of
stimulus-driven synchrony. The loss is the multi-positive InfoNCE of Khosla et al.
(2020, "Supervised Contrastive Learning"), with the moment as the class.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.models.config import ModelConfig, TrainingConfig
from src.models.data import SynchronyDataset
from src.models.encoders import build_encoder

_logger = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class ContrastiveStats(NamedTuple):
    loss: torch.Tensor
    accuracy: torch.Tensor  # anchors whose most similar other window is a positive
    chance: torch.Tensor  # the accuracy a random embedding would reach on this batch


def contrastive_loss(
    z: torch.Tensor,
    groups: torch.Tensor,
    temperature: float,
    ignore: torch.Tensor | None = None,
) -> ContrastiveStats:
    """Multi-positive InfoNCE over one batch.

    :param z: ``(N, D)`` L2-normalised embeddings.
    :param groups: ``(N,)`` group id; two windows of one group are a positive pair.
    :param temperature: Softmax temperature on the cosine similarities.
    :param ignore: ``(N, N)`` pairs to drop from the denominator — windows so close in
        time that calling them negatives would be wrong. Positives are never dropped.
    :return: The loss and two diagnostics; anchors without a positive are skipped.
    """
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    positive = (groups[:, None] == groups[None, :]) & ~eye
    excluded = eye if ignore is None else eye | (ignore & ~positive)

    logits = (z @ z.T / temperature).masked_fill(excluded, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # where() rather than a product: log_prob is -inf on excluded pairs and 0 * -inf is nan.
    pos_log_prob = torch.where(positive, log_prob, torch.zeros_like(log_prob)).sum(1)
    n_pos = positive.sum(1)
    has_pos = n_pos > 0

    loss = -(pos_log_prob[has_pos] / n_pos[has_pos]).mean()
    hit = positive.gather(1, logits.argmax(dim=1, keepdim=True)).squeeze(1)
    chance = n_pos / (~excluded).sum(1).clamp(min=1)
    return ContrastiveStats(
        loss, hit[has_pos].float().mean(), chance[has_pos].float().mean()
    )


class WindowSampler:
    """Draws contrastive batches: ``times_per_batch`` moments, each seen by
    ``subjects_per_time`` different subjects of the same track.

    Moments are drawn from all tracks in proportion to their length. Two moments of
    one track closer than ``exclusion`` frames are marked as ignored pairs, so
    overlapping or adjacent windows are not pushed apart as if they were unrelated.
    """

    def __init__(
        self,
        datasets: Sequence[SynchronyDataset],
        window: int,
        times_per_batch: int,
        subjects_per_time: int,
        exclusion: int,
        device: torch.device,
        seed: int,
    ) -> None:
        too_short = [ds.label for ds in datasets if ds.n_frames < window]
        if too_short:
            raise ValueError(
                f"Track(s) {too_short} are shorter than one {window}-frame window."
            )
        self.tracks = [torch.as_tensor(ds.data, device=device) for ds in datasets]
        self.window = window
        self.times_per_batch = times_per_batch
        self.subjects_per_time = min(
            subjects_per_time, min(ds.n_subjects for ds in datasets)
        )
        self.exclusion = exclusion
        self.device = device
        n_starts = np.array([ds.n_frames - window + 1 for ds in datasets], dtype=float)
        self.weights = n_starts / n_starts.sum()
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One batch.

        :return: ``x`` ``(N, features, window)``, ``groups`` ``(N,)`` and the
            ``(N, N)`` ignore mask for :func:`contrastive_loss`.
        """
        k = self.subjects_per_time
        track_of = self.rng.choice(
            len(self.tracks), size=self.times_per_batch, p=self.weights
        )
        xs, track_ids, starts = [], [], []
        for track in np.unique(track_of):
            data = self.tracks[track]
            n = int((track_of == track).sum())
            start = self.rng.integers(0, data.shape[-1] - self.window + 1, size=n)
            # k distinct subjects per moment: the first k of a random permutation.
            subjects = self.rng.random((n, data.shape[0])).argsort(axis=1)[:, :k]
            # unfold -> (S, F, n_starts, W); the two index arrays broadcast to (n, k)
            # and, being split by a slice, move to the front: (n, k, F, W).
            windows = data.unfold(-1, self.window, 1)[
                torch.as_tensor(subjects, device=self.device),
                :,
                torch.as_tensor(start[:, None], device=self.device),
            ]
            xs.append(windows.reshape(n * k, *windows.shape[2:]))
            track_ids.append(np.full(n * k, track))
            starts.append(np.repeat(start, k))

        x = torch.cat(xs)
        track_t = torch.as_tensor(np.concatenate(track_ids), device=self.device)
        start_t = torch.as_tensor(np.concatenate(starts), device=self.device)
        # A group is one (track, moment). Equal starts on one track share a group
        # only if drawn as one moment, so key on the draw order rather than the start.
        groups = torch.arange(
            self.times_per_batch, device=self.device
        ).repeat_interleave(k)
        same_track = track_t[:, None] == track_t[None, :]
        close = (start_t[:, None] - start_t[None, :]).abs() < self.exclusion
        ignore = same_track & close & (groups[:, None] != groups[None, :])
        return x, groups, ignore


def _run_steps(
    model: nn.Module,
    sampler: WindowSampler,
    n_steps: int,
    temperature: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    totals = np.zeros(3)
    for _ in range(n_steps):
        x, groups, ignore = sampler.sample()
        stats = contrastive_loss(
            F.normalize(model(x), dim=-1), groups, temperature, ignore
        )
        if optimizer is not None:
            optimizer.zero_grad()
            stats.loss.backward()
            optimizer.step()
        totals += [value.item() for value in stats]
    loss, accuracy, chance = totals / n_steps
    return {"loss": loss, "accuracy": accuracy, "chance": chance}


def train_model(
    model: nn.Module,
    train_sets: Sequence[SynchronyDataset],
    val_sets: Sequence[SynchronyDataset],
    config: TrainingConfig,
    window: int,
    frame_rate: float,
    device: torch.device,
) -> pd.DataFrame:
    """Train *model* in place for ``config.epochs`` epochs; the last one is kept.

    The validation batches are the same every epoch (the sampler is re-seeded), so the
    validation curve moves only because the model does. It is a diagnostic only: under
    cross-fitting the validation block is also the block the model will score, so
    choosing the epoch on it would leak. Choose ``epochs`` from these curves instead.

    :param val_sets: Held-out data for the diagnostic curves.
    :param window: Window length in frames.
    :param frame_rate: Frames per second, to turn ``config.exclusion_s`` into frames.
    :return: One row per epoch: training and validation loss, accuracy and chance.
    """
    torch.manual_seed(config.seed)
    exclusion = int(round(config.exclusion_s * frame_rate))
    common = dict(
        window=window,
        times_per_batch=config.times_per_batch,
        subjects_per_time=config.subjects_per_time,
        exclusion=exclusion,
        device=device,
    )
    train_sampler = WindowSampler(train_sets, seed=config.seed, **common)
    val_sampler = WindowSampler(val_sets, seed=config.seed + 1, **common)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    rows: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train = _run_steps(
            model, train_sampler, config.steps_per_epoch, config.temperature, optimizer
        )
        model.eval()
        val_sampler.reset(config.seed + 1)
        with torch.no_grad():
            val = _run_steps(
                model, val_sampler, config.val_steps, config.temperature, None
            )
        rows.append(
            {"epoch": epoch}
            | {f"train_{key}": value for key, value in train.items()}
            | {f"val_{key}": value for key, value in val.items()}
        )
        _logger.info(
            f"epoch {epoch:3d}/{config.epochs}  train loss {train['loss']:.3f} "
            f"acc {train['accuracy']:.3f} | val loss {val['loss']:.3f} "
            f"acc {val['accuracy']:.3f} (chance {val['chance']:.3f})"
        )

    return pd.DataFrame(rows)


@torch.no_grad()
def embed_windows(
    model: nn.Module,
    data: np.ndarray,
    window: int,
    starts: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Embed every subject's window at every start.

    :param data: ``(subjects, features, frames)``.
    :return: ``(subjects, len(starts), embedding_dim)``, L2-normalised.
    """
    model.eval()
    unfolded = torch.as_tensor(data, device=device).unfold(-1, window, 1)
    index = torch.as_tensor(starts, device=device)
    out = []
    for subject in unfolded:  # (F, n_starts, W)
        windows = subject[:, index].permute(1, 0, 2)  # (n_windows, F, W)
        chunks = [
            F.normalize(model(windows[i : i + batch_size]), dim=-1).cpu()
            for i in range(0, len(windows), batch_size)
        ]
        out.append(torch.cat(chunks).numpy())
    return np.stack(out)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: ModelConfig,
    n_features: int,
    n_frames: int,
) -> None:
    """Write the weights with what is needed to rebuild the encoder around them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "encoder": {
            "name": model_config.name,
            "embedding_dim": model_config.embedding_dim,
            "params": model_config.params,
            "n_features": n_features,
            "n_frames": n_frames,
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> nn.Module:
    payload = torch.load(path, map_location=device)
    spec = payload["encoder"]
    model = build_encoder(
        spec["name"],
        spec["n_features"],
        spec["n_frames"],
        spec["embedding_dim"],
        spec["params"],
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
