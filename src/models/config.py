"""
Configuration of the stage-08 contrastive synchrony model.

Every setting of a run lives in one YAML file,
``config/models/contrastive_synchrony.yaml``, which is also the single source of
defaults. A variant file is deep-merged on top of it and ``key.path=value`` overrides on
top of that, so a variant states only what it changes. The merged result is validated
into frozen dataclasses, and an unknown key raises rather than being ignored.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    MusicTypeVariants,
)

DEFAULT_CONFIG_PATH = ProjectPaths.CONFIG_DIR / "models" / "contrastive_synchrony.yaml"

FREQ_POOLINGS = ("bands", "none")
NORMALIZATIONS = ("subject", "none")


def _require(value: str, allowed: Sequence[str], key: str) -> None:
    if value not in allowed:
        raise ValueError(f"{key} must be one of {list(allowed)}; got {value!r}.")


@dataclass(frozen=True)
class DataConfig:
    """Which recordings feed the model and how a window of them is represented."""

    experiment: str
    conditions: list[str]
    music_types: list[str]
    exclusions: list[str]
    frame_rate: float
    freq_range: list[float]
    freq_pooling: str
    channels: list[str] | None
    normalize: str
    window_s: float

    def __post_init__(self) -> None:
        if self.experiment_name == ExperimentNames.ASSR:
            raise ValueError(
                "data.experiment is 'assr', but the ASSR recordings are held out as the "
                "validation set of the synchrony model and must never be trained on."
            )
        # Building the enums is the validation: a typo fails here, at load time.
        _ = (self.recordings, self.exclusion_categories)
        _require(self.freq_pooling, FREQ_POOLINGS, "data.freq_pooling")
        _require(self.normalize, NORMALIZATIONS, "data.normalize")
        if len(self.freq_range) != 2 or self.freq_range[0] > self.freq_range[1]:
            raise ValueError(
                f"data.freq_range must be [low, high]; got {self.freq_range}."
            )

    @property
    def experiment_name(self) -> ExperimentNames:
        return ExperimentNames(self.experiment)

    @property
    def recordings(self) -> list[tuple[ConditionVariants, MusicTypeVariants]]:
        """Every (condition, music type) pair the run reads, one track each."""
        return [
            (ConditionVariants(condition), MusicTypeVariants(music_type))
            for condition in self.conditions
            for music_type in self.music_types
        ]

    @property
    def exclusion_categories(self) -> list[ExclusionCategories]:
        return [ExclusionCategories(category) for category in self.exclusions]


@dataclass(frozen=True)
class ModelConfig:
    """The encoder: a registered name plus constructor arguments per encoder."""

    name: str
    embedding_dim: int
    encoders: dict[str, dict[str, Any]]

    @property
    def params(self) -> dict[str, Any]:
        """Constructor arguments of the selected encoder."""
        return dict(self.encoders.get(self.name) or {})


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    steps_per_epoch: int
    times_per_batch: int
    subjects_per_time: int
    temperature: float
    learning_rate: float
    weight_decay: float
    exclusion_s: float
    n_folds: int
    val_steps: int
    seed: int
    device: str

    def __post_init__(self) -> None:
        if self.subjects_per_time < 2:
            raise ValueError(
                "training.subjects_per_time must be at least 2: a window needs another "
                "subject at the same moment to have a positive."
            )
        if self.n_folds < 2:
            raise ValueError(
                "training.n_folds must be at least 2: every window is scored by a model "
                f"trained without it, which needs a held-out block; got {self.n_folds}."
            )


@dataclass(frozen=True)
class ScoringConfig:
    stride_s: float
    n_null: int
    null_quantile: float


@dataclass(frozen=True)
class SynchronyConfig:
    run_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    scoring: ScoringConfig


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "scoring": ScoringConfig,
}


def _build(cls: type, raw: Any, where: str) -> Any:
    """Instantiate *cls* from *raw*, refusing unknown and missing keys."""
    if not isinstance(raw, dict):
        raise ValueError(f"'{where}' must be a mapping; got {raw!r}.")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    missing = sorted(set(known) - set(raw))
    if unknown or missing:
        raise ValueError(
            f"Config section '{where}': unknown key(s) {unknown}, missing key(s) "
            f"{missing}. Known keys: {sorted(known)}."
        )
    # YAML reads "10" as an int; a float field should not depend on how it was typed.
    values = {
        key: float(value) if known[key].type == "float" else value
        for key, value in raw.items()
    }
    return cls(**values)


def config_from_dict(raw: dict[str, Any]) -> SynchronyConfig:
    """Validate a merged config mapping into a :class:`SynchronyConfig`.

    :raises ValueError: On an unknown or missing key, or an invalid value.
    """
    sections = {
        name: _build(cls, raw.get(name), name) for name, cls in _SECTIONS.items()
    }
    return _build(SynchronyConfig, {**raw, **sections}, "<top level>")


def config_to_dict(config: SynchronyConfig) -> dict[str, Any]:
    return asdict(config)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_override(raw: dict[str, Any], override: str) -> None:
    """Set one ``dotted.key=value`` in *raw* in place.

    The value is parsed as YAML, so ``5`` is an int, ``[1, 50]`` a list and ``null``
    ``None``. Intermediate mappings are created as needed, which is what lets an
    override introduce a new encoder's arguments under ``model.encoders``.

    :raises ValueError: If *override* has no ``=``.
    """
    key, sep, value = override.partition("=")
    if not sep or not key:
        raise ValueError(
            f"An override must look like 'section.key=value'; got {override!r}."
        )
    *parents, leaf = key.strip().split(".")
    node = raw
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def load_config(
    path: Path | None = None, overrides: Sequence[str] = ()
) -> SynchronyConfig:
    """The default config, with a variant file and overrides applied on top.

    :param path: A variant YAML file merged over the defaults; ``None`` (or the default
        file itself) uses the defaults alone.
    :param overrides: ``dotted.key=value`` strings, applied in order after the merge.
    :return: The validated config.
    """
    raw = _read_yaml(DEFAULT_CONFIG_PATH)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH.resolve():
        raw = _deep_merge(raw, _read_yaml(Path(path)))
    for override in overrides:
        apply_override(raw, override)
    return config_from_dict(raw)


def save_config(config: SynchronyConfig, path: Path) -> None:
    """Write the fully resolved *config*, so a run records exactly what it used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)
