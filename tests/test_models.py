"""
Tests for src/models — the stage-08 contrastive synchrony model.

Pinned here: that the config refuses what it should (unknown keys, ASSR), that the
dataset is read from the per-subject wavelet store with its axes named, that
the synchrony measures equal their naive definitions, and that the contrastive loss
and sampler build the positive pairs the method rests on. One end-to-end test plants
a shared signal and checks the encoder finds it. Torch tests skip without torch.
"""

import numpy as np
import pytest

from src.definitions.fields import ConditionVariants, ExperimentNames, MusicTypeVariants
from src.io.wavelet_store import wavelet_label_dir, write_wavelet_subject
from src.models.config import apply_override, load_config
from src.models.data import (
    SynchronyDataset,
    build_dataset,
    cross_fit_split,
    frame_average,
    pool_frequencies,
    samples_per_frame,
    zscore_over_time,
)
from src.models.synchrony import (
    candidate_intervals,
    group_synchrony,
    score_track,
    shifted_null,
    subject_synchrony,
)


def _dataset(data: np.ndarray, label: str = "Placebo_CLASSIC") -> SynchronyDataset:
    return SynchronyDataset(
        label=label,
        condition=ConditionVariants.PLACEBO,
        music_type=MusicTypeVariants.CLASSICAL,
        data=data.astype(np.float32),
        participants=[f"9{i:02d}" for i in range(data.shape[0])],
        feature_names=[f"f{i}" for i in range(data.shape[1])],
        frame_rate=10.0,
    )


def _planted(n_subjects=6, n_features=8, n_frames=600, noise=0.5, seed=0):
    """Subjects share one latent time course, each through its own mixing."""
    rng = np.random.default_rng(seed)
    latent = np.cumsum(rng.standard_normal((3, n_frames)), axis=1)
    latent = (latent - latent.mean(1, keepdims=True)) / latent.std(1, keepdims=True)
    mixing = rng.standard_normal((n_subjects, n_features, 3))
    data = mixing @ latent + noise * rng.standard_normal(
        (n_subjects, n_features, n_frames)
    )
    return zscore_over_time(data)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_config_loads_and_overrides_parse_as_yaml():
    config = load_config(overrides=["training.epochs=3", "data.freq_range=[4, 8]"])
    assert config.training.epochs == 3
    assert config.data.freq_range == [4, 8]
    assert config.data.frame_rate == 25.0 and isinstance(config.data.frame_rate, float)


def test_switching_encoder_uses_its_own_arguments():
    config = load_config(overrides=["model.name=mlp"])
    assert config.model.params == config.model.encoders["mlp"]


def test_unknown_key_is_refused():
    with pytest.raises(ValueError, match="unknown key"):
        load_config(overrides=["training.epoch=3"])


def test_assr_is_never_a_training_set():
    with pytest.raises(ValueError, match="held out"):
        load_config(overrides=["data.experiment=assr"])


def test_override_creates_intermediate_mappings():
    raw = {}
    apply_override(raw, "model.encoders.new.hidden=32")
    assert raw == {"model": {"encoders": {"new": {"hidden": 32}}}}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_frame_average_means_blocks_and_drops_the_tail():
    power = np.arange(11, dtype=float)[None, :]
    np.testing.assert_allclose(frame_average(power, 5), [[2.0, 7.0]])


def test_band_pooling_assigns_bins_by_half_open_band_edges():
    freqs = np.arange(1.0, 51.0)
    data = np.broadcast_to(freqs[None, None, :, None], (1, 1, 50, 2))
    pooled, names = pool_frequencies(data, freqs, [1.0, 50.0], "bands")
    assert names == ["delta", "theta", "alpha", "beta", "gamma"]
    # delta = 1-3 Hz, theta = 4-7 Hz: a bin on an edge belongs to the band above.
    np.testing.assert_allclose(pooled[0, 0, :2, 0], [2.0, 5.5])

    pooled, names = pool_frequencies(data, freqs, [8.0, 12.0], "none")
    assert names == ["8Hz", "9Hz", "10Hz", "11Hz", "12Hz"]


def test_cross_fitting_holds_out_each_block_once_and_never_glues_segments():
    dataset = _dataset(np.zeros((4, 2, 100)))
    held = []
    for fold in range(4):
        segments, block = cross_fit_split(dataset, fold, 4, min_frames=10)
        held.append((block.frame_offset, block.n_frames))
        covered = sorted(
            (s.frame_offset, s.frame_offset + s.n_frames) for s in segments
        )
        assert sum(stop - start for start, stop in covered) == 75
        assert all(
            stop <= block.frame_offset or start >= block.frame_offset + 25
            for start, stop in covered
        )
    assert held == [(0, 25), (25, 25), (50, 25), (75, 25)]


def test_cross_fitting_refuses_blocks_shorter_than_a_window():
    with pytest.raises(ValueError, match="fewer folds"):
        cross_fit_split(_dataset(np.zeros((4, 2, 100))), 0, 5, min_frames=30)


def test_dataset_is_read_from_the_wavelet_store_as_log_frames(tmp_path):
    n_chan, n_freq, n_times = 2, 50, 100
    rng = np.random.default_rng(0)
    power = rng.uniform(1.0, 2.0, (3, n_chan, n_freq, n_times))
    freqs = np.linspace(1, 50, n_freq)
    label = "Placebo_CLASSIC"
    directory = wavelet_label_dir(tmp_path, ExperimentNames.PSILO_MUSIC, label)
    for participant, subject in zip(["901", "902", "903"], power):
        write_wavelet_subject(
            directory,
            participant,
            subject,
            label=label,
            representation="power",
            freqs=freqs,
            n_cycles=freqs / 2,
            sfreq=250.0,
            channel_names=["E1", "E2"],
        )

    config = load_config(
        overrides=["data.normalize=none", "data.freq_pooling=none", "data.frame_rate=10"]
    ).data
    dataset = build_dataset(
        config,
        ConditionVariants.PLACEBO,
        MusicTypeVariants.CLASSICAL,
        root=tmp_path,
        n_jobs=2,
    )
    assert dataset.participants == ["901", "902", "903"]
    assert dataset.data.shape == (3, n_chan * n_freq, n_times // 25)
    # Frame 2 of subject 1, channel 0, bin 3: log of the mean of samples 50-74.
    expected = np.log10(power[1, 0, 3, 50:75].astype(np.float32).mean())
    np.testing.assert_allclose(dataset.data[1, 3, 2], expected, rtol=1e-5)

    banded = build_dataset(
        load_config(overrides=["data.normalize=none"]).data,
        ConditionVariants.PLACEBO,
        MusicTypeVariants.CLASSICAL,
        root=tmp_path,
        n_subjects=2,
        n_times=2,
    )
    assert banded.data.shape == (2, n_chan * 5, 2)
    assert banded.feature_names[:2] == ["E1@delta", "E1@theta"]


def test_frame_rate_must_divide_the_sampling_rate():
    with pytest.raises(ValueError, match="whole number"):
        samples_per_frame(250.0, 30.0)
    assert samples_per_frame(250.0, 25.0) == 10


# ---------------------------------------------------------------------------
# Synchrony
# ---------------------------------------------------------------------------


def test_group_synchrony_is_the_mean_pairwise_cosine():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((5, 7, 4))
    unit = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    naive = np.mean(
        [(unit[i] * unit[j]).sum(-1) for i in range(5) for j in range(5) if i != j],
        axis=0,
    )
    np.testing.assert_allclose(group_synchrony(emb), naive)
    assert subject_synchrony(emb).shape == (5, 7)


def test_time_shifts_destroy_synchrony_but_not_the_embeddings():
    rng = np.random.default_rng(2)
    shared = rng.standard_normal((1, 200, 8))
    emb = shared + 0.1 * rng.standard_normal((6, 200, 8))
    assert group_synchrony(emb).mean() > 0.9
    assert abs(shifted_null(emb, 20, 5, rng).mean()) < 0.2


def test_candidate_runs_merge_into_intervals():
    rng = np.random.default_rng(3)
    emb = rng.standard_normal((6, 40, 8))
    emb[:, 10:14] = rng.standard_normal(8)  # four synchronous windows
    scores = score_track(
        emb,
        np.arange(40.0),
        4.0,
        n_null=50,
        null_quantile=0.99,
        min_shift=4,
        seed=0,
    )
    assert scores["candidate"].iloc[10:14].all()
    intervals = candidate_intervals(scores)
    assert ((intervals["start_s"] <= 10.0) & (intervals["end_s"] >= 17.0)).sum() == 1

    # Overlapping spans merge even when a non-candidate window sits between them.
    gapped = scores.assign(candidate=False)
    gapped.loc[[20, 21, 23], "candidate"] = True
    merged = candidate_intervals(gapped)
    assert merged[["start_s", "end_s", "n_windows"]].values.tolist() == [[20, 27, 3]]


# ---------------------------------------------------------------------------
# Training (torch)
# ---------------------------------------------------------------------------


def test_loss_rewards_embeddings_that_match_by_group():
    torch = pytest.importorskip("torch")
    from src.models.training import contrastive_loss

    groups = torch.arange(4).repeat_interleave(3)
    perfect = torch.nn.functional.one_hot(groups, 4).float()
    random = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    good = contrastive_loss(perfect, groups, 0.1)
    bad = contrastive_loss(random, groups, 0.1)
    assert good.accuracy == 1.0 and good.loss < bad.loss
    assert good.chance == pytest.approx(2 / 11)


def test_ignored_pairs_leave_the_denominator_but_positives_stay():
    torch = pytest.importorskip("torch")
    from src.models.training import contrastive_loss

    groups = torch.tensor([0, 0, 1, 1])
    z = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    ignore = torch.ones(4, 4, dtype=torch.bool)  # ignore everything that may be ignored
    stats = contrastive_loss(z, groups, 0.1, ignore)
    # Only the positive is left in each row: log-softmax over one entry is 0.
    assert stats.loss.item() == pytest.approx(0.0, abs=1e-6)
    assert stats.chance.item() == 1.0


def test_sampler_draws_one_moment_from_distinct_subjects_per_group():
    torch = pytest.importorskip("torch")
    from src.models.training import WindowSampler

    # Encode subject and time in the value, so every window reveals where it came from.
    data = 1000.0 * np.arange(5)[:, None, None] + np.arange(60)[None, None, :]
    sampler = WindowSampler(
        [_dataset(np.repeat(data, 2, axis=1))],
        window=6,
        times_per_batch=8,
        subjects_per_time=3,
        exclusion=6,
        device=torch.device("cpu"),
        seed=0,
    )
    x, groups, ignore = sampler.sample()
    assert x.shape == (24, 2, 6)
    first = x[:, 0, 0].numpy()
    subject, start = first // 1000, first % 1000
    np.testing.assert_array_equal(x[:, 0, :].numpy(), first[:, None] + np.arange(6))
    for group in range(8):
        rows = groups.numpy() == group
        assert len(set(start[rows])) == 1 and len(set(subject[rows])) == 3
    close = np.abs(start[:, None] - start[None, :]) < 6
    different = groups.numpy()[:, None] != groups.numpy()[None, :]
    np.testing.assert_array_equal(ignore.numpy(), close & different)


def test_encoder_learns_a_planted_shared_signal():
    torch = pytest.importorskip("torch")
    from src.models.encoders import build_encoder
    from src.models.training import embed_windows, train_model

    config = load_config(
        overrides=[
            "training.epochs=4",
            "training.steps_per_epoch=25",
            "training.val_steps=5",
            "training.times_per_batch=16",
            "training.subjects_per_time=3",
            "training.exclusion_s=2",
        ]
    )
    dataset = _dataset(_planted())
    segments, held_out = cross_fit_split(dataset, 3, 4, min_frames=20)
    torch.manual_seed(0)
    model = build_encoder("conv1d", 8, 20, 16, {"hidden": 32, "n_layers": 2})
    history = train_model(
        model, segments, [held_out], config.training, 20, 10.0, torch.device("cpu")
    )
    last = history.iloc[-1]
    assert last["val_accuracy"] > 2 * last["val_chance"]

    embeddings = embed_windows(model, dataset.data, 20, np.arange(0, 581, 10), "cpu")
    assert embeddings.shape == (6, 59, 16)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=-1), 1.0, rtol=1e-5)
