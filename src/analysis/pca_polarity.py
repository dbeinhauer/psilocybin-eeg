"""
Cross-participant polarity alignment for first-principal-component results.

A PCA component is defined only up to sign: ``(loading, score)`` and
``(-loading, -score)`` describe the same decomposition, and which one
``sklearn`` returns depends on numerical details, not on the data. Averaging
per-subject PC1 results without resolving that ambiguity averages arbitrarily
signed maps, which drives the group mean toward zero — with ``S`` subjects an
unresolved sign shrinks the group amplitude to roughly ``1/sqrt(S)`` of the
aligned one and leaves noise behind.

Because the goal is to **aggregate participants** — to find what they share and
average it — the sign is resolved by the criterion that directly serves that
goal: every subject's loading is flipped toward a common **template** (the
iteratively-refined group-mean loading), which maximises cross-subject
topography agreement. A subject that comes out strongly *anti*-correlated with
the others is almost always just sign-flipped, and this is what un-flips it.

A per-participant criterion (e.g. "flip when the loading sums negative") cannot
do that job: an evoked channel-PCA loading is typically **dipolar**, so its
entries very nearly cancel (on the ASSR set ``|sum| / |loading|_1`` runs
0.03–0.19, against 1.0 for a uniformly signed map). Its sign is then decided by
a few percent of residual imbalance, and subjects come out mutually inverted
even though each one individually "sums positive".

Template alignment fixes the signs *relative to each other*; the group's overall
orientation is still arbitrary, so it is anchored by making
:data:`REFERENCE_CHANNEL` positive in the template (falling back to the
template's strongest channel). That keeps the result reproducible and
independent of subject ordering.

Use :func:`align_pc1_signs` to derive the signs and :func:`apply_pc1_signs` to
apply them to loadings *and* to whatever the scores are (a time course, a
time-frequency map) so topography and score stay mutually consistent.
:func:`topography_consistency` is a diagnostic: it reports whether the aligned
loadings really do agree, so a genuine topographic outlier is not silently
averaged into the group.

All functions here are **pure**: no I/O, no plotting, no mutation of inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Electrode anchoring the group's overall orientation once relative signs are
#: aligned. Matches ``src.analysis.iva_quality.REF_POLARITY_CHANNEL``.
REFERENCE_CHANNEL = "Cz"

#: Default number of template-refinement iterations. The template converges in a
#: few passes; the default leaves ample margin.
N_TEMPLATE_ITER = 10


@dataclass(frozen=True)
class SignConsistency:
    """Whether sign-aligned PC1 loadings actually agree across participants.

    :param n_subjects: Number of subjects scored.
    :param n_agreeing: Subjects correlating **positively** with the group-mean
        loading. Anything below *n_subjects* is a topographic outlier that
        alignment could not resolve — inspect it before trusting a group average.
    :param median_pairwise_r: Median off-diagonal pairwise correlation between
        subject loadings. High values mean one shared spatial mode.
    :param min_subject_r: Weakest single subject-vs-group correlation.
    """

    n_subjects: int
    n_agreeing: int
    median_pairwise_r: float
    min_subject_r: float


def align_pc1_signs(
    loadings: np.ndarray,
    *,
    channel_names: list[str] | None = None,
    reference: str = REFERENCE_CHANNEL,
    n_iter: int = N_TEMPLATE_ITER,
) -> tuple[np.ndarray, str | None]:
    """Signs aligning every subject's PC1 loading to one shared orientation.

    Each subject's loading is iteratively flipped toward the group-mean template,
    maximising cross-subject agreement, then the group's overall orientation is
    anchored so *reference* is positive in the template. Multiply both the
    loadings and the scores by the returned signs (see :func:`apply_pc1_signs`).

    :param loadings: ``(n_subjects, n_features)`` per-subject PC1 loadings
        (``pca.components_[0]`` stacked over subjects) — **not** mutated.
    :param channel_names: Names matching the loading columns, needed to anchor
        the overall orientation. When ``None`` the anchor falls back to the
        template's largest-magnitude entry.
    :param reference: Channel anchoring the overall orientation; ignored when
        absent from *channel_names*.
    :param n_iter: Template-refinement iterations.
    :return: ``(signs, anchor)`` where *signs* is a ``(n_subjects,)`` float array
        of ``+1.0`` / ``-1.0`` and *anchor* is the channel name that fixed the
        overall orientation (``None`` when *channel_names* was not given).
    :raises ValueError: If *loadings* is not 2-D with at least one subject and
        one feature, or *channel_names* does not match the feature axis.
    """
    arr = np.asarray(loadings, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(
            f"loadings must be 2-D (n_subjects, n_features) with at least one "
            f"subject and one feature; got shape {np.shape(loadings)}."
        )
    if channel_names is not None and len(channel_names) != arr.shape[1]:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries but loadings has "
            f"{arr.shape[1]} features."
        )

    # Iteratively align every subject to the running group mean. This is what
    # maximises cross-subject agreement; it converges within a few passes.
    aligned = arr.copy()
    template = aligned[0].copy()
    for _ in range(n_iter):
        flips = np.sign(aligned @ template)
        flips[flips == 0] = 1.0
        aligned = aligned * flips[:, None]
        template = aligned.mean(axis=0)

    # Relative signs are now consistent, but the group's overall orientation is
    # still arbitrary — anchor it so the reference channel reads positive.
    anchor: str | None = None
    if channel_names is not None:
        anchor = (
            reference
            if reference in channel_names
            else channel_names[int(np.argmax(np.abs(template)))]
        )
        if template[channel_names.index(anchor)] < 0:
            aligned = -aligned

    signs = np.sign((arr * aligned).sum(axis=1))
    signs[signs == 0] = 1.0
    return signs, anchor


def apply_pc1_signs(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply per-subject PC1 signs along the leading (subject) axis.

    Broadcasts *signs* against any trailing shape, so the same call orients
    loadings ``(S, C)``, time courses ``(S, T)`` and time-frequency maps
    ``(S, F, T)``. Apply it to the loadings and to the scores with the *same*
    signs, otherwise a subject's topography and score end up mutually flipped.

    :param values: Array whose first axis is the subject axis — **not** mutated.
    :param signs: ``(n_subjects,)`` signs from :func:`align_pc1_signs`.
    :return: A **new** array of the same shape, sign-aligned.
    :raises ValueError: If the subject axes of *values* and *signs* disagree.
    """
    vals = np.asarray(values)
    sgn = np.asarray(signs, dtype=np.float64)
    if sgn.ndim != 1:
        raise ValueError(f"signs must be 1-D; got shape {sgn.shape}.")
    if vals.ndim < 1 or vals.shape[0] != sgn.shape[0]:
        raise ValueError(
            f"subject axis mismatch: values has {vals.shape[0] if vals.ndim else 0}, "
            f"signs has {sgn.shape[0]}."
        )
    return vals * sgn.reshape((-1,) + (1,) * (vals.ndim - 1))


def topography_consistency(loadings: np.ndarray) -> SignConsistency:
    """Score how well already-aligned PC1 loadings agree across participants.

    Call this on **sign-aligned** loadings. After alignment every subject should
    correlate positively with the group mean; a subject that does not is a
    topographic outlier rather than a sign problem, and averaging it into a group
    map cancels real signal.

    Constant (zero-variance) loadings have undefined correlation and are counted
    as non-agreeing rather than raising.

    :param loadings: ``(n_subjects, n_features)`` sign-aligned PC1 loadings —
        **not** mutated.
    :return: A :class:`SignConsistency` summary.
    :raises ValueError: If *loadings* is not 2-D with at least two features.
    """
    arr = np.asarray(loadings, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"loadings must be 2-D (n_subjects, n_features) with at least two "
            f"features; got shape {np.shape(loadings)}."
        )

    n_subjects = arr.shape[0]
    group = arr.mean(axis=0)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        sa, sb = a.std(), b.std()
        if sa == 0.0 or sb == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    subject_r = np.array([corr(row, group) for row in arr])
    n_agreeing = int(np.sum(subject_r > 0.0))  # NaN compares False

    pairs = [
        corr(arr[i], arr[j])
        for i in range(n_subjects)
        for j in range(i + 1, n_subjects)
    ]
    finite_pairs = [p for p in pairs if np.isfinite(p)]
    finite_subject = subject_r[np.isfinite(subject_r)]

    return SignConsistency(
        n_subjects=n_subjects,
        n_agreeing=n_agreeing,
        median_pairwise_r=(
            float(np.median(finite_pairs)) if finite_pairs else float("nan")
        ),
        min_subject_r=(
            float(finite_subject.min()) if finite_subject.size else float("nan")
        ),
    )
