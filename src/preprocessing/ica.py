"""
This module contains ICA-related functions for EEG artifact removal,
including ICLabel classification and component exclusion.
"""

import logging

import numpy as np
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components

import mne

from src.definitions.fields import ICLabelComponentsClasses

_logger = logging.getLogger(__name__)

# Order of the classes in ICLabel tool.
# Based on https://mne.tools/mne-icalabel/dev/generated/api/mne_icalabel.iclabel.iclabel_label_components.html#mne_icalabel.iclabel.iclabel_label_components
IC_LABEL_CLASSES_ORDER = [
    ICLabelComponentsClasses.BRAIN.value,
    ICLabelComponentsClasses.MUSCLE.value,
    ICLabelComponentsClasses.EYE.value,
    ICLabelComponentsClasses.HEART.value,
    ICLabelComponentsClasses.LINE.value,
    ICLabelComponentsClasses.CHANNEL.value,
    ICLabelComponentsClasses.OTHER.value,
]


def get_ic_labeling_probabilities(
    component_probabilities,
) -> dict[ICLabelComponentsClasses, np.ndarray]:
    """
    Links each IC label to its probability.

    :param component_probabilities: Probabilities of all ICLabel Components.
    :return: Returns dictionary of key ICLabel component and its probabilities in np.ndarray form.
    """

    selected_idxs = {
        component: IC_LABEL_CLASSES_ORDER.index(component.value)
        for component in ICLabelComponentsClasses
    }
    return {
        component: component_probabilities[:, selected_idxs[component]]
        for component in selected_idxs
    }


def check_ic_component_probability(
    all_probabilities: dict[ICLabelComponentsClasses, np.ndarray],
    ic_idx: int,
    tested_component: ICLabelComponentsClasses,
    tested_threshold: float = 0.6,
    brain_threshold: float = 0.3,
) -> bool:
    """
    Check whether a given IC component exceeds the artifact threshold while
    remaining below the brain threshold (indicating it is likely an artifact).

    :param all_probabilities: Dictionary mapping ICLabel classes to their probability arrays.
    :param ic_idx: Index of the IC component to check.
    :param tested_component: The artifact class to test against.
    :param tested_threshold: Minimum probability for the tested artifact class.
    :param brain_threshold: Maximum allowed brain probability (components above this are kept).
    :return: True if the component should be considered an artifact, False otherwise.
    """
    return (
        all_probabilities[tested_component][ic_idx] >= tested_threshold
        and all_probabilities[ICLabelComponentsClasses.BRAIN][ic_idx]
        < brain_threshold
    )


def mark_ic_for_exclusion(
    component_probabilities,
    ica: ICA,
    brain_threshold=0.3,
    component_thresholds: dict[ICLabelComponentsClasses, float] = {
        ICLabelComponentsClasses.EYE: 0.40,
        ICLabelComponentsClasses.MUSCLE: 0.60,
        ICLabelComponentsClasses.HEART: 0.40,
        ICLabelComponentsClasses.CHANNEL: 0.5,
    },
    logger=None,
):
    """
    Based on the provided predicted probabilities of the ICA classes, mark
    putative artifact components for exclusion.

    We want to exclude muscle, eye and heartbeat .

    :param component_probabilities: Probability of each ICLabel class for each IC.
    :param ica: ICA decomposition results.
    :param brain_threshold: Maximum allowed brain probability threshold.
    :param component_thresholds: Dictionary of applied threshold for selected subset of ICLabel
    classes that we want to use in our IC exclusion. NOTE: The threshold for 'brain` signal
    defines the lowest value it is considered to be partially brain signal (will be kept). For
    other classes if the thresholds is surpasses -> it belongs to this class.
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: Returns ICA decomposition with ICs marked for exclusion (if classified as artifact components).
    """
    log = logger or _logger

    log.info("Getting probabilites of IC components.")
    labeled_probabilities = get_ic_labeling_probabilities(
        component_probabilities
    )
    # --- Auto-exclusion rule (tune thresholds to taste) ---
    # Conservative defaults: remove clear artifacts, keep 'brain' and usually keep 'other'
    # ---- Tuning thresholds ----
    exclude = []

    log.info(
        "Starting exclusion of the components passing selected threshold."
    )

    for ic_idx in range(len(component_probabilities)):
        for component, threshold in component_thresholds.items():
            if check_ic_component_probability(
                labeled_probabilities,
                ic_idx,
                component,
                tested_threshold=threshold,
                brain_threshold=brain_threshold,
            ):
                exclude.append(ic_idx)

    ica.exclude = sorted(set(exclude))

    log.info("All problematic components excluded.")
    log.info(f"ICs to exclude: {ica.exclude}")

    return ica


def apply_ica_component_filtering(
    interpolated_data: mne.io.Raw,
    logger=None,
) -> tuple[mne.io.Raw, ICA, np.ndarray]:
    """
    Runs ICA and then executes ICLabel tool to predict probabilities
    of the several artifacts in each component, excludes the artifact
    components and returns the filtered signal.

    :param interpolated_data: Already preprocessed and interpolated data without artifacts.
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: Returns tuple of processed data series by applying ICLabeling, ICAs with
    marked artifact ICs and array of probabilities of each IC artifact class.
    """
    log = logger or _logger

    log.info("Starting ICA decomposition.")
    # Run ICA on the interpolated data.
    ica = ICA(
        n_components=0.99,  # or an int
        method="infomax",  # extended infomax recommended :contentReference[oaicite:4]{index=4}
        fit_params=dict(extended=True),
        random_state=97,
        max_iter="auto",
    )
    log.info("Start ICA fitting.")
    ica.fit(interpolated_data, reject_by_annotation=True)
    log.info("ICA fitting finished.")

    # IC labeling (select each component probabilities).
    log.info("Start computing ICLabel components.")
    component_probabilities = iclabel_label_components(interpolated_data, ica)
    log.info("ICLabel component probabilites found.")
    # Based on the IC labeling select components for exclusion
    ica = mark_ic_for_exclusion(
        component_probabilities,
        ica,
        logger=log,
    )

    # Apply ICA to interpolated data (however, mne might be able to
    # work with the Raw data before interpolation).
    return ica.apply(interpolated_data), ica, component_probabilities
