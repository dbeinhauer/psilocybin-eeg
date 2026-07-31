from pathlib import Path

import numpy as np

from src.definitions.fields import ExperimentNames, CoordinateSystems


class AssrEpoch:
    """
    Stimulus-locked epoch geometry of the ASSR paradigm.

    Single source of truth for every onset-locked ASSR analysis, so the evoked
    epoch, the wavelet epoch and the onset-locked quality references all cut the
    same window. The stimulus is a 500 ms train; the epoch keeps a short
    pre-onset baseline and then the stimulus plus an equally long post-stimulus
    interval, so a response outlasting the stimulus is still visible.

    Never derive the epoch length from the observed inter-onset gap alone: that
    makes the window a property of whatever jitter the recording happened to
    have rather than of the paradigm. Use :meth:`post_onset_samples`, which
    applies the paradigm length and only *caps* it by the shortest gap.
    """

    PRE_ONSET_S = 0.1  # Baseline kept before each onset (negative time).
    STIMULUS_DURATION_S = 0.5  # Length of the stimulus train itself.
    POST_STIMULUS_S = 0.5  # Interval kept after stimulus offset.
    POST_ONSET_S = 1.0  # Total post-onset span (stimulus + post-stimulus).

    # Offset added to a stimulus marker annotation (``fam+``) to obtain the
    # acoustic onset of the train. Negative: the marker LAGS the sound, landing
    # near the end of the 500 ms train, so every window above is defined relative
    # to ``marker_time + MARKER_ONSET_OFFSET_S``, not to the marker itself.
    #
    # This is a property of the paradigm, not of the analysis. It was measured
    # from 40 Hz inter-trial phase coherence over all 38 ASSR recordings — the
    # driven response spans [-370, +100] ms around the marker (per-recording
    # median lag -380 ms, IQR [-398, -360]); see
    # ``notebooks/00-preprocessing/assr_stimulus_onset_offset.ipynb``. Rounded to
    # -0.4 s pending confirmation from the paradigm documentation.
    #
    # Change this single value to re-time every onset-locked ASSR analysis. Doing
    # so invalidates the stimulus-aligned products on disk (RAW_CROPPED, the
    # concatenated array with its ``.stimulus_onsets.npy``, and the wavelet
    # cache): they must be regenerated, in that order, before any onset-locked
    # analysis is re-run. Recordings up to and including RAW_AFTER_ICA are
    # unaffected — the offset is applied when annotations are read, never stored.
    MARKER_ONSET_OFFSET_S = -0.4

    @staticmethod
    def pre_onset_samples(sfreq: float) -> int:
        """
        Pre-onset baseline length in samples.

        :param sfreq: Sampling frequency in Hz.
        :return: Number of samples kept before each onset.
        """
        return int(round(AssrEpoch.PRE_ONSET_S * sfreq))

    @staticmethod
    def post_onset_samples(sfreq: float, min_gap: int | None = None) -> int:
        """
        Post-onset epoch length in samples, capped so epochs never overlap.

        :param sfreq: Sampling frequency in Hz.
        :param min_gap: Shortest inter-onset gap in samples. When given and
            shorter than the paradigm window, it caps the returned length so no
            epoch reaches the next stimulus.
        :return: Number of samples kept after each onset (at least 1).
        :raises ValueError: If *min_gap* is not positive.
        """
        post = int(round(AssrEpoch.POST_ONSET_S * sfreq))
        if min_gap is not None:
            if min_gap <= 0:
                raise ValueError(f"min_gap must be positive; got {min_gap}.")
            post = min(post, int(min_gap))
        return max(1, post)

    @staticmethod
    def stimulus_mask(epoch_times: np.ndarray) -> np.ndarray:
        """
        Boolean mask selecting the stimulus interval of an epoch.

        Restricting to the driven interval matters for steady-state measures: a
        1 s epoch around a 500 ms stimulus is half silence, which dilutes any
        stimulus-locked estimate computed over the whole post-onset window.

        :param epoch_times: ``(win,)`` epoch times in seconds, ``0`` at onset.
        :return: Boolean ``(win,)`` mask, true for ``0 <= t < stimulus end``.
        """
        times = np.asarray(epoch_times, dtype=float)
        return (times >= 0.0) & (times < AssrEpoch.STIMULUS_DURATION_S)


class ProjectPaths:
    """
    Unified project paths used across all experiment (mainly for data handling).
    """

    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()  # Root of the project.
    CONFIG_DIR = (
        PROJECT_ROOT / "config"
    )  # Directory for checked-in configuration files.
    DATA_DIR = PROJECT_ROOT / "data"  # Directory where all data is stored.
    COORDINATES_DIR = CONFIG_DIR / "coordinates"  # Directory for all coordinate files.
    RAW_DATA_DIR = DATA_DIR / "raw"  # Directory for all raw data files.
    INTERIM_DATA_DIR = (
        DATA_DIR / "interim"
    )  # Directory for intermediate products (before ICA, etc.).
    PROCESSED_DATA_DIR = (
        DATA_DIR / "processed"
    )  # Directory for all processed data files.
    PARTICIPANT_MAPPING_DIR = (
        CONFIG_DIR / "participant_mappings"
    )  # Directory for participant mapping csv files.
    EXCLUDED_ELECTRODES_DIR = (
        CONFIG_DIR / "excluded_electrodes"
    )  # Directory where excluded electrodes from processing are stored (we want to typically omit the boundary electrodes).
    EXCLUDED_ICS_FILENAME_MAPPING = "excluded_ics_mapping.csv"  # Filename where the mapping of all ICs selected for exclusion are stored alongside with their category.
    STIMULUS_ONSETS_SUFFIX = ".stimulus_onsets.npy"  # Filename suffix for the stimulus-onset sample positions saved next to a concatenated data array (same prefix as the array).
    PLOTS_PATH = PROJECT_ROOT / "plots"  # Path to all project plots.
    RESULTS_DB_PATH = (
        PROJECT_ROOT / "results_db"
    )  # Path to the CSV results database for the Interactive Explorer.
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"  # Path to all project notebooks.
    EXCLUDED_PARTICIPANTS_DIR = (
        DATA_DIR / "excluded_participants"
    )  # Directory where excluded participants from processing are stored (typically due to bad data quality).

    @staticmethod
    def get_experiment_data_dir(
        experiment_name: ExperimentNames, is_processed: bool = True
    ) -> tuple[Path, Path]:
        """
        Get path to directory containing dataset and to a CSV file containing mapping of
        the participant keys.

        :param experiment_name: Value of the `ExperimentNames` field equals to experiment
        data directory name.
        :param is_processed: If `True`, get path to processed data directory, else to unprocessed data (raw dataset).
        :return: Returns tuple containing path to data directory and path to participant
        mapping CSV file.
        """

        root_data_dir = (
            ProjectPaths.PROCESSED_DATA_DIR
            if is_processed
            else ProjectPaths.RAW_DATA_DIR
        )
        experiment_name_str = experiment_name.value
        data_dir = root_data_dir / experiment_name_str
        participant_mapping_path = (
            ProjectPaths.PARTICIPANT_MAPPING_DIR / f"{experiment_name_str}.csv"
        )

        return data_dir, participant_mapping_path

    @staticmethod
    def get_experiment_interim_dir(experiment_name: ExperimentNames) -> Path:
        """
        Get path to the interim data directory for the given experiment.

        Interim data includes intermediate products such as data before ICA,
        ICA components, and IC probabilities.

        :param experiment_name: Value of the `ExperimentNames` field equals to experiment
        data directory name.
        :return: Path to interim data directory for the experiment.
        """
        return ProjectPaths.INTERIM_DATA_DIR / experiment_name.value

    @staticmethod
    def get_coordinates_file_path(
        coordinate_system: CoordinateSystems,
    ) -> tuple[Path, Path]:
        """
        Get path to coordinate file based on the coordinate system.

        :param coordinate_system: Value of the `CoordinateSystems` enum.
        :return: Tuple of the path to the coordinate file and path to excluded electrodes for the analysis.
        """
        coordinates_filename = f"{coordinate_system.value}.sfp"
        excluded_electrodes_filename = f"{coordinate_system.value}.csv"
        return (
            ProjectPaths.COORDINATES_DIR / coordinates_filename,
            ProjectPaths.EXCLUDED_ELECTRODES_DIR / excluded_electrodes_filename,
        )
