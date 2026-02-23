from pathlib import Path

from src.definitions.fields import ExperimentNames, CoordinateSystems


class ProjectPaths:
    """
    Unified project paths used across all experiment (mainly for data handling).
    """

    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()  # Root of the project.
    DATA_DIR = PROJECT_ROOT / "data"  # Directory where all data is stored.
    COORDINATES_DIR = DATA_DIR / "coordinates"  # Directory for all coordinate files.
    RAW_DATA_DIR = DATA_DIR / "raw"  # Directory for all raw data files.
    PROCESSED_DATA_DIR = (
        DATA_DIR / "processed"
    )  # Directory for all processed data files.
    PARTICIPANT_MAPPING_DIR = (
        DATA_DIR / "participant_mappings"
    )  # Directory for participant mapping csv files.
    EXCLUDED_ELECTRODES_DIR = (
        DATA_DIR / "excluded_electrodes"
    )  # Directory where excluded electrodes from processing are stored (we want to typically omit the boundary electrodes).
    EXCLUDED_ICS_FILENAME_MAPPING = "excluded_ics_mapping.csv"  # Filename where the mapping of all ICs selected for exclusion are stored alongside with their category.
    PLOTS_PATH = PROJECT_ROOT / "plots"  # Path to all project plots.
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
