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

    @staticmethod
    def get_experiment_data_dir(
        experiment_name: ExperimentNames, raw: bool = True
    ) -> tuple[Path, Path]:
        """
        Get path to directory containing dataset and to a CSV file containing mapping of
        the participant keys.

        :param experiment_name: Value of the `ExperimentNames` field equals to experiment
        data directory name.
        :param raw: If `True`, get path to raw data directory, else to processed data.
        :return: Returns tuple containing path to data directory and path to participant
        mapping CSV file.
        """

        root_data_dir = (
            ProjectPaths.RAW_DATA_DIR if raw else ProjectPaths.PROCESSED_DATA_DIR
        )
        experiment_name_str = experiment_name.value
        data_dir = root_data_dir / experiment_name_str
        participant_mapping_path = (
            ProjectPaths.PARTICIPANT_MAPPING_DIR / f"{experiment_name_str}.csv"
        )

        return data_dir, participant_mapping_path

    @staticmethod
    def get_coordinates_file_path(coordinate_system: CoordinateSystems) -> Path:
        """
        Get path to coordinate file based on the coordinate system.

        :param coordinate_system: Value of the `CoordinateSystems` enum.
        :return: Path to the coordinate file.
        """
        filename = f"{coordinate_system.value}.sfp"
        return ProjectPaths.COORDINATES_DIR / filename
