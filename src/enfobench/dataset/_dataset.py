import logging
from abc import ABCMeta
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Metadata:
    unique_id: str
    location_id: str
    latitude: float
    longitude: float
    building_type: str


class Subset(metaclass=ABCMeta):  # noqa: B024
    """Subset class representing one of the subset of the HuggingFace dataset.

    Args:
        file_path: The path to the subset file.
    """

    def __init__(self, file_path: Path | str) -> None:
        file_path = Path(file_path).resolve()
        if not file_path.is_file() or not file_path.exists():
            msg = "Please provide the existing file where the subset is located."
            raise ValueError(msg)
        self.file_path = file_path

    def __repr__(self):
        return f"{self.__class__.__name__}(file_path={self.file_path})"

    def read(self) -> pd.DataFrame:
        """Read the subset from the file."""
        return pd.read_parquet(self.file_path)


class MetadataSubset(Subset):
    """Metadata subset of the HuggingFace dataset containing all metadata about the meters.

    Args:
        file_path: The path to the subset file.
    """

    def list_unique_ids(self) -> list[str]:
        """Lists all unique ids."""
        query = """
            SELECT DISTINCT unique_id
            FROM read_parquet(?)
        """
        conn = duckdb.connect(":memory:")
        return conn.execute(query, parameters=[str(self.file_path)]).fetch_df().unique_id.tolist()

    def get_by_unique_id(self, unique_id: str) -> Metadata:
        """Returns the metadata for the given unique id.

        Args:
            unique_id: The unique id of the meter.
        """
        query = """
            SELECT *
            FROM read_parquet(?)
            WHERE unique_id = ?
        """
        conn = duckdb.connect(":memory:")
        df = conn.execute(query, parameters=[str(self.file_path), unique_id]).fetch_df()
        if df.empty:
            msg = f"Unique id '{unique_id}' was not found."
            raise KeyError(msg)
        return Metadata(**df.to_dict(orient="records")[0])


class WeatherSubset(Subset):
    """Weather subset of the HuggingFace dataset containing all weather data.

    Args:
        file_path: The path to the subset file.
    """

    def list_location_ids(self) -> list[str]:
        """Lists all location ids."""
        query = """
            SELECT DISTINCT location_id
            FROM read_parquet(?)
        """
        conn = duckdb.connect(":memory:")
        return conn.execute(query, parameters=[str(self.file_path)]).fetch_df().location_id.tolist()

    def get_by_location_id(self, location_id: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Returns the weather data for the given location id.

        Args:
            location_id: The location id of the weather station.
            columns: The columns to return. If None, all columns are returned.
        """
        conn = duckdb.connect(":memory:")

        if columns:
            query = f"""
                SELECT timestamp, {", ".join(columns)}
                FROM read_parquet(?)
                WHERE location_id = ?
            """  # noqa: S608
        else:
            query = """
                SELECT *
                FROM read_parquet(?)
                WHERE location_id = ?
            """
        df = conn.execute(query, parameters=[str(self.file_path), location_id]).fetch_df()
        if df.empty:
            msg = f"Location id '{location_id}' was not found."
            raise KeyError(msg)

        # Remove location_id and set timestamp as index
        df.drop(columns=["location_id"], inplace=True, errors="ignore")
        df.set_index("timestamp", inplace=True)
        return df


class DemandSubset(Subset):
    """Demand subset of the HuggingFace dataset containing all demand data.

    Args:
        file_path: The path to the subset file.
    """

    def get_by_unique_id(self, unique_id: str):
        """Returns the demand data for the given unique id.

        Args:
            unique_id: The unique id of the meter.
        """
        query = """
            SELECT *
            FROM read_parquet(?)
            WHERE unique_id = ?
        """
        conn = duckdb.connect(":memory:")
        df = conn.execute(query, parameters=[str(self.file_path), unique_id]).fetch_df()
        if df.empty:
            msg = f"Unique id '{unique_id}' was not found."
            raise KeyError(msg)

        # Remove unique_id and set timestamp as index
        df.drop(columns=["unique_id"], inplace=True, errors="ignore")
        df.set_index("timestamp", inplace=True)
        return df


class DemandDataset:
    """DemandDataset class representing the HuggingFace dataset.

    This class is a collection of all subsets inside HuggingFace dataset.
    It provides an easy way to access the different subsets.

    Args:
        directory: The directory where the HuggingFace dataset is located.
                   This directory should contain all the subset files.
    """

    HUGGINGFACE_DATASET = "attila-balint-kul/electricity-demand"
    SUBSETS = ("demand", "metadata", "weather")

    def __init__(self, directory: Path | str) -> None:
        directory = Path(directory).resolve()
        if not directory.is_dir() or not directory.exists():
            msg = f"Please provide the existing directory where the '{self.HUGGINGFACE_DATASET}' dataset is located."
            raise ValueError(msg)
        self.directory = directory.resolve()

    def __repr__(self) -> str:
        return f"DemandDataset(directory={self.directory})"

    def _check_for_valid_subset(self, subset: str):
        if subset not in self.SUBSETS:
            msg = f"Please provide a valid subset. Available subsets: {self.SUBSETS}"
            raise ValueError(msg)

    @property
    def metadata_subset(self) -> MetadataSubset:
        """Returns the metadata subset."""
        return MetadataSubset(self._get_subset_path("metadata"))

    @property
    def weather_subset(self) -> WeatherSubset:
        """Returns the weather subset."""
        return WeatherSubset(self._get_subset_path("weather"))

    @property
    def demand_subset(self) -> DemandSubset:
        """Returns the demand subset."""
        return DemandSubset(self._get_subset_path("demand"))

    def get_subset(self, subset: str) -> Subset:
        """Returns the selected subset."""
        self._check_for_valid_subset(subset)
        if subset == "metadata":
            return self.metadata_subset
        elif subset == "weather":
            return self.weather_subset
        elif subset == "demand":
            return self.demand_subset
        msg = f"Please provide a valid subset. Available subsets: {self.SUBSETS}"
        raise ValueError(msg)

    def _get_subset_path(self, subset: str) -> Path:
        filepath = self.directory / f"{subset}.parquet"
        if not filepath.exists():
            msg = (
                f"There is no {subset} in the directory. "
                f"Make sure to download all subsets from the HuggingFace dataset: {self.HUGGINGFACE_DATASET}."
            )
            raise ValueError(msg)
        return self.directory / f"{subset}.parquet"

    def read_subset(self, subset: str) -> pd.DataFrame:
        """Reads the selected subset."""
        return self.get_subset(subset).read()

    def list_unique_ids(self) -> list[str]:
        return self.metadata_subset.list_unique_ids()

    def list_location_ids(self) -> list[str]:
        return self.weather_subset.list_location_ids()

    def get_data_by_unique_id(self, unique_id: str) -> tuple[pd.DataFrame | pd.DataFrame | Metadata]:
        metadata = self.metadata_subset.get_by_unique_id(unique_id)

        demand = self.demand_subset.get_by_unique_id(unique_id)
        weather = self.weather_subset.get_by_location_id(metadata.location_id)
        return demand, weather, metadata


class Dataset:
    """Dataset class representing a collection of data required for forecasting task.

    Args:
        target: The target variable.
        past_covariates: The past covariates.
        future_covariates: The future covariates.
        metadata: The metadata.
    """

    def __init__(
        self,
        target: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        metadata: Metadata | None = None,
    ):
        self._target = self._check_target(target.copy())
        self._first_available_target_date: pd.Timestamp = self._target.index[0]
        self._last_available_target_date: pd.Timestamp = self._target.index[-1]

        self._past_covariates = (
            self._check_past_covariates(past_covariates.copy()) if past_covariates is not None else None
        )

        self._future_covariates = (
            self._check_external_forecasts(future_covariates.copy()) if future_covariates is not None else None
        )
        self.metadata = metadata

    @property
    def target_available_since(self) -> pd.Timestamp:
        """Returns the first available target date."""
        return self._first_available_target_date

    @property
    def target_available_until(self) -> pd.Timestamp:
        """Returns the last available target date."""
        return self._last_available_target_date

    @property
    def target_freq(self) -> str:
        """Returns the frequency of the target."""
        return self._target.index.inferred_freq

    @staticmethod
    def _check_target(y: pd.DataFrame) -> pd.DataFrame:
        if isinstance(y, pd.Series):
            logger.warning("Target is a Series, converting to DataFrame.")
            y = y.to_frame("y")
        if not isinstance(y.index, pd.DatetimeIndex):
            msg = f"Target dataframe must have DatetimeIndex, have {y.index.__class__.__name__}."
            raise ValueError(msg)
        y.rename_axis("timestamp", inplace=True)
        y.sort_index(inplace=True)
        return y

    def _check_past_covariates(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        if not isinstance(X.index, pd.DatetimeIndex):
            msg = f"Past covariates must have DatetimeIndex, have {X.index.__class__.__name__}."
            raise ValueError(msg)

        X.rename_axis("timestamp", inplace=True)
        X.sort_index(inplace=True)

        if X.index[0] > self._first_available_target_date:
            msg = "Covariates must be provided for the full target timeframe, covariates start after target values."
            raise ValueError(msg)

        if X.index[-1] < self._last_available_target_date:
            msg = "Covariates must be provided for the full target timeframe, covariates end before target values."
            raise ValueError(msg)
        return X

    def _check_external_forecasts(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        first_forecast_date = X.cutoff_date.min()
        last_forecast_date = X.cutoff_date.max()
        last_forecast_end_date = X[X.cutoff_date == last_forecast_date].timestamp.max()

        if first_forecast_date > self._first_available_target_date:
            msg = (
                "External forecasts must be provided for the full target timeframe, "
                "forecasts start after target values."
            )
            raise ValueError(msg)

        if last_forecast_end_date < self._last_available_target_date:
            msg = (
                "External forecasts must be provided for the full target timeframe, "
                "forecasts end before target values."
            )
            raise ValueError(msg)

        return X

    def _check_cutoff_in_rage(self, cutoff_date: pd.Timestamp):
        if cutoff_date < self._first_available_target_date:
            msg = f"Cutoff date is before the start date: {cutoff_date} < {self._first_available_target_date}."
            raise IndexError(msg)

        if cutoff_date > self._last_available_target_date:
            msg = f"Cutoff date is after the end date: {cutoff_date} > {self._last_available_target_date}."
            raise IndexError(msg)

    def get_history(self, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        """Returns the history of the target variable up to the cutoff date.

        Args:
            cutoff_date: The cutoff date.

        Returns:
            The history of the target variable up to the cutoff date.
        """
        self._check_cutoff_in_rage(cutoff_date)
        return self._target[self._target.index <= cutoff_date]

    def get_past_covariates(self, cutoff_date: pd.Timestamp) -> pd.DataFrame | None:
        """Returns the past covariates for the cutoff date.

        Args:
            cutoff_date: The cutoff date.

        Returns:
            The past covariates up until the cutoff date.
        """
        if self._past_covariates is None:
            return None

        self._check_cutoff_in_rage(cutoff_date)
        return self._past_covariates[self._past_covariates.index <= cutoff_date]

    def get_future_covariates(self, cutoff_date: pd.Timestamp) -> pd.DataFrame | None:
        """Returns the future covariates for the cutoff date.

        Args:
            cutoff_date: The cutoff date.

        Returns:
            The last external forecasts made before the cutoff date.
        """
        if self._future_covariates is None:
            return None

        self._check_cutoff_in_rage(cutoff_date)
        last_past_cutoff_date = self._future_covariates.cutoff_date[
            self._future_covariates.cutoff_date <= cutoff_date
        ].max()

        future_covariates = self._future_covariates[
            (self._future_covariates.cutoff_date == last_past_cutoff_date)
            & (self._future_covariates.timestamp > cutoff_date)
        ]
        future_covariates.set_index("timestamp", inplace=True)
        return future_covariates
