from typing import Any

import duckdb
import pandas as pd

from enfobench.core import Subset
from enfobench.datasets.base import DatasetBase

Metadata = dict[str, Any]


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
        return df.iloc[0].to_dict()


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
    """Data subset of the HuggingFace dataset containing all electricity load data.

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


class ElectricityDemandDataset(DatasetBase):
    """ElectricityDemandDataset class representing the HuggingFace dataset.

    This class is a collection of all subsets inside HuggingFace dataset.
    It provides an easy way to access the different subsets.

    Args:
        directory: The directory where the HuggingFace dataset is located.
                   This directory should contain all the subset files.
    """

    HUGGINGFACE_DATASET = "EDS-lab/electricity-demand"
    SUBSETS = ("demand", "metadata", "weather")

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

    def list_unique_ids(self) -> list[str]:
        return self.metadata_subset.list_unique_ids()

    def list_location_ids(self) -> list[str]:
        return self.weather_subset.list_location_ids()

    def get_data_by_unique_id(self, unique_id: str) -> tuple[pd.DataFrame, pd.DataFrame, Metadata]:
        metadata = self.metadata_subset.get_by_unique_id(unique_id)
        location_id = metadata["location_id"]

        demand = self.demand_subset.get_by_unique_id(unique_id)
        weather = self.weather_subset.get_by_location_id(location_id)
        # Filter weather data to match the period of the demand data
        weather = weather.loc[demand.index[0] - pd.Timedelta("7 days") : demand.index[-1] + pd.Timedelta("7 days")]
        return demand, weather, metadata
