import logging
import warnings
from abc import ABCMeta
from pathlib import Path

import pandas as pd
from pandas.tseries.frequencies import to_offset

logger = logging.getLogger(__name__)


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
        metadata: dict | None = None,
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
            msg = "Target is a Series, converting to DataFrame."
            warnings.warn(msg, UserWarning, stacklevel=2)
            y = y.to_frame("y")

        if not isinstance(y.index, pd.DatetimeIndex):
            msg = f"Target dataframe must have DatetimeIndex, have {y.index.__class__.__name__}."
            raise ValueError(msg)

        if y.index.inferred_freq is None:
            msg = "Frequency cannot be inferred from target's index, resampling target."
            warnings.warn(msg, UserWarning, stacklevel=2)
            freq = to_offset(y.index[1] - y.index[0]).freqstr
            freq = "1" + freq if not freq[0].isdigit() else freq
            y = y.resample(freq).asfreq()

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
            msg = "Covariates should be provided for the full target timeframe, covariates start after target values."
            warnings.warn(msg, UserWarning, stacklevel=2)

        if X.index[-1] < self._last_available_target_date:
            msg = "Covariates should be provided for the full target timeframe, covariates end before target values."
            warnings.warn(msg, UserWarning, stacklevel=2)
        return X

    def _check_external_forecasts(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        first_forecast_date = X.cutoff_date.min()
        last_forecast_date = X.cutoff_date.max()
        last_forecast_end_date = X[X.cutoff_date == last_forecast_date].timestamp.max()

        if first_forecast_date > self._first_available_target_date:
            msg = (
                "External forecasts should be provided for the full target timeframe, "
                "forecasts start after target values."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        if last_forecast_end_date < self._last_available_target_date:
            msg = (
                "External forecasts should be provided for the full target timeframe, "
                "forecasts end before target values."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

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

        The cutoff date is the timestamp when the forecast is made,
        therefore the cutoff_date is not included in the history.

        Args:
            cutoff_date: The cutoff date.

        Returns:
            The history of the target variable up to the cutoff date.
        """
        self._check_cutoff_in_rage(cutoff_date)
        return self._target[self._target.index < cutoff_date]

    def get_past_covariates(self, cutoff_date: pd.Timestamp) -> pd.DataFrame | None:
        """Returns the past covariates for the cutoff date.

        The cutoff date is the timestamp when the forecast is made.
        As the covariates are weather parameters measured at the indicated timestamp,
        the cutoff_date is included in the past covariates.

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

        The cutoff date is the timestamp when the forecast is made.
        As the covariates are weather parameters measured at the indicated timestamp,
        the cutoff_date is not included in the future covariates.

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
