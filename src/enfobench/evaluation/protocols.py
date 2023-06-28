from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd
from pydantic import BaseModel


class ForecasterType(str, Enum):
    point = "point"
    quantile = "quantile"
    density = "density"
    ensemble = "ensemble"


class ModelInfo(BaseModel):
    """Model information.

    Args
    ----
    name:
        Name of the model.
    type:
        Type of the model.
    params:
        Parameters of the model.
    """

    name: str
    type: ForecasterType
    params: Dict[str, Any]


class EnvironmentInfo(BaseModel):
    packages: Dict[str, str]


class Model(Protocol):
    def info(self) -> ModelInfo:
        ...

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: Optional[pd.DataFrame] = None,
        future_covariates: Optional[pd.DataFrame] = None,
        level: Optional[List[int]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        ...


class Dataset:
    def __init__(
        self,
        target: pd.Series,
        covariates: Optional[pd.DataFrame] = None,
        external_forecasts: Optional[pd.DataFrame] = None,
        freq: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.freq = freq or target.index.inferred_freq
        self.target = self._check_target(target.copy())
        if self.freq is None:
            raise ValueError("Frequency of the target time series cannot be inferred.")

        self.start_date = self.target["ds"].iloc[0]
        self.end_date = self.target["ds"].iloc[-1]

        self.covariates = (
            self._check_covariates(covariates.copy()) if covariates is not None else None
        )
        self.external_forecasts = (
            self._check_external_forecasts(external_forecasts.copy())
            if external_forecasts is not None
            else None
        )
        self.metadata = metadata or {}

    def _check_target(self, y: pd.Series) -> pd.DataFrame:
        # TODO: replace manual checks with pandera schema
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("Index of y must be a DatetimeIndex")
        y.rename_axis("ds", inplace=True)
        y.sort_index(inplace=True)
        y = y.to_frame("y").reset_index()
        return y

    def _check_covariates(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO: replace manual checks with pandera schema
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Index of X must be a DatetimeIndex")

        X.rename_axis("ds", inplace=True)
        X.sort_index(inplace=True)

        if X.index[0] > self.start_date:
            raise ValueError(
                "Covariates must be provided for the full target timeframe, covariates start after target values."
            )

        if X.index[-1] < self.end_date:
            raise ValueError(
                "Covariates must be provided for the full target timeframe, covariates end before target values."
            )

        X.reset_index(inplace=True)
        return X

    def _check_external_forecasts(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO: replace manual checks with pandera schema
        first_forecast_date = X.cutoff_date.min()
        last_forecast_date = X.cutoff_date.max()
        last_forecast_end_date = X[X.cutoff_date == last_forecast_date].ds.max()

        if first_forecast_date > self.start_date:
            raise ValueError(
                "External forecasts must be provided for the full target timeframe, "
                "forecasts start after target values."
            )

        if last_forecast_end_date < self.end_date:
            raise ValueError(
                "External forecasts must be provided for the full target timeframe, "
                "forecasts end before target values."
            )

        return X

    def _check_cutoff_in_rage(self, cutoff_date: pd.Timestamp):
        if cutoff_date < self.start_date:
            raise IndexError(
                f"Cutoff date is before the start date: {cutoff_date} < {self.start_date}."
            )

        if cutoff_date > self.end_date:
            raise IndexError(
                f"Cutoff date is after the end date: {cutoff_date} > {self.end_date}."
            )

    def get_history(self, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        """Returns the history of the target variable up to the cutoff date.

        Parameters
        ----------
        cutoff_date : pd.Timestamp
            The cutoff date.

        Returns
        -------
            The history of the target variable up to the cutoff date.
        """
        self._check_cutoff_in_rage(cutoff_date)
        return self.target[self.target.ds <= cutoff_date]

    def get_past_covariates(self, cutoff_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Returns the past covariates for the cutoff date.

        Parameters
        ----------
        cutoff_date : pd.Timestamp
            The cutoff date.

        Returns
        -------
            The past covariates up until the cutoff date.

        """
        if self.covariates is None:
            return None

        self._check_cutoff_in_rage(cutoff_date)
        return self.covariates[self.covariates.ds <= cutoff_date]

    def get_future_covariates(self, cutoff_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Returns the future covariates for the cutoff date.

        Parameters
        ----------
        cutoff_date : pd.Timestamp
            The cutoff date.

        Returns
        -------
            The last external forecasts made before the cutoff date.
        """
        if self.external_forecasts is None:
            return None

        self._check_cutoff_in_rage(cutoff_date)
        last_past_cutoff_date = self.external_forecasts.ds[
            self.external_forecasts.ds <= cutoff_date
        ].max()
        return self.external_forecasts[
            (self.external_forecasts.cutoff_date == last_past_cutoff_date)
            & (self.external_forecasts.ds > cutoff_date)
        ]
