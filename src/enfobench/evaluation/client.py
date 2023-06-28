import io
from typing import Dict, List, Optional, Union

import pandas as pd
import requests

from enfobench.evaluation.protocols import EnvironmentInfo, ModelInfo


def to_buffer(df: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    return buffer


class ForecastClient:
    def __init__(
        self, host: str = "localhost", port: int = 3000, secure: bool = False, client=None
    ):
        self.base_url = f"{'https' if secure else 'http'}://{host}:{port}"
        self._session = requests.Session() if client is None else client

    def info(self) -> ModelInfo:
        response = self._session.get(f"{self.base_url}/info")
        if response.status_code != 200:
            response.raise_for_status()

        return ModelInfo(**response.json())

    def environment(self) -> EnvironmentInfo:
        response = self._session.get(f"{self.base_url}/environment")
        if response.status_code != 200:
            response.raise_for_status()

        return EnvironmentInfo(**response.json())

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: Optional[pd.DataFrame] = None,
        future_covariates: Optional[pd.DataFrame] = None,
        level: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        params: Dict[str, Union[int, List[int]]] = {
            "horizon": horizon,
        }
        if level is not None:
            params["level"] = level

        files = {
            "history": to_buffer(history),
        }
        if past_covariates is not None:
            files["past_covariates"] = to_buffer(past_covariates)
        if future_covariates is not None:
            files["future_covariates"] = to_buffer(future_covariates)

        response = self._session.post(
            url=f"{self.base_url}/forecast",
            params=params,
            files=files,
        )
        if response.status_code != 200:
            response.raise_for_status()

        df = pd.DataFrame.from_records(response.json()["forecast"])
        df["ds"] = pd.to_datetime(df["ds"])
        return df
