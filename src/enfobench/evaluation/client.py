import io
import json
from http import HTTPStatus

import pandas as pd
import requests
from requests import HTTPError

from enfobench.core import ModelInfo
from enfobench.evaluation.server import EnvironmentInfo


def df_to_buffer(df: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=True)
    buffer.seek(0)
    return buffer


def metadata_to_buffer(metadata: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    buffer.write(json.dumps(metadata).encode())
    buffer.seek(0)
    return buffer


class ForecastClient:
    def __init__(self, host: str = "localhost", port: int = 3000, *, use_https: bool = False, client=None):
        self.base_url = f"{'https' if use_https else 'http'}://{host}:{port}"
        self._session = requests.Session() if client is None else client

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        metadata: dict | None = None,
        level: list[int] | None = None,
    ) -> pd.DataFrame:
        params: dict[str, int | list[int]] = {
            "horizon": horizon,
        }
        if level is not None:
            params["level"] = level

        files = {
            "history": df_to_buffer(history),
        }
        if past_covariates is not None:
            files["past_covariates"] = df_to_buffer(past_covariates)
        if future_covariates is not None:
            files["future_covariates"] = df_to_buffer(future_covariates)
        if metadata is not None:
            files["metadata"] = metadata_to_buffer(metadata)

        response = self._session.post(
            url=f"{self.base_url}/forecast",
            params=params,
            files=files,
        )
        if response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
            response = json.loads(response.text)
            raise HTTPError(response.get("error", "Internal Server Error"), response=response)
        elif response.status_code != HTTPStatus.OK:
            response.raise_for_status()

        df = pd.DataFrame.from_records(response.json()["forecast"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def info(self) -> ModelInfo:
        response = self._session.get(f"{self.base_url}/info")
        if response.status_code != HTTPStatus.OK:
            response.raise_for_status()

        return ModelInfo(**response.json())

    def environment(self) -> EnvironmentInfo:
        response = self._session.get(f"{self.base_url}/environment")
        if response.status_code != HTTPStatus.OK:
            response.raise_for_status()

        return EnvironmentInfo(**response.json())
