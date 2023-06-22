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
    def __init__(self, host: str = "localhost", port: int = 3000, secure: bool = False):
        self.base_url = f"{'https' if secure else 'http'}://{host}:{port}"
        self.session = requests.Session()

    def info(self) -> ModelInfo:
        response = self.session.get(f"{self.base_url}/info")
        if not response.ok:
            response.raise_for_status()

        return ModelInfo(**response.json())

    def environment(self) -> EnvironmentInfo:
        response = self.session.get(f"{self.base_url}/environment")
        if not response.ok:
            response.raise_for_status()

        return EnvironmentInfo(**response.json())

    def predict(
        self,
        horizon: int,
        y: pd.Series,
        # X: pd.DataFrame,
        level: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        params: Dict[str, Union[int, List[int]]] = {
            "horizon": horizon,
        }
        if level is not None:
            params["level"] = level

        y_df = y.rename_axis("ds").reset_index()
        files = {
            "y": to_buffer(y_df),
            # "X": to_buffer(X),
        }

        response = self.session.post(
            url=f"{self.base_url}/predict",
            params=params,
            files=files,
        )
        if not response.ok:
            response.raise_for_status()

        df = pd.DataFrame.from_records(response.json()["forecast"])
        df["ds"] = pd.to_datetime(df["ds"])
        return df
