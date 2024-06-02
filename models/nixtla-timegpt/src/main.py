import logging
import os

import pandas as pd
from nixtla import NixtlaClient
from pyrate_limiter import Duration, Rate, Limiter, BucketFullException

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory

logger = logging.getLogger(__name__)


class NixtlaTimeGPTModel:
    def __init__(self, api_key: str, long_horizon: bool):
        self.long_horizon = long_horizon
        self.model = "timegpt-1-long-horizon" if long_horizon else "timegpt-1"
        self.client = NixtlaClient(api_key=api_key)
        self.limiter = Limiter(Rate(200, Duration.MINUTE), max_delay=Duration.MINUTE)

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f'Nixtla.TimeGPT{".LH" if self.long_horizon else ""}',
            authors=[
                AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be"),
            ],
            type=ForecasterType.quantile,
            params={
                "model": self.model,
            },
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        metadata: dict | None = None,
        level: list[int] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        # Fill missing values
        history = history.fillna(history.y.mean())

        # Rate limit forecast requests
        self.limiter.try_acquire(name=f"{history.index[-1]}")
        # Make request
        timegpt_fcst_df = self.client.forecast(
            df=history,
            h=horizon,
            level=level,
            model=self.model,
            target_col="y",
        )

        # post-process forecast
        forecast = timegpt_fcst_df.rename(columns={"TimeGPT": 'yhat'})
        forecast['timestamp'] = pd.to_datetime(forecast.timestamp)
        forecast = forecast.set_index("timestamp")
        return forecast


api_key = os.getenv("NIXTLA_API_KEY")
long_horizon = bool(int(os.getenv("ENFOBENCH_MODEL_LONG_HORIZON", 0)))

# Instantiate your model
model = NixtlaTimeGPTModel(api_key=api_key, long_horizon=long_horizon)

# Create a forecast server by passing in your model
app = server_factory(model)
