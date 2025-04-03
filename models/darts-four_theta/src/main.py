import os

import pandas as pd
from darts import TimeSeries
from darts.models import FourTheta
from darts.utils.utils import SeasonalityMode

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import periods_in_duration


class DartsFourThetaModel:
    """FourTheta model from Darts.

    Args:
        seasonality: The seasonality of the time series. E.g. "1D" for daily seasonality.

    References:
        https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html
    """

    def __init__(self, seasonality: str):
        self.seasonality = seasonality.upper()

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Darts-FourTheta-{self.seasonality}-SM-A",
            authors=[
                AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be"),
            ],
            type=ForecasterType.point,
            params={
                "seasonality": self.seasonality,
            },
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        # Fill missing values
        history = history.fillna(history.y.mean())

        # Create model
        seasonality_period = periods_in_duration(history.index, duration=self.seasonality)
        model = FourTheta(
            seasonality_period=seasonality_period,
            season_mode=SeasonalityMode.ADDITIVE,
        )

        # Fit model
        series = TimeSeries.from_dataframe(history, value_cols=["y"])
        model.fit(series)

        # Make forecast
        pred = model.predict(horizon)

        # Postprocess forecast
        forecast = pred.pd_dataframe().rename(columns={"y": "yhat"}).fillna(history.y.mean())
        return forecast


# Load parameters
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY", "1D")

# Instantiate your model
model = DartsFourThetaModel(seasonality)

# Create a forecast server by passing in your model
app = server_factory(model)
