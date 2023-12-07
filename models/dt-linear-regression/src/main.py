import os

import pandas as pd
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.linear_model import LinearRegression

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import periods_in_duration


class DartsLinearRegressionModel:
    def __init__(self, seasonality: str, direct: bool):
        self.seasonality = seasonality.upper()
        self.direct = direct

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Darts.LinearRegression.{'Direct.' if self.direct else ''}{self.seasonality}",
            authors=[
                AuthorInfo(name="Mohamad Khalil", email="coo17619@newcastle.ac.uk"),
                AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be"),
            ],
            type=ForecasterType.point,
            params={
                "seasonality": self.seasonality,
                "direct": self.direct,
            },
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        # Fill missing values
        history = history.fillna(history.y.mean())

        # Create model
        periods = periods_in_duration(history.index, duration=self.seasonality)
        model = RegressionModel(
            lags=list(range(-periods, 0)),
            output_chunk_length=horizon,
            model=LinearRegression(),
            multi_models=not self.direct,
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
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY")
direct = bool(int(os.getenv("ENFOBENCH_MODEL_DIRECT")))

# Instantiate your model
model = DartsLinearRegressionModel(seasonality=seasonality, direct=direct)

# Create a forecast server by passing in your model
app = server_factory(model)
