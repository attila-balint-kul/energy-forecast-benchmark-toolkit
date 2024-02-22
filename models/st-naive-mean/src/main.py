import pandas as pd
from sktime.forecasting.naive import NaiveForecaster

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index


class NaiveForecasterModel:
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Sktime.NaiveForecaster.Mean",
            authors=[AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")],
            type=ForecasterType.point,
            params={
                "strategy": "mean",
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
        y = history.y.fillna(history.y.mean())

        # Create model
        model = NaiveForecaster(strategy="mean")

        # Make forecast
        index = create_forecast_index(history=history, horizon=horizon)
        pred: pd.Series = model.fit_predict(y, fh=index, **kwargs)

        # Postprocess forecast
        forecast = pred.to_frame("yhat").fillna(y.mean())
        return forecast


# Instantiate your model
model = NaiveForecasterModel()

# Create a forecast server by passing in your model
app = server_factory(model)
