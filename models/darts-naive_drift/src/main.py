import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.baselines import NaiveDrift

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory


class DartsNaiveDriftModel:
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Darts-NaiveDrift",
            authors=[AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")],
            type=ForecasterType.point,
            params={},
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
        model = NaiveDrift()

        # Fit model
        series = TimeSeries.from_dataframe(history, value_cols=["y"])
        model.fit(series)

        # Make forecast
        pred = model.predict(horizon)

        # Postprocess forecast
        forecast = pred.pd_dataframe().rename(columns={"y": "yhat"}).fillna(history.y.mean())
        return forecast


# Instantiate your model
model = DartsNaiveDriftModel()

# Create a forecast server by passing in your model
app = server_factory(model)
