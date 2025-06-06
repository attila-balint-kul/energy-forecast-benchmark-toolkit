import pandas as pd
from statsforecast.models import HistoricAverage

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index


class HistoricAverageModel:
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Statsforecast-HistoricAverage",
            authors=[AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")],
            type=ForecasterType.quantile,
            params={},
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
        model = HistoricAverage()

        # Make forecast
        pred = model.forecast(y=y.values, h=horizon, level=level, **kwargs)

        # Create index for forecast
        index = create_forecast_index(history=history, horizon=horizon)

        # Postprocess forecast
        forecast = pd.DataFrame(index=index, data=pred).rename(columns={"mean": "yhat"}).fillna(y.mean())
        return forecast


# Instantiate your model
model = HistoricAverageModel()

# Create a forecast server by passing in your model
app = server_factory(model)
