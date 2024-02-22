import pandas as pd
from statsforecast.models import MSTL

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration


class MSTLModel:
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Statsforecast.MSTL.1D.7D",
            authors=[AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")],
            type=ForecasterType.quantile,
            params={
                "seasonalities": [
                    "1D",
                    "7D",
                ],
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
        periods_in_one_day = periods_in_duration(y.index, duration="1D")
        periods_in_one_week = periods_in_duration(y.index, duration="7D")
        model = MSTL(season_length=[periods_in_one_day, periods_in_one_week])

        # Make forecast
        pred = model.forecast(y=y.values, h=horizon, level=level, **kwargs)

        # Create index for forecast
        index = create_forecast_index(history=history, horizon=horizon)

        # Postprocess forecast
        forecast = pd.DataFrame(index=index, data=pred).rename(columns={"mean": "yhat"}).fillna(y.mean())
        return forecast


# Instantiate your model
model = MSTLModel()

# Create a forecast server by passing in your model
app = server_factory(model)
