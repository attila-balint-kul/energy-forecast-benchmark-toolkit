import pandas as pd
from darts import TimeSeries
from darts.models import NaiveSeasonal

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import periods_in_duration


class DartsDailyNaiveSeasonalModel:
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Darts.DailyNaive",
            authors=[AuthorInfo(name="Mohamad Khalil", email="coo17619@newcastle.ac.uk")],
            type=ForecasterType.point,
            params={},
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
        periods = periods_in_duration(history.index, duration=pd.Timedelta("1D"))
        model = NaiveSeasonal(periods)

        # Fit model
        series = TimeSeries.from_dataframe(history, value_cols=["y"])
        model.fit(series)

        # Make forecast
        pred = model.predict(horizon)

        # Postprocess forecast
        forecast = pred.pd_dataframe().rename(columns={"y": "yhat"}).fillna(history.y.mean())
        return forecast


# Instantiate your model
model = DartsDailyNaiveSeasonalModel()

# Create a forecast server by passing in your model
app = server_factory(model)
