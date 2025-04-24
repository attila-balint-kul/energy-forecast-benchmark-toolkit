from enfobench.dataset import Dataset
import pandas as pd
import numpy as np
from darts import TimeSeries
import os
import pickle
from darts.models import RegressionModel
from sklearn.linear_model import BayesianRidge,LinearRegression
from enfobench.evaluation import evaluate_metrics
from enfobench.evaluation.metrics import mean_absolute_error, mean_bias_error,root_mean_squared_error
from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration
from enfobench.dataset.utils import create_perfect_forecasts_from_covariates
from darts.utils.missing_values import missing_values_ratio, fill_missing_values

with open('Global_Regression_model_darts.pkl', 'rb') as f:
     Global_Regression_model = pickle.load(f)

class GlobalMultipleLinearRegressionDarts:

    def __init__(self, model):
        self.model=model

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Darts.GlobalMultipleLinearRegression",
            authors=[
                AuthorInfo(
                    name="Mohamad Khalil",
                    email="coo17619@newcastle.ac.uk"
                )
            ],
            
            type=ForecasterType.point,
            params={},
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame  | None = None,
        future_covariates: pd.DataFrame | None = None,
        **kwargs
        
    ) -> pd.DataFrame:
        
        
        series = TimeSeries.from_dataframe(history, value_cols=['y'])
        self.model.fit(series,)
        
        # Make forecast
        pred = self.model.predict(horizon)
        forecast = (
            pred.pd_dataframe()
            .rename(columns={"y": "yhat"})
            .fillna(history['y'].mean())
        )
        
        return forecast

model = GlobalMultipleLinearRegressionDarts(Global_Regression_model)

app = server_factory(model)
