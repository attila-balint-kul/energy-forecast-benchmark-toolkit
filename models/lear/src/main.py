import pandas as pd
from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index
from datetime import datetime, timedelta
from epftoolbox.models import LEAR


class LEARModel:
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="LEARModel",
            authors=[AuthorInfo(name="Margarida Mascarenhas", email="margarida.mascarenhas@kuleuven.be")],
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


        history = history.resample('1h').mean()

        if history.isna().any().any():
            history.interpolate(method='linear', inplace=True)

        steps=len(future_covariates)
        #merged_df = pd.merge(history, past_covariates, on='timestamp')
        merged_df = pd.merge(history, past_covariates, left_index=True, right_index=True, how='outer')

        future_covariates_modified = future_covariates.iloc[:, 1:].copy() if future_covariates is not None else pd.DataFrame()
        merged_df = pd.concat([merged_df, future_covariates_modified], axis=0)
      
        model = LEAR(calibration_window=500)

        next_day_date = future_covariates['cutoff_date'][0] #history.index[-steps]
   
        # Use the recalibrate_and_forecast_next_day method of LEAR
        y_pred = model.predict_with_horizon(
            df=merged_df,
            initial_date=next_day_date,
            forecast_horizon_steps=steps
        )

        hourly_index = pd.date_range(start=next_day_date, periods=steps, freq='1h')

        # Create the DataFrame
        forecast = pd.DataFrame({'date': hourly_index, 'yhat': y_pred}).set_index('date')

        freq_real=metadata['freq']
                        
        if not any(char.isdigit() for char in freq_real):
            freq_real = '1' + freq_real
        step_timedelta = pd.Timedelta(freq_real)
        steps_to_add = int((1 / (step_timedelta.total_seconds() / 3600)) - 1)
        last_timestamp = forecast.index[-1]

        new_index = pd.date_range(start=forecast.index[0], end=last_timestamp + steps_to_add * step_timedelta, freq=freq_real)
        prediction = forecast.reindex(new_index).interpolate(method='linear')

        return prediction

# Instantiate your model
model = LEARModel()

# Create a forecast server by passing in your model
app = server_factory(model)

# Run the server if this script is the main one being executed
if __name__ == "__main__":
    app.run()