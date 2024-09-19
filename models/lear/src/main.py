import pandas as pd

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index
from epftoolbox.models import LEAR


class LEARModel:
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="LEARModel",
            authors=[AuthorInfo(name="Margarida Mascarenhas", email="margarida.mascarenhas@kuleuven.be")],
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
        # Create index for prediction
        original_forecast_index = create_forecast_index(history, horizon)
        hourly_forecast_index = pd.date_range(
            start=original_forecast_index[0],
            end=original_forecast_index[-1]
                + pd.Timedelta(original_forecast_index.freq),  # Make it one step longer for interpolation
            freq='1h',
        )
        Feat_selection = True
        steps = len(hourly_forecast_index)

        # Resample the history to hourly frequency
        resampled_history = history.resample('1h').mean()
        if resampled_history.isna().any().any():
            resampled_history.interpolate(method='linear', inplace=True)

        # Merge the history with the weather data
        merged_df = pd.merge(resampled_history, past_covariates, left_index=True, right_index=True, how='outer')

        # Merge the future covariates
        if future_covariates is not None:
            merged_df = pd.concat(
                [merged_df, future_covariates.drop(columns=['cutoff_date'])], axis=0  # don't need the cutoff dates
            )

        calibration_window = (hourly_forecast_index[0].date() - pd.Timedelta(weeks=2) - history.first_valid_index().date()).days
        if calibration_window < 473:
            Feat_selection = False

        model = LEAR(calibration_window=calibration_window)

        # Use the recalibrate_and_forecast_next_day method of LEAR
        y_pred = model.predict_with_horizon(
            df=merged_df,
            initial_date=pd.Timestamp(hourly_forecast_index[0].date()),
            forecast_horizon_steps=steps,
            Feat_selection=Feat_selection,
        )

        # Create the prediction DataFrame by resampling the forecast to the original frequency
        original_freq = metadata['freq']
        new_index = pd.date_range(start=hourly_forecast_index.min(), end=hourly_forecast_index.max(), freq=original_freq)
        forecast = (
            pd.DataFrame({'timestamp': hourly_forecast_index, 'yhat': y_pred})
            .set_index('timestamp')
            .reindex(new_index)
            .interpolate(method="linear")
            .loc[original_forecast_index]
        )
        return forecast


# Instantiate your model
model = LEARModel()

# Create a forecast server by passing in your model
app = server_factory(model)

# Run the server if this script is the main one being executed
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=3000)
