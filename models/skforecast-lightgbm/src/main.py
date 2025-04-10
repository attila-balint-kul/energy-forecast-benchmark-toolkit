from typing import Literal

import holidays
import pandas as pd
from environs import Env
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.timeseries.forecasting import WindowFeatures
from lightgbm import LGBMRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.preprocessing import SplineTransformer

from enfobench import ModelInfo, AuthorInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import periods_in_duration, create_forecast_index


class SkforecastLightGBMModel:

    def __init__(
        self,
        model_name: str,
        lags: str = '1D',
        calendar_features: list[str] | None = None,
        cyclical_calendar_features: list[str] | None = None,
        bspline_calendar_features: list[str] | None = None,
        holiday_features: bool = False,
        exog_window_features: list[tuple[str, str, Literal['min', 'max', 'mean', 'std', 'sum']]] | None = None,
        window_features: list[tuple, str, Literal['min', 'max', 'mean', 'std', 'sum']] | None = None,
        exogenous_features: list[str] | Literal['all'] | None = None,
    ):
        self.model_name = model_name
        self.lags = lags
        self.calendar_features = calendar_features or []
        self.cyclical_calendar_features = cyclical_calendar_features or []
        self.bspline_calendar_features = bspline_calendar_features or []
        self.holiday_features = holiday_features
        self.exog_window_features = exog_window_features or []
        self.window_features = window_features
        self.rolling_features = None
        self.exogenous_features = exogenous_features

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Skforecast-LightGBM-{self.lags}-{self.model_name}",
            authors=[
                AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be"),
            ],
            type=ForecasterType.point,
            params={
                "lags": self.lags,
                "calendar_features": self.calendar_features,
                "cyclical_calendar_features": self.cyclical_calendar_features,
                "bspline_calendar_features": self.bspline_calendar_features,
                "holiday_features": self.holiday_features,
                "exog_window_features": self.exog_window_features,
                "window_features": self.window_features,
                "exogenous_features": self.exogenous_features,
            },
        )

    @staticmethod
    def _extract_holiday_features(index: pd.DatetimeIndex, location: str) -> pd.DataFrame:
        """Extracts holiday features from the input data.

        The following features are extracted:
            - holiday: 1 if the date is a holiday, 0 otherwise
            - working_day: 1 if the date is a working day, 0 otherwise

        Args:
            index: Index for which to extract the features.
            location: Location of series.
        """
        # Extracting weekend feature
        holiday_features = pd.DataFrame(index=index)

        # Extracting holiday feature
        country_code = location if ',' not in location else location.split(',')[1].strip()
        country_holidays = holidays.country_holidays(country_code)
        holiday_features['holiday'] = [int(date in country_holidays) for date in holiday_features.index.date]

        # Extracting working day feature
        holiday_features['working_day'] = (holiday_features.index.dayofweek < 5 & ~holiday_features.holiday).astype(int)
        return holiday_features

    @staticmethod
    def _extract_calendar_features(index: pd.DatetimeIndex, calendar_features: list[str]) -> pd.DataFrame:
        """Extracts calendar features from the input data.

        The following features are supported:
            - month
            - quarter
            - semester
            - year
            - week
            - day_of_week
            - day_of_month
            - day_of_year
            - weekend
            - month_start
            - month_end
            - quarter_start
            - quarter_end
            - year_start
            - year_end
            - leap_year
            - days_in_month
            - hour
            - minute
            - second

        Args:
            index: Index for which to extract the features.
        """
        df = pd.DataFrame(index=index).assign(y=0)
        calendar_transformer = DatetimeFeatures(variables='index', features_to_extract=calendar_features)
        calendar_features = calendar_transformer.fit_transform(df).drop(columns='y')
        return calendar_features

    @staticmethod
    def _extract_cyclical_calendar_features(index: pd.DatetimeIndex, calendar_features: list[str]) -> pd.DataFrame:
        """Extracts cyclical calendar features from the input data.

        The following features are supported:
            - month
            - quarter
            - semester
            - week
            - day_of_week
            - day_of_month
            - day_of_year
            - hour
            - minute
            - second

        Args:
            index: Index for which to extract the features.
        """
        df = pd.DataFrame(index=index).assign(y=0)
        calendar_transformer = DatetimeFeatures(variables='index', features_to_extract=calendar_features)
        calendar_features_df = calendar_transformer.fit_transform(df).drop(columns='y')

        # Cyclical encoding
        max_values = {
            "month": 12,
            "quarter": 4,
            "semester": 2,
            "week": 52,
            "day_of_week": 7,
            "day_of_month": 31,
            "day_of_year": 366,
            "hour": 24,
            "minute": 60,
            "second": 60,
        }
        cyclical_encoder = CyclicalFeatures(
            variables=calendar_features,
            max_values={k: v for k, v in max_values.items() if k in calendar_features},
            drop_original=True
        )
        cyclical_features = cyclical_encoder.fit_transform(calendar_features_df)
        return cyclical_features

    @staticmethod
    def _extract_bspline_calendar_features(index: pd.DatetimeIndex, calendar_features: list[str]) -> pd.DataFrame:
        """Extracts calendar features from the input data.

        The following features are supported:
            - month
            - quarter
            - week
            - day_of_week
            - day_of_month
            - hour

        Args:
            index: Index for which to extract the features.
        """
        df = pd.DataFrame(index=index).assign(y=0)
        calendar_transformer = DatetimeFeatures(variables='index', features_to_extract=calendar_features)
        calendar_features = calendar_transformer.fit_transform(df).drop(columns='y')

        def spline_transformer(period: int, degree: int = 3, extrapolation="periodic"):
            """
            Returns a transformer that applies B-spline transformation.
            """
            return SplineTransformer(
                degree=degree,
                n_knots=period + 1,
                knots='uniform',
                extrapolation=extrapolation,
                include_bias=True
            ).set_output(transform="pandas")

        # Cyclical encoding
        periods = {
            "month": 12,
            "quarter": 4,
            "week": 52,
            "day_of_week": 7,
            "day_of_month": 31,
            "hour": 24,
        }
        spline_features = pd.DataFrame(index=index)
        for feature in calendar_features:
            splines_ = spline_transformer(period=periods[feature]).fit_transform(calendar_features[[feature]])
            splines_.columns = [f"{feature}_spline_{i}" for i in range(1, 1 + len(splines_.columns))]
            spline_features = pd.concat([spline_features, splines_], axis=1)
        return spline_features

    @staticmethod
    def _extract_windowed_features(
        exog: pd.DataFrame,
        window_features: list[tuple[str, str, Literal['min', 'max', 'mean', 'std', 'sum']]],
    ) -> pd.DataFrame:
        """Extracts windowed features from the input data.

        Args:
            exog: Exogenous variables.
            window_features: Features to extract as a tuple. The first element is the variable, the second element is the window size and the third is the windowed function.
        """
        windowed_features_df = pd.DataFrame(index=exog.index)
        for variable, window, function in window_features:
            window_transformer = WindowFeatures(
                variables=[variable],
                window=[window],
                functions=[function],
                missing_values="ignore",
                drop_na=False,
                drop_original=True,
            )
            window_values = window_transformer.fit_transform(exog[[variable]])
            windowed_features_df = windowed_features_df.merge(
                window_values, left_index=True, right_index=True, how="left"
            )

        return windowed_features_df

    def _prepare_data(
        self,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        metadata: dict | None = None,
    ) -> tuple[pd.Series, pd.DataFrame | None]:
        freq = 'h' if metadata['freq'] == '1hour' else metadata['freq']

        target = (
            history.copy()
            .resample(freq)
            .asfreq()
            .ffill()
        ).y

        if self.exogenous_features is None or self.exogenous_features == []:
            return target, None

        exog = (
            past_covariates
            .combine_first(future_covariates.drop(columns=['cutoff_date']))
            .resample(freq).asfreq()
            .interpolate()
        )
        if self.exogenous_features != 'all' and isinstance(self.exogenous_features, list):
            exog = exog.loc[:, self.exogenous_features]

        # Calendar features
        if self.calendar_features:
            calendar_features = self._extract_calendar_features(
                index=exog.index,
                calendar_features=self.calendar_features,
            )
            exog = exog.merge(calendar_features, left_index=True, right_index=True, how='left')

        if self.cyclical_calendar_features:
            cyclical_calendar_features = self._extract_cyclical_calendar_features(
                index=exog.index,
                calendar_features=self.cyclical_calendar_features,
            )
            exog = exog.merge(cyclical_calendar_features, left_index=True, right_index=True, how='left')

        if self.bspline_calendar_features:
            bspline_calendar_features = self._extract_bspline_calendar_features(
                index=exog.index,
                calendar_features=self.bspline_calendar_features,
            )
            exog = exog.merge(bspline_calendar_features, left_index=True, right_index=True, how='left')

        # Holiday features
        if self.holiday_features:
            holiday_features = self._extract_holiday_features(
                index=exog.index,
                location=metadata['location']
            )
            exog = exog.merge(holiday_features, left_index=True, right_index=True, how='left')

        # Windowed features
        if self.exog_window_features:
            exog_window_features = self._extract_windowed_features(
                exog=exog,
                window_features=self.exog_window_features,
            )
            exog = exog.merge(exog_window_features, left_index=True, right_index=True, how='left')

        return target, exog

    def _get_window_features(self, y) -> RollingFeatures | None:
        if isinstance(self.rolling_features, RollingFeatures):
            return self.rolling_features

        if self.window_features is None:
            return None

        stats = []
        window_sizes = []
        for window_size, stat in self.window_features:
            stats.append(stat)
            window_sizes.append(periods_in_duration(y.index, duration=window_size))

        self.rolling_features = RollingFeatures(
            stats=stats,
            window_sizes=window_sizes,
        )
        return self.rolling_features

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
        # Feature engineering
        y, exog = self._prepare_data(
            history=history,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            metadata=metadata,
        )

        # Model specification
        forecaster = ForecasterRecursive(
            regressor=LGBMRegressor(random_state=42, verbose=-1),
            lags=periods_in_duration(y.index, duration=self.lags),
            window_features=self._get_window_features(y=y),
        )

        # Fit the forecaster
        exog_train = exog.loc[y.index, :] if exog is not None else None
        forecaster.fit(y=y, exog=exog_train)

        # Predict the future
        forecast_horizon = create_forecast_index(history, horizon)
        exog_predict = exog.loc[forecast_horizon, :] if exog is not None else None
        prediction = forecaster.predict(steps=horizon, exog=exog_predict)

        # Postprocess forecast
        forecast = (
            prediction
            .to_frame("yhat")
            .rename_axis("timestamp")
            .fillna(y.mean())
        )
        return forecast


env = Env(prefix="ENFOBENCH_MODEL_")

# Instantiate your model
model = SkforecastLightGBMModel(
    model_name=env.str("NAME"),
    lags=env.str("LAGS", default="1D"),
    holiday_features=env.bool("HOLIDAY_FEATURES", default=False),
    calendar_features=env.list("CALENDAR_FEATURES", default=None),
    cyclical_calendar_features=env.list("CYCLICAL_CALENDAR_FEATURES", default=None),
    bspline_calendar_features=env.list("BSPLINE_CALENDAR_FEATURES", default=None),
    exog_window_features=env.list("EXOG_WINDOW_FEATURES", default=None),
    window_features=env.list("WINDOW_FEATURES", default=None),
    exogenous_features=env.list("EXOGENOUS_FEATURES", default=None),
)

# Create a forecast server by passing in your model
app = server_factory(model)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
