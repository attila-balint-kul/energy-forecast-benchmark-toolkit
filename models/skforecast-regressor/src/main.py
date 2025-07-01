from typing import Literal

import holidays
import pandas as pd
from environs import Env
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.timeseries.forecasting import WindowFeatures
from marshmallow.validate import OneOf
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures

from enfobench import ModelInfo, AuthorInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import periods_in_duration, create_forecast_index

REGRESSOR_CHOICES = ["LGBMRegressor", "LinearRegression",]
SCALER_CHOICES = [
    'StandardScaler',
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "NormalQuantileTransformer",
]

CalendarFeature = Literal[
    'month',
    'quarter',
    'semester',
    'year',
    'week',
    'day_of_week',
    'day_of_month',
    'day_of_year',
    'weekend',
    'month_start',
    'month_end',
    'quarter_start',
    'quarter_end',
    'year_start',
    'year_end',
    'leap_year',
    'days_in_month',
    'hour',
    'minute',
    'second',
]

CyclicalCalendarFeature = Literal[
    'month',
    'quarter',
    'semester',
    'week',
    'day_of_week',
    'day_of_month',
    'day_of_year',
    'hour',
    'minute',
    'second',
]

BSplineCalendarFeature = Literal[
    "month",
    "quarter",
    "week",
    "day_of_week",
    "day_of_month",
    "hour",
]

WindowStats = Literal['min', 'max', 'mean', 'std', 'sum']


class SkforecastLightGBMModel:

    def __init__(
        self,
        regressor: str,
        model_name: str,
        lags: str = '1D',
        calendar_features: list[CalendarFeature] | None = None,
        cyclical_calendar_features: list[CyclicalCalendarFeature] | None = None,
        bspline_calendar_features: list[BSplineCalendarFeature] | None = None,
        holiday_features: bool = False,
        y_scaler: str | None = None,
        exog_scaler: str | None = None,
        window_features: list[tuple, str, WindowStats] | None = None,
        exogenous_features: list[str] | Literal['all'] | None = None,
        exog_window_features: list[tuple[str, str, WindowStats]] | None = None,
        polynomial_features: dict | None = None,
    ):
        self.regressor = regressor
        self.model_name = model_name
        self.lags = lags
        self.calendar_features = calendar_features or []
        self.cyclical_calendar_features = cyclical_calendar_features or []
        self.bspline_calendar_features = bspline_calendar_features or []
        self.holiday_features = holiday_features
        self.y_scaler = y_scaler
        self.exog_scaler = exog_scaler
        self.window_features = window_features
        self.exogenous_features = exogenous_features
        self.exog_window_features = exog_window_features or []
        self.rolling_features = None
        self.poly_features = polynomial_features or {}

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Skforecast-{self.regressor}-{self.lags}-{self.model_name}",
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
                "y_scaler": self.y_scaler,
                "exogenous_scaler": self.exog_scaler,
                "window_features": self.window_features,
                "exogenous_features": self.exogenous_features,
                "exog_window_features": self.exog_window_features,
                "rolling_features": self.rolling_features,
                "poly_features": self.poly_features,
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

        periods_in_a_day = periods_in_duration(index, duration=pd.Timedelta(days=1))
        holiday_features['holiday_previous_day'] = holiday_features['holiday'].shift(periods_in_a_day).fillna(0).astype(
            int
        )
        holiday_features['holiday_next_day'] = holiday_features['holiday'].shift(-periods_in_a_day).fillna(0).astype(
            int
        )

        # Extracting working day feature
        holiday_features['working_day'] = (holiday_features.index.dayofweek < 5 & ~holiday_features.holiday).astype(int)
        return holiday_features

    @staticmethod
    def _extract_calendar_features(index: pd.DatetimeIndex, calendar_features: list[CalendarFeature]) -> pd.DataFrame:
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

    @staticmethod
    def _extract_polynomial_features(
        exog: pd.DataFrame,
        columns: list[str],
        degree: int,
        interaction_only: bool,
        include_bias: bool,
    ) -> pd.DataFrame:
        if not columns:
            raise ValueError("No columns provided.")

        transformer_poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        ).set_output(transform="pandas")

        poly_features = transformer_poly.fit_transform(exog[columns])
        poly_features = poly_features.drop(columns=columns)
        poly_features.columns = [f"poly_{col.replace(" ", "__")}" for col in poly_features.columns]
        return poly_features

    def _prepare_data(
        self,
        forecast_index: pd.DatetimeIndex,
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

        exog = (
            past_covariates
            .combine_first(future_covariates.drop(columns=['cutoff_date']))
            .resample(freq).asfreq()
            .interpolate()
        )

        selected_exog = pd.DataFrame(index=history.index.union(forecast_index))
        if self.exogenous_features is not None:
            if 'all' in self.exogenous_features:
                weather_features = exog.copy()
            elif isinstance(self.exogenous_features, list):
                weather_features = exog.loc[:, self.exogenous_features].copy()
            else:
                msg = f"Could not parse exogenous features: {self.exogenous_features}"
                raise ValueError(msg)
            selected_exog = selected_exog.merge(weather_features, left_index=True, right_index=True, how="left")

        if self.calendar_features:
            calendar_features = self._extract_calendar_features(
                index=selected_exog.index,
                calendar_features=self.calendar_features,
            )
            selected_exog = selected_exog.merge(calendar_features, left_index=True, right_index=True, how='left')

        if self.cyclical_calendar_features:
            cyclical_calendar_features = self._extract_cyclical_calendar_features(
                index=selected_exog.index,
                calendar_features=self.cyclical_calendar_features,
            )
            selected_exog = selected_exog.merge(
                cyclical_calendar_features, left_index=True, right_index=True, how='left'
            )

        if self.bspline_calendar_features:
            bspline_calendar_features = self._extract_bspline_calendar_features(
                index=selected_exog.index,
                calendar_features=self.bspline_calendar_features,
            )
            selected_exog = selected_exog.merge(
                bspline_calendar_features, left_index=True, right_index=True, how='left'
            )

        if self.holiday_features:
            holiday_features = self._extract_holiday_features(
                index=selected_exog.index,
                location=metadata['location']
            )
            selected_exog = selected_exog.merge(holiday_features, left_index=True, right_index=True, how='left')

        if self.exog_window_features:
            exog_window_features = self._extract_windowed_features(
                exog=exog,
                window_features=self.exog_window_features,
            )
            selected_exog = selected_exog.merge(exog_window_features, left_index=True, right_index=True, how='left')

        if self.poly_features:
            exog_poly_features = self._extract_polynomial_features(
                exog=selected_exog,
                columns=self.poly_features.get('columns', "").split(','),
                degree=int(self.poly_features.get('degree', 2)),
                interaction_only=bool(self.poly_features.get('interaction_only', True)),
                include_bias=bool(self.poly_features.get('include_bias', False)),
            )
            selected_exog = selected_exog.merge(exog_poly_features, left_index=True, right_index=True, how='left')

        return target, selected_exog

    def _get_window_features(self, y) -> RollingFeatures | None:
        if self.window_features is None:
            return None

        stats = []
        window_sizes = []
        for window_size, stat in self.window_features:
            stats.append(stat)
            window_sizes.append(periods_in_duration(y.index, duration=window_size))

        rolling_features = RollingFeatures(
            stats=stats,
            window_sizes=window_sizes,
        )
        return rolling_features

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
        forecast_index = create_forecast_index(history, horizon=horizon)

        # Feature engineering
        y, exog = self._prepare_data(
            forecast_index=forecast_index,
            history=history,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            metadata=metadata,
        )

        forecaster = ForecasterRecursive(
            regressor=self._get_regressor(self.regressor),
            lags=periods_in_duration(y.index, duration=self.lags),
            window_features=self._get_window_features(y=y),
            transformer_y=self._get_scaler(self.y_scaler),
            transformer_exog=self._get_scaler(self.exog_scaler)
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

    @staticmethod
    def _get_regressor(regressor: str):
        # Model specification
        if regressor == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        elif regressor == 'LGBMRegressor':
            from lightgbm import LGBMRegressor
            return LGBMRegressor(random_state=42, verbose=-1)

        msg = f"Unknown regressor: {regressor}"
        raise ValueError(msg)

    @staticmethod
    def _get_scaler(scaler: str | None):
        if scaler is None:
            return None
        elif scaler == "StandardScaler":
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()
        elif scaler == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
        elif scaler == "MaxAbsScaler":
            from sklearn.preprocessing import MaxAbsScaler
            return MaxAbsScaler()
        elif scaler == "RobustScaler":
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        elif scaler == "NormalQuantileTransformer":
            from sklearn.preprocessing import QuantileTransformer
            return QuantileTransformer(output_distribution='normal', random_state=0)

        msg = f"Unknown scaler: {scaler}"
        raise ValueError(msg)


env = Env(prefix="ENFOBENCH_MODEL_")


# Register a new parser method for paths
@env.parser_for("exog_window_feature")
def exog_window_feature_parser(value):
    window_features = value.split(";") if value != "" else []

    parsed = []
    for window_feature in window_features:
        window_config = window_feature.split(",", maxsplit=2)
        parsed.append(tuple(window_config))
    return parsed


@env.parser_for("window_feature")
def window_feature_parser(value):
    window_features = value.split(";") if value != "" else []

    parsed = []
    for window_feature in window_features:
        window_config = window_feature.split(",", maxsplit=1)
        parsed.append(tuple(window_config))
    return parsed


# Instantiate your model
model = SkforecastLightGBMModel(
    model_name=env.str("NAME"),
    regressor=env.str(
        "REGRESSOR",
        default="LGBMRegressor",
        validate=OneOf(REGRESSOR_CHOICES, error="ENFOBENCH_MODEL_REGRESSOR must be one of: {choices}"),
    ),
    lags=env.str("LAGS", default="1D"),
    holiday_features=env.bool("HOLIDAY_FEATURES", default=False),
    calendar_features=env.list("CALENDAR_FEATURES", default=None),
    cyclical_calendar_features=env.list("CYCLICAL_CALENDAR_FEATURES", default=None),
    bspline_calendar_features=env.list("BSPLINE_CALENDAR_FEATURES", default=None),
    y_scaler=env.str(
        "Y_SCALER",
        default=None,
        validate=OneOf(SCALER_CHOICES, error="ENFOBENCH_MODEL_Y_SCALER must be one of: {choices}"),
    ),
    exog_scaler=env.str(
        "EXOGENOUS_SCALER",
        default=None,
        validate=OneOf(SCALER_CHOICES, error="ENFOBENCH_MODEL_Y_SCALER must be one of: {choices}"),
    ),
    window_features=env.window_feature("WINDOW_FEATURES", default=""),
    exogenous_features=env.list("EXOGENOUS_FEATURES", default=None),
    exog_window_features=env.exog_window_feature("EXOGENOUS_WINDOW_FEATURES", default=""),
    polynomial_features=env.dict("POLYNOMIAL_FEATURES", default=None, delimiter=";"),
)

# Create a forecast server by passing in your model
app = server_factory(model)
