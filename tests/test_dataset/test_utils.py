import numpy as np
import pandas as pd

from enfobench.datasets.utils import create_perfect_forecasts_from_covariates


def test_create_perfect_forecasts_from_covariates():
    index = pd.date_range(start="2020-01-01", end="2020-10-02 13:54:00", freq="1H")
    past_covariates = pd.DataFrame(
        index=index,
        data=np.random.rand(len(index), 2),
        columns=["covariate_1", "covariate_2"],
    )
    past_covariates.drop(index[2], inplace=True)

    future_covariates = create_perfect_forecasts_from_covariates(
        past_covariates,
        start=pd.Timestamp("2020-01-01"),
        step=pd.Timedelta("1D"),
        horizon=pd.Timedelta("7D"),
    )

    assert isinstance(future_covariates, pd.DataFrame)
    assert "covariate_1" in future_covariates.columns
    assert "covariate_2" in future_covariates.columns
    assert "timestamp" in future_covariates.columns
    assert "cutoff_date" in future_covariates.columns
