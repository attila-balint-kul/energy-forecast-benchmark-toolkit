import numpy as np
from pandas import Series


def hampel(series: Series, window_size: int = 5, n_sigmas: float = 3.0) -> tuple[Series, Series]:
    """Apply the Hampel filter to a series.

    The Hampel filter is generally used to detect anomalies in data with a timeseries structure.
    It basically consists of a sliding window of a parameterizable size. For each window, each observation
    will be compared with the Median Absolute Deviation (MAD). The observation will be considered an outlier
    in the case in which it exceeds the MAD by n times (the parameter n is also parameterizable).

    Args:
        series: Series to filter.
        window_size: The size of the moving window for outlier detection (default is 5).
        n_sigmas: The number of standard deviations for outlier detection (default is 3.0).
    """
    # make a copy of the series to avoid modifying the original
    series = series.copy()

    scaling_constant = 1.4826  # for Gaussian distribution

    rolling_median = series.rolling(window=window_size, center=True).median()
    rolling_median_absolute_deviation = scaling_constant * series.rolling(window=window_size, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )

    deviation_from_median_absolute_deviation = np.abs(series - rolling_median)

    # select values that deviate more than n_sigma from the mad and replace them with the median
    mask = deviation_from_median_absolute_deviation > (n_sigmas * rolling_median_absolute_deviation)
    series[mask] = rolling_median[mask]
    return series, mask
