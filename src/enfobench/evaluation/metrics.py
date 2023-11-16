import numpy as np
from numpy import ndarray


def check_not_empty(*arrays: ndarray) -> None:
    """Check that none of the arrays are not empty.

    Args:
        *arrays: Objects that will be checked for emptiness.
    """
    if any(array.size == 0 for array in arrays):
        msg = "Found empty array in inputs."
        raise ValueError(msg)


def check_consistent_length(*arrays: ndarray) -> None:
    """Check that all arrays have consistent length.

    Checks whether all input arrays have the same length.

    Args:
        *arrays: Objects that will be checked for consistent length.
    """
    if any(array.ndim != 1 for array in arrays):
        mag = "Found multi dimensional array in inputs."
        raise ValueError(mag)

    lengths = [len(array) for array in arrays]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        msg = f"Found input variables with inconsistent numbers of samples: {lengths}"
        raise ValueError(msg)


def check_has_no_nan(*arrays: ndarray) -> None:
    """Check that all arrays have no NaNs.

    Args:
        *arrays: Objects that will be checked for NaNs.
    """
    for array in arrays:
        if np.isnan(array).any():
            msg = f"Found NaNs in input variables: {array.__repr__()}"
            raise ValueError(msg)


def check_arrays(*arrays: ndarray) -> None:
    """Check that all arrays are valid.

    Args:
        *arrays: Objects that will be checked for validity.
    """
    check_not_empty(*arrays)
    check_consistent_length(*arrays)
    check_has_no_nan(*arrays)


def mean_absolute_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean absolute error regression loss.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
    """
    check_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_bias_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean bias error regression loss.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
    """
    check_arrays(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def mean_squared_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean squared error regression loss.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
    """
    check_arrays(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Root mean squared error regression loss.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
    """
    check_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean absolute percentage error regression loss.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
    """
    check_arrays(y_true, y_pred)
    if np.any(y_true == 0):
        msg = "Found zero in true values. MAPE is undefined."
        raise ValueError(msg)
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))
