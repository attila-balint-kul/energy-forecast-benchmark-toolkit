import numpy as np
from numpy import ndarray


def check_not_empty(*arrays: ndarray) -> None:
    """Check that none of the arrays are not empty.

    Parameters
    ----------
    *arrays: list or tuple of input arrays.
        Objects that will be checked for emptiness.
    """
    if any(X.size == 0 for X in arrays):
        raise ValueError("Found empty array in inputs.")


def check_consistent_length(*arrays: ndarray) -> None:
    """Check that all arrays have consistent length.

    Checks whether all input arrays have the same length.

    Parameters
    ----------
    *arrays : list or tuple of input arrays.
        Objects that will be checked for consistent length.
    """
    if any(X.ndim != 1 for X in arrays):
        raise ValueError("Found multi dimensional array in inputs.")

    lengths = [len(X) for X in arrays]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: {lengths}")


def check_has_no_nan(*arrays: ndarray) -> None:
    """Check that all arrays have no NaNs.

    Parameters
    ----------
    *arrays : list or tuple of input arrays.
        Objects that will be checked for NaNs.
    """
    for X in arrays:
        if np.isnan(X).any():
            raise ValueError(f"Found NaNs in input variables: {X}")


def check_arrays(*arrays: ndarray) -> None:
    """Check that all arrays are valid.

    Parameters
    ----------
    *arrays : list or tuple of input arrays.
        Objects that will be checked for validity.
    """
    check_not_empty(*arrays)
    check_consistent_length(*arrays)
    check_has_no_nan(*arrays)


def mean_absolute_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean absolute error regression loss.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """
    check_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_bias_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean bias error regression loss.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """
    check_arrays(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def mean_squared_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean squared error regression loss.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """
    check_arrays(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Root mean squared error regression loss.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """
    check_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean absolute percentage error regression loss.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """
    check_arrays(y_true, y_pred)
    if np.any(y_true == 0):
        raise ValueError("Found zero in true values. MAPE is undefined.")
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))
