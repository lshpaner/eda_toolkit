import numpy as np
import pandas as pd


def _flag_iqr(series: pd.Series, thresh: float):
    """
    Flag outliers in a Series using the IQR method.

    Parameters:
    -----------
    series : pd.Series
        Numeric series to evaluate (NaNs should be dropped before calling).
    thresh : float
        IQR multiplier (e.g. 1.5 for Tukey fences, 3.0 for extreme outliers).

    Returns:
    --------
    low_mask : pd.Series of bool
        True where values are below the lower bound.
    high_mask : pd.Series of bool
        True where values are above the upper bound.
    lower : float
        Computed lower bound.
    upper : float
        Computed upper bound.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - thresh * iqr
    upper = q3 + thresh * iqr
    return series < lower, series > upper, lower, upper


def _flag_zscore(series: pd.Series, thresh: float):
    """
    Flag outliers in a Series using the Z-score method.

    Parameters:
    -----------
    series : pd.Series
        Numeric series to evaluate (NaNs should be dropped before calling).
    thresh : float
        Absolute Z-score cutoff (e.g. 3.0).

    Returns:
    --------
    low_mask : pd.Series of bool
        True where Z-score is below -thresh.
    high_mask : pd.Series of bool
        True where Z-score is above +thresh.
    lower : float
        Computed lower bound (mean - thresh * std), or np.nan if std == 0.
    upper : float
        Computed upper bound (mean + thresh * std), or np.nan if std == 0.
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return (
            pd.Series(False, index=series.index),
            pd.Series(False, index=series.index),
            np.nan,
            np.nan,
        )
    z = (series - mean) / std
    lower = mean - thresh * std
    upper = mean + thresh * std
    return z < -thresh, z > thresh, lower, upper