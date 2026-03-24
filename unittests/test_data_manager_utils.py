import pytest
import numpy as np
import pandas as pd

from eda_toolkit._data_manager_utils import _flag_iqr, _flag_zscore
from eda_toolkit.data_manager import _df_to_markdown


# ------------------------------------------------------------------
# _flag_iqr
# ------------------------------------------------------------------

@pytest.fixture
def basic_series():
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -100.0])


def test_flag_iqr_returns_four_values(basic_series):
    result = _flag_iqr(basic_series, thresh=1.5)
    assert len(result) == 4


def test_flag_iqr_low_mask_is_series(basic_series):
    low_mask, _, _, _ = _flag_iqr(basic_series, thresh=1.5)
    assert isinstance(low_mask, pd.Series)
    assert low_mask.dtype == bool


def test_flag_iqr_high_mask_is_series(basic_series):
    _, high_mask, _, _ = _flag_iqr(basic_series, thresh=1.5)
    assert isinstance(high_mask, pd.Series)
    assert high_mask.dtype == bool


def test_flag_iqr_flags_high_outlier(basic_series):
    _, high_mask, _, _ = _flag_iqr(basic_series, thresh=1.5)
    assert high_mask.iloc[-2] == True   # 100.0


def test_flag_iqr_flags_low_outlier(basic_series):
    low_mask, _, _, _ = _flag_iqr(basic_series, thresh=1.5)
    assert low_mask.iloc[-1] == True    # -100.0


def test_flag_iqr_bounds_are_floats(basic_series):
    _, _, lower, upper = _flag_iqr(basic_series, thresh=1.5)
    assert isinstance(lower, float)
    assert isinstance(upper, float)


def test_flag_iqr_upper_greater_than_lower(basic_series):
    _, _, lower, upper = _flag_iqr(basic_series, thresh=1.5)
    assert upper > lower


def test_flag_iqr_no_outliers_in_normal_range():
    series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    low_mask, high_mask, _, _ = _flag_iqr(series, thresh=1.5)
    assert not low_mask.any()
    assert not high_mask.any()


def test_flag_iqr_strict_threshold_flags_more():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 20.0])
    _, high_strict, _, _ = _flag_iqr(series, thresh=0.5)
    _, high_loose, _, _ = _flag_iqr(series, thresh=3.0)
    assert high_strict.sum() >= high_loose.sum()


def test_flag_iqr_constant_series():
    series = pd.Series([5.0, 5.0, 5.0, 5.0])
    low_mask, high_mask, lower, upper = _flag_iqr(series, thresh=1.5)
    assert not low_mask.any()
    assert not high_mask.any()
    assert lower == upper == 5.0


# ------------------------------------------------------------------
# _flag_zscore
# ------------------------------------------------------------------

@pytest.fixture
def zscore_series():
    return pd.Series([0.0, 1.0, -1.0, 0.5, -0.5, 10.0, -10.0])


def test_flag_zscore_returns_four_values(zscore_series):
    result = _flag_zscore(zscore_series, thresh=3.0)
    assert len(result) == 4


def test_flag_zscore_masks_are_series(zscore_series):
    low_mask, high_mask, _, _ = _flag_zscore(zscore_series, thresh=3.0)
    assert isinstance(low_mask, pd.Series)
    assert isinstance(high_mask, pd.Series)


def test_flag_zscore_flags_high_outlier(zscore_series):
    # Z-scores of 10.0 and -10.0 are ~1.72 — use thresh=1.5 to flag them
    _, high_mask, _, _ = _flag_zscore(zscore_series, thresh=1.5)
    assert high_mask.iloc[-2] == True   # 10.0


def test_flag_zscore_flags_low_outlier(zscore_series):
    low_mask, _, _, _ = _flag_zscore(zscore_series, thresh=1.5)
    assert low_mask.iloc[-1] == True    # -10.0


def test_flag_zscore_bounds_are_symmetric():
    series = pd.Series([0.0, 1.0, -1.0, 0.5, -0.5])
    _, _, lower, upper = _flag_zscore(series, thresh=2.0)
    assert abs(abs(lower) - abs(upper)) < 1e-10


def test_flag_zscore_zero_std_returns_no_flags():
    series = pd.Series([3.0, 3.0, 3.0, 3.0])
    low_mask, high_mask, lower, upper = _flag_zscore(series, thresh=3.0)
    assert not low_mask.any()
    assert not high_mask.any()
    assert np.isnan(lower)
    assert np.isnan(upper)


def test_flag_zscore_no_outliers_in_normal_range():
    series = pd.Series([0.0, 0.1, -0.1, 0.2, -0.2])
    low_mask, high_mask, _, _ = _flag_zscore(series, thresh=3.0)
    assert not low_mask.any()
    assert not high_mask.any()


def test_flag_zscore_strict_threshold_flags_more():
    series = pd.Series([0.0, 1.0, -1.0, 0.5, -0.5, 3.5])
    _, high_strict, _, _ = _flag_zscore(series, thresh=1.0)
    _, high_loose, _, _ = _flag_zscore(series, thresh=3.0)
    assert high_strict.sum() >= high_loose.sum()


# ------------------------------------------------------------------
# _df_to_markdown
# ------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "Variable": ["age", "income"],
        "Mean": [35.5, 50000.0],
        "Missing (%)": [0.0, 2.5],
    })


def test_df_to_markdown_returns_string(simple_df):
    result = _df_to_markdown(simple_df)
    assert isinstance(result, str)


def test_df_to_markdown_has_header(simple_df):
    result = _df_to_markdown(simple_df)
    assert "Variable" in result
    assert "Mean" in result
    assert "Missing (%)" in result


def test_df_to_markdown_has_separator(simple_df):
    lines = _df_to_markdown(simple_df).splitlines()
    assert lines[1].replace(" ", "").replace("|", "").replace("-", "") == ""


def test_df_to_markdown_row_count(simple_df):
    result = _df_to_markdown(simple_df)
    lines = result.splitlines()
    # header + separator + 2 data rows
    assert len(lines) == 4


def test_df_to_markdown_contains_values(simple_df):
    result = _df_to_markdown(simple_df)
    assert "age" in result
    assert "35.5" in result


def test_df_to_markdown_empty_string_for_blank():
    df = pd.DataFrame({"A": ["x", ""], "B": [1, ""]})
    result = _df_to_markdown(df)
    assert "x" in result


def test_df_to_markdown_pipe_delimited(simple_df):
    result = _df_to_markdown(simple_df)
    for line in result.splitlines():
        assert line.startswith("|")
        assert line.endswith("|")


def test_df_to_markdown_single_row():
    df = pd.DataFrame({"A": ["val"], "B": [1]})
    result = _df_to_markdown(df)
    lines = result.splitlines()
    assert len(lines) == 3  # header + separator + 1 row


def test_df_to_markdown_single_column():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = _df_to_markdown(df)
    assert isinstance(result, str)
    assert "A" in result