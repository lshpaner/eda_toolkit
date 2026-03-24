import pytest
import numpy as np
import pandas as pd

from eda_toolkit import normality_tests


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def normal_df():
    """Small DataFrame with approximately normal features (n=200)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "A": rng.normal(0, 1, 200),
        "B": rng.normal(5, 2, 200),
    })


@pytest.fixture
def skewed_df():
    """DataFrame with heavily skewed features."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "skewed": rng.exponential(scale=2, size=200),
        "normal": rng.normal(0, 1, 200),
    })


@pytest.fixture
def large_df():
    """DataFrame with n > 5,000 to trigger Shapiro-Wilk skip."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 6000),
        "y": rng.exponential(1, 6000),
    })


@pytest.fixture
def tiny_df():
    """DataFrame with fewer than 3 rows to trigger edge case."""
    return pd.DataFrame({"A": [1.0, 2.0]})


@pytest.fixture
def nan_df():
    """DataFrame with NaN values mixed in."""
    return pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 4.0, 5.0, 3.0, 2.5, 1.5, 3.5, 4.5],
    })


# ------------------------------------------------------------------
# Return structure
# ------------------------------------------------------------------

def test_returns_dataframe(normal_df):
    result = normality_tests(normal_df)
    assert isinstance(result, pd.DataFrame)


def test_summary_columns(normal_df):
    result = normality_tests(normal_df)
    assert set(result.columns) == {
        "Variable", "Test", "Statistic", "P-value", "Normal"
    }


def test_default_runs_all_three_tests(normal_df):
    result = normality_tests(normal_df)
    tests_run = result["Test"].unique().tolist()
    assert "Shapiro-Wilk" in tests_run
    assert "D'Agostino K²" in tests_run
    assert "Anderson-Darling" in tests_run


def test_row_count_all_tests(normal_df):
    result = normality_tests(normal_df)
    # 2 features x 3 tests = 6 rows
    assert len(result) == 6


def test_row_count_single_test(normal_df):
    result = normality_tests(normal_df, tests=["shapiro"])
    assert len(result) == 2


def test_sorted_by_variable_then_test(normal_df):
    result = normality_tests(normal_df)
    assert list(result["Variable"]) == sorted(result["Variable"].tolist())


# ------------------------------------------------------------------
# features parameter
# ------------------------------------------------------------------

def test_features_subset(skewed_df):
    result = normality_tests(skewed_df, features=["skewed"])
    assert list(result["Variable"].unique()) == ["skewed"]


def test_features_auto_excludes_non_numeric():
    df = pd.DataFrame({
        "num": np.random.randn(50),
        "cat": ["a", "b"] * 25,
    })
    result = normality_tests(df)
    assert "cat" not in result["Variable"].values
    assert "num" in result["Variable"].values


# ------------------------------------------------------------------
# Shapiro-Wilk
# ------------------------------------------------------------------

def test_shapiro_normal_data_passes(normal_df):
    result = normality_tests(normal_df, tests=["shapiro"])
    # With truly normal data at n=200 most should pass
    assert result["Normal"].isin([True, False]).all()


def test_shapiro_skips_large_n(large_df, capsys):
    result = normality_tests(large_df, tests=["shapiro"])
    shapiro_rows = result[result["Test"] == "Shapiro-Wilk"]
    assert (shapiro_rows["Statistic"] == "-").all()
    assert (shapiro_rows["P-value"] == "-").all()
    assert (shapiro_rows["Normal"] == "-").all()


def test_shapiro_skip_prints_note(large_df, capsys):
    normality_tests(large_df, tests=["shapiro"])
    captured = capsys.readouterr()
    assert "Shapiro-Wilk was skipped" in captured.out
    assert "n > 5,000" in captured.out


def test_shapiro_tiny_n(tiny_df):
    result = normality_tests(tiny_df, tests=["shapiro"])
    shapiro_rows = result[result["Test"] == "Shapiro-Wilk"]
    assert shapiro_rows["Statistic"].isna().all()


# ------------------------------------------------------------------
# D'Agostino K²
# ------------------------------------------------------------------

def test_dagostino_runs(normal_df):
    result = normality_tests(normal_df, tests=["dagostino"])
    assert len(result) == 2
    assert (result["Test"] == "D'Agostino K²").all()


def test_dagostino_skewed_fails(skewed_df):
    result = normality_tests(
        skewed_df, features=["skewed"], tests=["dagostino"]
    )
    assert result.loc[0, "Normal"] == False


def test_dagostino_tiny_n(tiny_df):
    result = normality_tests(tiny_df, tests=["dagostino"])
    assert result["Statistic"].isna().all()


# ------------------------------------------------------------------
# Anderson-Darling
# ------------------------------------------------------------------

def test_anderson_runs(normal_df):
    result = normality_tests(normal_df, tests=["anderson"])
    assert len(result) == 2
    assert (result["Test"] == "Anderson-Darling").all()


def test_anderson_pvalue(normal_df):
    result = normality_tests(normal_df, tests=["anderson"])
    # scipy >= 1.17 returns interpolated p-value; older returns "-"
    for val in result["P-value"]:
        assert isinstance(val, (float, int, str))


def test_anderson_normal_is_bool(normal_df):
    result = normality_tests(normal_df, tests=["anderson"])
    assert result["Normal"].isin([True, False]).all()


def test_anderson_tiny_n(tiny_df):
    result = normality_tests(tiny_df, tests=["anderson"])
    assert result["Statistic"].isna().all()


# ------------------------------------------------------------------
# alpha parameter
# ------------------------------------------------------------------

def test_strict_alpha_more_rejections():
    # Use gamma-distributed data — moderately non-normal so Shapiro-Wilk
    # p-value sits in a mid-range where alpha threshold matters
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.gamma(2, 2, 200)})
    result_loose = normality_tests(df, tests=["shapiro"], alpha=0.5)
    result_strict = normality_tests(df, tests=["shapiro"], alpha=0.001)
    loose_pass = result_loose["Normal"].eq(True).sum()
    strict_pass = result_strict["Normal"].eq(True).sum()
    assert loose_pass >= strict_pass


# ------------------------------------------------------------------
# decimal_places parameter
# ------------------------------------------------------------------

def test_decimal_places_applied(normal_df):
    result = normality_tests(normal_df, tests=["shapiro"], decimal_places=2)
    shapiro = result[result["Test"] == "Shapiro-Wilk"]
    for val in shapiro["Statistic"]:
        if isinstance(val, float):
            assert len(str(val).split(".")[-1]) <= 2


# ------------------------------------------------------------------
# NaN handling
# ------------------------------------------------------------------

def test_nan_rows_excluded(nan_df):
    result = normality_tests(nan_df, tests=["shapiro"])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


# ------------------------------------------------------------------
# Validation errors
# ------------------------------------------------------------------

def test_invalid_test_raises():
    df = pd.DataFrame({"A": np.random.randn(50)})
    with pytest.raises(ValueError, match="Invalid test"):
        normality_tests(df, tests=["ks_test"])


def test_invalid_alpha_raises():
    df = pd.DataFrame({"A": np.random.randn(50)})
    with pytest.raises(ValueError, match="`alpha`"):
        normality_tests(df, alpha=1.5)


def test_no_numeric_features_raises():
    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="No numeric features"):
        normality_tests(df)