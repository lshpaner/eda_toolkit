import pytest
from collections import Counter
import pandas as pd
import numpy as np
import sys
import os
import datetime
import builtins
from unittest.mock import MagicMock, patch
from eda_toolkit import (
    ensure_directory,
    generate_table1,
    table1_to_str,
    add_ids,
    strip_trailing_period,
    parse_date_with_rule,
    dataframe_profiler,
    contingency_table,
    save_dataframes_to_excel,
    summarize_all_combinations,
    highlight_columns,
    custom_help,
    eda_toolkit_logo,
    detailed_doc,
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {"Feature1": [1, 2, 3, 4, 5], "Feature2": [10, 20, 30, 40, 50]},
    )


@pytest.fixture
def sample_dataframe_values():
    return pd.DataFrame({"values": [1, 2, 3, 4, 5]})


@pytest.fixture
def sample_dataframes():
    return {
        "Sheet1": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        "Sheet2": pd.DataFrame({"X": [7, 8, 9], "Y": [10, 11, 12]}),
    }


@pytest.fixture
def sample_dataframe_with_text():
    return pd.DataFrame({"col": ["123.", "456", "789.", None, "text."]})


@pytest.fixture
def sample_dataframe_dates():
    return pd.DataFrame({"dates": ["25/12/2023", "12/25/2023", "05/06/2023"]})


@pytest.fixture
def sample_dataframe_profiler():
    return pd.DataFrame({"col1": [1, 2, 2, None], "col2": ["A", "B", "B", "A"]})


@pytest.fixture
def sample_dataframe_contingency():
    return pd.DataFrame(
        {"Category": ["A", "B", "A", "B", "C"], "Values": [1, 2, 1, 2, 3]}
    )


@pytest.fixture
def sample_categorical_dataframe():
    return pd.DataFrame(
        {
            "Category": ["A", "A", "B", "B", "C"],
            "Subcategory": ["X", "Y", "X", "Y", "X"],
            "Value": [10, 15, 20, 25, 30],
        }
    )


@pytest.fixture
def sample_corr_dataframe():
    return pd.DataFrame(
        {
            "Feature1": [1, 2, 3, 4, 5],
            "Feature2": [5, 4, 3, 2, 1],
            "Feature3": [2, 3, 4, 5, 6],
        }
    )


@pytest.fixture
def sample_scatter_dataframe():
    """Create a sample DataFrame for scatter plots."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Feature1": np.random.rand(50) * 10,
            "Feature2": np.random.rand(50) * 20,
            "Category": np.random.choice(["A", "B"], size=50),
        }
    )


@pytest.fixture
def sample_box_violin_dataframe():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Metric1": np.random.rand(50) * 100,
            "Metric2": np.random.rand(50) * 200,
            "Category": np.random.choice(["A", "B", "C"], size=50),
        }
    )


@pytest.fixture
def sample_crosstab_dataframe():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Category": np.random.choice(["A", "B", "C"], size=50),
            "Group": np.random.choice(["X", "Y"], size=50),
            "Outcome": np.random.choice(["Success", "Failure"], size=50),
        }
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, None, 40],
            "income": [50000, None, 60000, 80000, 75000],
            "gender": ["Male", "Female", "Female", "Male", None],
            "binary_flag": [1, 0, 1, 0, 1],
            "is_married": [True, False, True, True, False],
        }
    )


def test_ensure_directory(tmp_path):
    dir_path = tmp_path / "test_dir"
    ensure_directory(str(dir_path))
    assert dir_path.exists()


def test_save_dataframes_to_excel(sample_dataframes, tmp_path):
    file_path = tmp_path / "test_output.xlsx"

    try:
        # Debugging
        print(f"DEBUG: file_path = {file_path}, type = {type(file_path)}")
        assert isinstance(
            file_path, (str, os.PathLike)
        ), "file_path must be a string or Path-like object"

        save_dataframes_to_excel(
            file_path,  # Ensure file path is first argument
            sample_dataframes,  # Ensure dataframes dictionary is passed correctly
        )

        assert file_path.exists()  # Ensure the file was created
    except Exception as e:
        pytest.fail(f"save_dataframes_to_excel failed: {e}")


def test_add_ids(sample_dataframe):
    num_digits = 5
    seed = 42

    df_with_ids = add_ids(
        sample_dataframe,
        id_colname="UniqueID",
        num_digits=num_digits,
        seed=seed,
    )

    # Check if UniqueID column is present
    assert "UniqueID" in df_with_ids.columns, "Column 'UniqueID' was not added"

    # Check that all UniqueID values are unique
    assert df_with_ids["UniqueID"].nunique() == len(
        sample_dataframe
    ), "Duplicate IDs found"

    # Check that the length of each ID matches num_digits
    assert (
        df_with_ids["UniqueID"].apply(lambda x: len(str(x)) == num_digits).all()
    ), f"Some IDs do not have {num_digits} digits"

    # Check if the function is deterministic (same seed should generate same IDs)
    df_with_ids_2 = add_ids(
        sample_dataframe, id_colname="UniqueID", num_digits=num_digits, seed=seed
    )

    assert (
        df_with_ids["UniqueID"].tolist() == df_with_ids_2["UniqueID"].tolist()
    ), "IDs generated with the same seed do not match"


def test_add_ids_exceed_pool_size():
    """
    Test that a ValueError is raised when the number of rows exceeds the pool
    of unique IDs.
    """

    # Set num_digits such that the pool size is too small for the given DataFrame
    num_digits = 3  # Pool size = 9 * (10^2) = 900 unique IDs
    pool_size = 9 * (10 ** (num_digits - 1))  # Pool size = 900
    n_rows = pool_size + 1  # Exceed the pool size

    # Create a DataFrame with more rows than the pool size
    df = pd.DataFrame({"data": range(n_rows)})

    # Expect a ValueError when trying to add IDs
    with pytest.raises(
        ValueError,
        match=r"The number of rows \(\d+\) exceeds the total pool of possible unique IDs",
    ):
        add_ids(df, num_digits=num_digits)


def test_strip_trailing_period(sample_dataframe_with_text):
    """Test that strip_trailing_period removes trailing periods correctly."""

    df_cleaned = strip_trailing_period(sample_dataframe_with_text, "col")

    # Define expected results (ensure None values match actual Pandas NaN behavior)
    expected_values = ["123", "456", "789", np.nan, "text"]

    # Convert column to list and check each value (handling NaN cases)
    cleaned_values = df_cleaned["col"].tolist()
    for actual, expected in zip(cleaned_values, expected_values):
        if pd.isna(expected):
            assert pd.isna(actual), f"Expected NaN but got {actual}"
        else:
            assert actual == expected, f"Expected {expected} but got {actual}"

    # Ensure the function does not modify other columns
    df_original = sample_dataframe_with_text.copy()
    df_cleaned_other = strip_trailing_period(df_original, "col")

    assert df_cleaned_other.drop(columns=["col"]).equals(
        df_original.drop(columns=["col"])
    ), "Function should not modify other columns"

    # Ensure type consistency (all values should remain str or NaN)
    assert all(
        isinstance(val, (str, float)) or pd.isna(val) for val in df_cleaned["col"]
    ), "All values should remain as strings or NaN"


def test_parse_date_with_rule(sample_dataframe_dates):
    assert parse_date_with_rule(sample_dataframe_dates.iloc[0, 0]) == "2023-12-25"
    assert parse_date_with_rule(sample_dataframe_dates.iloc[1, 0]) == "2023-12-25"
    assert parse_date_with_rule(sample_dataframe_dates.iloc[2, 0]) == "2023-06-05"


def test_dataframe_profiler(sample_dataframe_profiler):
    profile = dataframe_profiler(sample_dataframe_profiler, return_df=True)
    assert "column" in profile.columns
    assert "null_total" in profile.columns


def test_contingency_table(sample_dataframe_contingency):
    table = contingency_table(sample_dataframe_contingency, cols="Category")
    assert "Total" in table.columns
    assert "Percentage" in table.columns


def test_summarize_all_combinations(sample_categorical_dataframe, tmp_path):
    try:
        data_path = str(tmp_path)
        data_name = "summary.xlsx"
        summary_tables, all_combinations = summarize_all_combinations(
            df=sample_categorical_dataframe,
            variables=["Category", "Subcategory"],
            data_path=data_path,
            data_name=data_name,
            min_length=2,
        )

        assert isinstance(summary_tables, dict), "Output should be a dictionary"
        assert all(
            isinstance(df, pd.DataFrame) for df in summary_tables.values()
        ), "Each summary should be a DataFrame"
        assert all_combinations, "All combinations list should not be empty"
        assert os.path.exists(
            os.path.join(data_path, data_name)
        ), "Excel file should be created"

    except Exception as e:
        pytest.fail(f"summarize_all_combinations failed: {e}")


def test_datetime_import_version_gte_3_7(mocker):
    """Test datetime import behavior for Python versions >= 3.7"""
    mocker.patch.object(sys, "version_info", (3, 7))  # Mock Python 3.7+

    # Import datetime AFTER patching to ensure it behaves as expected
    from datetime import datetime

    assert "datetime" in sys.modules, "datetime module should be imported"
    assert hasattr(
        datetime, "fromisoformat"
    ), "datetime should support 'fromisoformat' in Python 3.7+"


def test_datetime_import_version_lt_3_7():
    """Test datetime import behavior for Python versions < 3.7."""

    # Mock `sys.version_info` to simulate Python 3.6
    with patch.object(sys, "version_info", (3, 6)):

        # Create a MagicMock instance for datetime.datetime
        mock_datetime = MagicMock()
        del (
            mock_datetime.fromisoformat
        )  # Remove `fromisoformat` to simulate Python <3.7

        # Patch `datetime.datetime` with our mocked version
        with patch("datetime.datetime", new=mock_datetime):
            assert not hasattr(
                datetime.datetime, "fromisoformat"
            ), "datetime should NOT support 'fromisoformat' in Python <3.7"


def test_ensure_directory_creates_new(mocker, tmp_path):
    """
    Test ensure_directory creates the directory when it does not exist.
    """
    dir_path = tmp_path / "new_dir"

    # Mock os.path.exists to return False initially
    mocker.patch("os.path.exists", return_value=False)
    mock_makedirs = mocker.patch("os.makedirs")

    ensure_directory(str(dir_path))

    # Ensure makedirs was called since the directory didn’t exist
    mock_makedirs.assert_called_once_with(str(dir_path))


def test_ensure_directory_already_exists(mocker, tmp_path):
    """
    Test ensure_directory when directory already exists.
    """
    dir_path = tmp_path / "existing_dir"

    # Mock os.path.exists to return True
    mocker.patch("os.path.exists", return_value=True)
    mock_makedirs = mocker.patch("os.makedirs")

    ensure_directory(str(dir_path))

    # Ensure makedirs was NOT called since the directory exists
    mock_makedirs.assert_not_called()


def test_highlight_columns():
    # Create a sample DataFrame
    data = {"A": [1, 2], "B": [3, 4], "C": [5, 6]}
    df = pd.DataFrame(data)

    # Apply highlight_columns function
    styled_df = highlight_columns(df, columns=["A", "C"], color="lightblue")

    # Ensure the return type is a pandas Styler
    assert isinstance(
        styled_df, pd.io.formats.style.Styler
    ), "Function should return a Styler object"

    # Extract the applied styles
    styles = styled_df._compute().ctx  # Extracts computed styles as a dictionary

    # Convert column names to positions
    col_positions = {col: pos for pos, col in enumerate(df.columns)}

    # Verify the expected styles
    expected_styles = {
        (0, col_positions["A"]): [("background-color", "lightblue")],
        (1, col_positions["A"]): [("background-color", "lightblue")],
        (0, col_positions["C"]): [("background-color", "lightblue")],
        (1, col_positions["C"]): [("background-color", "lightblue")],
    }

    for key, value in expected_styles.items():
        assert (
            styles.get(key) == value
        ), f"Expected {value} at {key}, but got {styles.get(key)}"

    # Ensure column "B" remains unstyled
    assert all(
        styles.get((i, col_positions["B"]), []) == [] for i in range(len(df))
    ), "Column B should not be styled"


def test_custom_help_module(capsys):
    """
    Test that custom_help() prints ASCII art and documentation when called with None.
    """
    custom_help()
    captured = capsys.readouterr()
    assert eda_toolkit_logo in captured.out, "ASCII art should be printed."
    assert detailed_doc in captured.out, "Detailed documentation should be printed."


def test_custom_help_other_objects(capsys):
    """Test that custom_help(obj) calls the original help function for other objects."""
    obj = int  # Using a built-in type as a test case
    custom_help = builtins.help  # Capture the original help function

    custom_help(obj)
    captured = capsys.readouterr()

    # Verify that ASCII art and custom docs are NOT printed
    assert eda_toolkit_logo not in captured.out
    assert detailed_doc not in captured.out

    # Ensure original help() output is present
    assert (
        "class int" in captured.out or "Help on class int" in captured.out
    ), "Original help should be invoked for int."


def test_custom_help_override():
    """Test that builtins.help is overridden by custom_help."""
    assert (
        builtins.help == custom_help
    ), "builtins.help should be overridden by custom_help."


def test_returns_dataframe(sample_df):
    result = generate_table1(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert "Variable" in result.columns


def test_returns_markdown_only(sample_df):
    result = generate_table1(
        sample_df,
        export_markdown=True,
        return_markdown_only=True,
    )
    assert isinstance(result, str)
    assert "| Variable |" in result


def test_exports_markdown_to_file(sample_df, tmp_path):
    md_path = tmp_path / "table1.md"
    generate_table1(
        sample_df,
        export_markdown=True,
        markdown_path=str(md_path),
    )
    assert md_path.exists()


def test_detect_binary_numeric_moves_column(sample_df):
    result = generate_table1(sample_df)
    assert any("binary_flag" in str(v) for v in result["Variable"])


def test_disable_detect_binary_keeps_in_continuous(sample_df):
    result = generate_table1(sample_df, detect_binary_numeric=False)
    assert any(
        (row["Variable"] == "binary_flag" and row["Type"] == "Continuous")
        for _, row in result.iterrows()
    )


def test_value_counts_false_shows_summary(sample_df):
    result = generate_table1(sample_df, value_counts=False)
    assert not any("=" in v for v in result["Variable"])


def test_value_counts_true_shows_detailed(sample_df):
    result = generate_table1(sample_df, value_counts=True)
    assert any("=" in v for v in result["Variable"])


def test_max_categories_limits_output(sample_df):
    result = generate_table1(
        sample_df,
        value_counts=True,
        max_categories=1,
    )
    detailed_vars = [v for v in result["Variable"] if "=" in v]
    # Group by prefix (i.e., variable name before the "=")

    prefixes = [v.split(" = ")[0] for v in detailed_vars]
    counts_by_var = Counter(prefixes)

    # Ensure no variable has more than 1 category
    assert all(c <= 1 for c in counts_by_var.values())


def test_handles_all_empty_df():
    empty_df = pd.DataFrame()
    result = generate_table1(empty_df)
    assert result.empty


def test_manual_column_specification(sample_df):
    result = generate_table1(
        sample_df,
        categorical_cols=["gender"],
        continuous_cols=["income"],
        detect_binary_numeric=False,
    )
    assert any(
        row["Variable"] == "income" and row["Type"] == "Continuous"
        for _, row in result.iterrows()
    )
    assert any("gender" in v for v in result["Variable"])


def test_basic_table_formatting():
    df = pd.DataFrame(
        {
            "Variable": ["age", "gender"],
            "Type": ["Continuous", "Categorical"],
            "Mean": [29.12345, None],
            "SD": [5.98765, None],
        }
    )

    output = table1_to_str(df)
    assert isinstance(output, str)
    assert "age" in output
    assert "Continuous" in output
    assert "29.12" in output  # Default float_precision = 2
    assert "5.99" in output  # Rounded


def test_padding_applied():
    df = pd.DataFrame({"Col": ["val"]})
    padded = table1_to_str(df, padding=4)
    assert padded.startswith("    Col    ")  # 4 spaces on each side


def test_max_col_width_truncates():
    df = pd.DataFrame({"ThisIsAVeryLongColumnName": ["ThisIsAVeryLongValue"]})
    output = table1_to_str(df, max_col_width=10)
    assert "ThisIsAVer" in output  # truncated column
    assert "ThisIsAVer" in output  # truncated value


def test_empty_dataframe():
    df = pd.DataFrame()
    output = table1_to_str(df)
    assert output == "[Empty Table]"


def test_none_dataframe():
    output = table1_to_str(None)
    assert output == "[Empty Table]"


def test_high_precision():
    df = pd.DataFrame(
        {
            "X": [1.123456],
            "Y": [2.987654],
        }
    )
    output = table1_to_str(df, float_precision=4)
    assert "1.1235" in output
    assert "2.9877" in output


def test_mixed_dtypes():
    df = pd.DataFrame(
        {"Name": ["Alice", "Bob"], "Age": [30, 40], "Score": [87.6789, 93.1234]}
    )
    output = table1_to_str(df, float_precision=1)
    assert "Alice" in output
    assert "Bob" in output
    assert "87.7" in output
    assert "93.1" in output


def test_include_types_continuous_only(sample_df):
    result = generate_table1(sample_df, include_types="continuous")
    assert result["Type"].nunique() == 1
    assert all(result["Type"] == "Continuous")


def test_include_types_categorical_only(sample_df):
    result = generate_table1(sample_df, include_types="categorical")
    assert result["Type"].nunique() == 1
    assert all(result["Type"] == "Categorical")


def test_include_types_both_returns_mixed(sample_df):
    result = generate_table1(sample_df, include_types="both")
    assert set(result["Type"].unique()).issubset({"Continuous", "Categorical"})


def test_include_types_invalid_raises():
    with pytest.raises(
        ValueError,
        match="`include_types` must be 'continuous', 'categorical', or 'both'",
    ):
        generate_table1(
            pd.DataFrame({"a": [1, 2, 3]}),
            include_types="invalid_option",
        )
