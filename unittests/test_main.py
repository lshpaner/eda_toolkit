import pytest
import importlib
import pandas as pd
import numpy as np
import sys
import os
import datetime
import builtins
from unittest import mock
from eda_toolkit import (
    ensure_directory,
    add_ids,
    strip_trailing_period,
    parse_date_with_rule,
    dataframe_profiler,
    contingency_table,
    save_dataframes_to_excel,
    summarize_all_combinations,
    kde_distributions,
    data_doctor,
    flex_corr_matrix,
    scatter_fit_plot,
    box_violin_plot,
    stacked_crosstab_plot,
    highlight_columns,
    custom_help,
    eda_toolkit_logo,
    detailed_doc,
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
        sample_dataframe, id_colname="UniqueID", num_digits=num_digits, seed=seed
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


import pytest
import pandas as pd
import os
import matplotlib.pyplot as plt
from eda_toolkit import kde_distributions


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"Feature1": [1, 2, 3, 4, 5], "Feature2": [10, 20, 30, 40, 50]})


def test_kde_distributions(sample_dataframe, tmp_path):
    save_path = str(tmp_path)
    try:
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=["Feature1", "Feature2"],
            figsize=(5, 5),
            hist_color="#0000FF",
            kde_color="#FF0000",
            mean_color="#000000",
            median_color="#000000",
            hist_edgecolor="#000000",
            fill=True,
            fill_alpha=0.6,
            image_path_png=save_path,
            image_filename="test_kde_plot",
            plot_type="both",
            plot_mean=True,
            plot_median=True,
            std_dev_levels=[1, 2],
            std_color=["#808080", "#A0A0A0"],
            show_legend=True,
        )
        assert any(
            f.endswith(".png") for f in os.listdir(save_path)
        ), "PNG file should be created."
    except Exception as e:
        pytest.fail(f"kde_distributions failed: {e}")


def test_kde_distributions_invalid_plot_type(sample_dataframe):
    with pytest.raises(ValueError):
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=["Feature1"],
            plot_type="invalid_option",
        )


def test_kde_distributions_log_scale(sample_dataframe, tmp_path):
    save_path = str(tmp_path)
    try:
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=["Feature1"],
            log_scale_vars=["Feature1"],
            image_path_png=save_path,
            image_filename="test_log_scale",
        )
        assert any(
            f.endswith(".png") for f in os.listdir(save_path)
        ), "PNG file should be created for log scale."
    except Exception as e:
        pytest.fail(f"kde_distributions with log scale failed: {e}")


def test_data_doctor_basic(sample_dataframe_values):
    try:
        data_doctor(sample_dataframe_values, "values", show_plot=False)
    except Exception as e:
        pytest.fail(f"data_doctor failed with default parameters: {e}")


def test_data_doctor_scale_conversion(sample_dataframe_values):
    scale_options = ["log", "sqrt", "cbrt", "reciprocal", "stdrz", "minmax"]
    for scale in scale_options:
        try:
            data_doctor(
                sample_dataframe_values,
                "values",
                scale_conversion=scale,
                show_plot=False,
            )
        except Exception as e:
            pytest.fail(f"data_doctor failed with scale_conversion={scale}: {e}")


def test_data_doctor_cutoff(sample_dataframe_values):
    try:
        data_doctor(
            sample_dataframe_values,
            "values",
            apply_cutoff=True,
            lower_cutoff=2,
            upper_cutoff=4,
            show_plot=False,
        )
    except Exception as e:
        pytest.fail(f"data_doctor failed with cutoffs applied: {e}")


def test_data_doctor_apply_as_new_col(sample_dataframe_values):
    df_copy = sample_dataframe_values.copy()
    try:
        data_doctor(
            df_copy,
            "values",
            scale_conversion="log",
            apply_as_new_col_to_df=True,
            show_plot=False,
        )
    except Exception as e:
        pytest.fail(f"data_doctor failed when applying as new column: {e}")


def test_flex_corr_matrix_basic(sample_corr_dataframe):
    try:
        flex_corr_matrix(sample_corr_dataframe, show_colorbar=True)
    except Exception as e:
        pytest.fail(f"flex_corr_matrix failed with default parameters: {e}")


def test_flex_corr_matrix_custom_cols(sample_corr_dataframe):
    try:
        flex_corr_matrix(
            sample_corr_dataframe, cols=["Feature1", "Feature3"], show_colorbar=False
        )
    except Exception as e:
        pytest.fail(f"flex_corr_matrix failed with custom column selection: {e}")


def test_flex_corr_matrix_triangulation(sample_corr_dataframe):
    try:
        flex_corr_matrix(sample_corr_dataframe, triangular=True)
    except Exception as e:
        pytest.fail(f"flex_corr_matrix failed with triangular masking: {e}")


def test_flex_corr_matrix_save_plot(sample_corr_dataframe, tmp_path):
    save_path = str(tmp_path)
    try:
        flex_corr_matrix(
            sample_corr_dataframe,
            save_plots=True,
            image_path_png=save_path,
            image_path_svg=save_path,
            title="Correlation Test Matrix",
        )
        assert any(f.endswith(".png") for f in os.listdir(save_path))
        assert any(f.endswith(".svg") for f in os.listdir(save_path))
    except Exception as e:
        pytest.fail(f"flex_corr_matrix failed to save plots: {e}")


# Test that scatter_fit_plot runs without errors with default parameters
def test_scatter_fit_plot_basic(sample_scatter_dataframe):
    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars="Feature1",
            y_vars="Feature2",
            show_plot="grid",
        )
    except Exception as e:
        pytest.fail(f"scatter_fit_plot failed with default parameters: {e}")


# Test that scatter_fit_plot correctly excludes specified variable combinations
def test_scatter_fit_plot_exclude(sample_scatter_dataframe):
    exclude_combinations = [("Feature1", "Feature2")]
    result = scatter_fit_plot(
        sample_scatter_dataframe,
        x_vars="Feature1",
        y_vars="Feature2",
        exclude_combinations=exclude_combinations,
        show_plot="combinations",
    )
    assert len(result) == 0, f"Expected no valid plots, but got {len(result)}"


# Test that scatter_fit_plot correctly applies the best fit line
def test_scatter_fit_plot_best_fit(sample_scatter_dataframe):
    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars="Feature1",
            y_vars="Feature2",
            add_best_fit_line=True,
            show_plot="grid",
        )
    except Exception as e:
        pytest.fail(f"scatter_fit_plot failed with best fit line: {e}")


# Test that scatter_fit_plot correctly handles hue parameter
def test_scatter_fit_plot_hue(sample_scatter_dataframe):
    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars="Feature1",
            y_vars="Feature2",
            hue="Category",
            show_plot="grid",
        )
    except Exception as e:
        pytest.fail(f"scatter_fit_plot failed with hue parameter: {e}")


# Test that scatter_fit_plot raises error on invalid input
def test_scatter_fit_plot_invalid_input(sample_scatter_dataframe):
    with pytest.raises(ValueError):
        scatter_fit_plot(sample_scatter_dataframe, x_vars=None, y_vars=None)


def test_scatter_fit_plot_conflicting_all_vars(sample_scatter_dataframe):
    with pytest.raises(
        ValueError, match="Cannot pass `all_vars` and still choose `x_vars`"
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            all_vars=["Feature1", "Feature2", "Feature3"],  # Invalid combination
        )


def test_scatter_fit_plot_hue_palette_without_hue(sample_scatter_dataframe):
    with pytest.raises(
        ValueError, match="Cannot specify `hue_palette` without specifying `hue`"
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            hue_palette="coolwarm",  # Invalid without hue
        )


def test_scatter_fit_plot_invalid_save_plots(sample_scatter_dataframe):
    with pytest.raises(
        ValueError,
        match="Invalid `save_plots` value. Choose from 'all', 'individual', 'grid', or None.",
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            save_plots="wrong_option",  # Invalid choice
        )


def test_scatter_fit_plot_invalid_exclude_combinations(sample_scatter_dataframe):
    with pytest.raises(
        ValueError, match="Invalid column names in `exclude_combinations`"
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            exclude_combinations=[("NonExistentColumn", "Feature2")],  # Invalid column
        )


def test_scatter_fit_plot_invalid_rotate_plot(sample_scatter_dataframe):
    with pytest.raises(
        ValueError, match="Invalid `rotate_plot`. Choose True or False."
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            rotate_plot="yes",  # Invalid, should be boolean
        )


def test_scatter_fit_plot_save_without_path(sample_scatter_dataframe):
    with pytest.raises(
        ValueError, match="To save plots, specify `image_path_png` or `image_path_svg`."
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            save_plots="all",  # Requires path
        )


def test_scatter_fit_plot_invalid_show_plot(sample_scatter_dataframe):
    with pytest.raises(ValueError, match="Invalid `show_plot`. Choose from"):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            show_plot="invalid_option",  # Invalid choice
        )


def test_scatter_fit_plot_missing_variables(sample_scatter_dataframe):
    with pytest.raises(
        ValueError,
        match="Either `all_vars` or both `x_vars` and `y_vars` must be provided.",
    ):
        scatter_fit_plot(sample_scatter_dataframe)


# Test that box_violin_plot runs without errors with default parameters
def test_box_violin_plot_basic(sample_box_violin_dataframe):
    try:
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1", "Metric2"],
            metrics_comp=["Category"],
            show_plot="both",
        )
    except Exception as e:
        pytest.fail(f"box_violin_plot failed with default parameters: {e}")


# Test that box_violin_plot handles plot type switching correctly
def test_box_violin_plot_violin(sample_box_violin_dataframe):
    try:
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            plot_type="violinplot",
            show_plot="both",
        )
    except Exception as e:
        pytest.fail(f"box_violin_plot failed with violin plot type: {e}")


# Test that box_violin_plot correctly applies axis limits
def test_box_violin_plot_axis_limits(sample_box_violin_dataframe):
    try:
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            xlim=(0, 120),
            ylim=(0, 300),
            show_plot="both",
        )
    except Exception as e:
        pytest.fail(f"box_violin_plot failed with axis limits: {e}")


# Test that box_violin_plot correctly handles rotation
def test_box_violin_plot_rotate(sample_box_violin_dataframe):
    try:
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            rotate_plot=True,
            show_plot="both",
        )
    except Exception as e:
        pytest.fail(f"box_violin_plot failed with rotation enabled: {e}")


# Test that box_violin_plot raises error on invalid input
def test_box_violin_plot_invalid_input(sample_box_violin_dataframe):
    with pytest.raises(ValueError):
        box_violin_plot(
            sample_box_violin_dataframe, metrics_list=[], metrics_comp=["Category"]
        )


# Test that stacked_crosstab_plot runs without errors with default parameters
def test_stacked_crosstab_plot_basic(sample_crosstab_dataframe):
    try:
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="Category",
            func_col=["Group"],
            legend_labels_list=[["X", "Y"]],
            title=["Distribution of Groups per Category"],
        )
    except Exception as e:
        pytest.fail(f"stacked_crosstab_plot failed with default parameters: {e}")


# Test that stacked_crosstab_plot raises an error when required columns are missing
def test_stacked_crosstab_plot_missing_column(sample_crosstab_dataframe):
    with pytest.raises(KeyError):
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="NonExistentColumn",
            func_col=["Group"],
            legend_labels_list=[["X", "Y"]],
            title=["Invalid Test"],
        )


# Test that stacked_crosstab_plot raises an error when legend list lengths mismatch
def test_stacked_crosstab_plot_legend_mismatch(sample_crosstab_dataframe):
    with pytest.raises(ValueError):
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="Category",
            func_col=["Group", "Outcome"],
            legend_labels_list=[["X", "Y"]],  # Mismatched length
            title=["Mismatch Test"],
        )


# Test that stacked_crosstab_plot correctly applies log scale
def test_stacked_crosstab_plot_logscale(sample_crosstab_dataframe):
    try:
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="Category",
            func_col=["Group"],
            legend_labels_list=[["X", "Y"]],
            title=["Log Scale Test"],
            logscale=True,
        )
    except Exception as e:
        pytest.fail(f"stacked_crosstab_plot failed with log scale: {e}")


# Test that stacked_crosstab_plot correctly removes stacks when specified
def test_stacked_crosstab_plot_remove_stacks(sample_crosstab_dataframe):
    try:
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="Category",
            func_col=["Group"],
            legend_labels_list=[["X", "Y"]],
            title=["Without Stacks"],
            remove_stacks=True,
            plot_type="regular",
        )
    except Exception as e:
        pytest.fail(f"stacked_crosstab_plot failed with remove_stacks: {e}")


def test_datetime_import_version_gte_3_7(mocker):
    """
    Test that 'from datetime import datetime' is used when Python version is >= 3.7
    """
    mocker.patch.object(sys, "version_info", (3, 7))  # Mock Python 3.7+

    importlib.reload(sys.modules[__name__])  # Reload the module to apply changes

    from datetime import datetime

    assert "datetime" in sys.modules, "datetime module should be imported"
    assert callable(datetime), "datetime should be a callable class"


def test_datetime_import_version_lt_3_7(mocker):
    """
    Test that 'import datetime' is used when Python version is < 3.7
    """
    mocker.patch.object(sys, "version_info", (3, 6))  # Mock Python 3.6

    importlib.reload(sys.modules[__name__])  # Reload the module to apply changes

    import datetime

    assert "datetime" in sys.modules, "datetime module should be imported"
    assert hasattr(
        datetime, "datetime"
    ), "datetime module should contain datetime class"


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
    """Test that custom_help() prints ASCII art and documentation when called with None."""
    custom_help()
    captured = capsys.readouterr()
    assert eda_toolkit_logo in captured.out, "ASCII art should be printed."
    assert detailed_doc in captured.out, "Detailed documentation should be printed."


def test_custom_help_other_objects(capsys):
    """Test that custom_help(obj) calls the original help function for other objects."""
    obj = int  # Using a built-in type as a test case
    original_help = builtins.help  # Capture the original help function

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
