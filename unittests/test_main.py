import pytest
import pandas as pd
import os
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
    df_with_ids = add_ids(
        sample_dataframe, id_colname="UniqueID", num_digits=5, seed=42
    )
    assert "UniqueID" in df_with_ids.columns
    assert df_with_ids["UniqueID"].nunique() == len(sample_dataframe)


def test_strip_trailing_period(sample_dataframe_with_text):
    df_cleaned = strip_trailing_period(sample_dataframe_with_text, "col")
    expected_values = ["123", "456", "789", None, "text"]
    assert df_cleaned["col"].tolist() == expected_values


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
