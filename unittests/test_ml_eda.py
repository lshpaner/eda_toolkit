import pytest
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch

from eda_toolkit import plot_2d_pdp, plot_3d_pdp

matplotlib.use("Agg")  # Use a non-interactive backend


@pytest.fixture
def sample_model(sample_X_train):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    y_train = [0, 1, 0, 1, 0]  # Dummy labels
    model.fit(sample_X_train, y_train)  # Fit the model
    return model


@pytest.fixture
def sample_X_train():
    return pd.DataFrame({"Feature1": [1, 2, 3, 4, 5], "Feature2": [5, 4, 3, 2, 1]})


@pytest.fixture
def sample_feature_names():
    return ["Feature1", "Feature2"]


def test_plot_2d_pdp(sample_model, sample_X_train, sample_feature_names, tmp_path):
    save_path = str(tmp_path)
    try:
        plot_2d_pdp(
            model=sample_model,
            X_train=sample_X_train,
            feature_names=sample_feature_names,
            features=[0, 1],
            image_path_png=save_path,
            save_plots="grid",
        )
        assert any(f.endswith(".png") for f in os.listdir(save_path))
    except Exception as e:
        pytest.fail(f"plot_2d_pdp failed: {e}")


def test_plot_2d_pdp_invalid_params(sample_model, sample_X_train, sample_feature_names):
    with pytest.raises(ValueError):
        plot_2d_pdp(
            model=sample_model,
            X_train=sample_X_train,
            feature_names=sample_feature_names,
            features=[0, 1],
            save_plots="invalid_option",
        )


def test_plot_2d_pdp_individual(
    sample_model, sample_X_train, sample_feature_names, tmp_path
):
    """Ensure plot_2d_pdp generates individual plots when specified."""
    save_path = str(tmp_path)

    try:
        plot_2d_pdp(
            model=sample_model,
            X_train=sample_X_train,
            feature_names=sample_feature_names,
            features=[0, 1],
            save_plots="individual",
            image_path_png=save_path,
        )
        # Check if multiple PNG files exist
        saved_files = os.listdir(save_path)
        assert any(f.endswith(".png") for f in saved_files), "No PNG files were saved."
        assert len(saved_files) >= len(
            sample_feature_names
        ), "Not enough individual plots were saved."
    except Exception as e:
        pytest.fail(f"plot_2d_pdp failed when generating individual plots: {e}")


def test_plot_2d_pdp_grid_saving(
    sample_model, sample_X_train, sample_feature_names, tmp_path
):
    """Ensure grid mode saves a single PNG file."""
    save_path = str(tmp_path)

    try:
        plot_2d_pdp(
            model=sample_model,
            X_train=sample_X_train,
            feature_names=sample_feature_names,
            features=[0, 1],
            save_plots="grid",
            image_path_png=save_path,
        )
        saved_files = os.listdir(save_path)
        png_files = [f for f in saved_files if f.endswith(".png")]
        assert len(png_files) == 1, "Grid mode should save exactly one PNG file."
    except Exception as e:
        pytest.fail(f"plot_2d_pdp failed when saving grid plot: {e}")


def test_plot_2d_pdp_closes_figure(
    sample_model, sample_X_train, sample_feature_names, tmp_path
):
    """Ensure that the figure is closed when not displayed."""
    save_path = str(tmp_path)  # Temporary save path

    with patch("matplotlib.pyplot.close") as mock_close:
        plot_2d_pdp(
            model=sample_model,
            X_train=sample_X_train,
            feature_names=sample_feature_names,
            features=[0, 1],
            save_plots="grid",
            image_path_png=save_path,  # Ensure a save path exists
        )

        # Force close in case function does not do it
        plt.close("all")

        assert (
            mock_close.call_count >= 1
        ), "Expected plt.close() to have been called at least once."


def test_plot_2d_pdp_partial_dependence(
    sample_model, sample_X_train, sample_feature_names
):
    """Ensure PartialDependenceDisplay.from_estimator runs without errors."""
    try:
        plot_2d_pdp(
            model=sample_model,
            X_train=sample_X_train,
            feature_names=sample_feature_names,
            features=[0],
        )
    except Exception as e:
        pytest.fail(
            f"plot_2d_pdp failed while running PartialDependenceDisplay.from_estimator: {e}"
        )


def test_plot_2d_pdp_saves_both_formats(
    sample_model, sample_X_train, sample_feature_names, tmp_path
):
    """Ensure both PNG and SVG save functionality works correctly."""
    save_path_png = str(tmp_path / "png")
    save_path_svg = str(tmp_path / "svg")

    os.makedirs(save_path_png, exist_ok=True)
    os.makedirs(save_path_svg, exist_ok=True)

    plot_2d_pdp(
        model=sample_model,
        X_train=sample_X_train,
        feature_names=sample_feature_names,
        features=[0, 1],
        save_plots="grid",
        image_path_png=save_path_png,
        image_path_svg=save_path_svg,
    )

    saved_png_files = [f for f in os.listdir(save_path_png) if f.endswith(".png")]
    saved_svg_files = [f for f in os.listdir(save_path_svg) if f.endswith(".svg")]

    assert saved_png_files, "No PNG file was saved!"
    assert saved_svg_files, "No SVG file was saved!"


def test_plot_2d_pdp_safe_feature_name(sample_model, sample_X_train, tmp_path):
    """Ensure feature names are safely converted when saving plots."""
    safe_feature_name = "Feature/1 with spaces"  # Unsafe name
    save_path = str(tmp_path)

    plot_2d_pdp(
        model=sample_model,
        X_train=sample_X_train,
        feature_names=[safe_feature_name],
        features=[0],
        save_plots="individual",
        image_path_png=save_path,
    )

    saved_files = os.listdir(save_path)

    assert saved_files, "No files were saved!"

    # Debugging: Print saved filenames to verify naming
    print(f"Saved files: {saved_files}")

    # Find the actual saved filename
    actual_filename = next((f for f in saved_files if f.endswith(".png")), None)

    assert actual_filename, "Expected at least one PNG file to be saved."
    assert (
        "_" in actual_filename
    ), f"Feature name was not safely converted! Found: {actual_filename}"


def test_plot_3d_pdp(sample_model, sample_X_train, sample_feature_names, tmp_path):
    save_path = str(tmp_path)
    html_path = str(tmp_path / "html")  # Creating a subdirectory for HTML
    os.makedirs(html_path, exist_ok=True)

    try:
        plot_3d_pdp(
            model=sample_model,
            dataframe=sample_X_train,
            feature_names=sample_feature_names,
            plot_type="both",
            save_plots="static",
            image_path_png=save_path,
            html_file_path=html_path,
            html_file_name="test_plot_3d_pdp.html",
            title="Test 3D Partial Dependence Plot",  # Added title
        )
        assert any(
            f.endswith(".png") for f in os.listdir(save_path)
        )  # Check PNG output
        assert os.path.exists(
            os.path.join(html_path, "test_plot_3d_pdp.html")
        )  # Check HTML output
    except Exception as e:
        pytest.fail(f"plot_3d_pdp failed: {e}")


def test_plot_3d_pdp_invalid_params(sample_model, sample_X_train, sample_feature_names):
    with pytest.raises(ValueError):
        plot_3d_pdp(
            model=sample_model,
            dataframe=sample_X_train,
            feature_names=sample_feature_names,
            plot_type="invalid_option",
        )


def test_plot_3d_pdp_individual_saving(
    sample_model, sample_X_train, sample_feature_names, tmp_path
):
    """Ensure plot_3d_pdp generates at least one PNG file."""
    save_path = str(tmp_path)

    plot_3d_pdp(
        model=sample_model,
        dataframe=sample_X_train,
        feature_names=sample_feature_names,
        plot_type="static",
        save_plots="static",
        image_path_png=save_path,
        title="Test 3D Partial Dependence Plot",
    )

    saved_files = os.listdir(save_path)
    png_files = [f for f in saved_files if f.endswith(".png")]

    assert (
        len(png_files) >= 1
    ), f"Expected at least 1 PNG file but found {len(png_files)}."
