import pytest
import pandas as pd
import os
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend

from eda_toolkit import plot_2d_pdp, plot_3d_pdp


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
