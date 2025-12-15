import numpy as np
import pandas as pd
import scipy.stats as stats
import pytest
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt

from eda_toolkit._plot_utils import (
    _save_figure,
    _add_best_fit,
    _get_label,
    _get_palette,
    _plot_density_overlays,
    _resolve_density_colors,
    _fit_distribution,
    _qq_plot,
    _cdf_exceedance_plot,
)


@pytest.fixture
def simple_data():
    return pd.DataFrame(
        {
            "x": np.random.normal(size=50),
            "group": np.random.choice([0, 1], size=50),
        }
    )


@pytest.fixture
def fig_ax():
    fig, ax = plt.subplots()
    return fig, ax


def test_save_figure_no_filename_noop(fig_ax):
    fig, _ = fig_ax
    _save_figure(fig=fig)


def test_save_figure_no_paths_noop(fig_ax):
    fig, _ = fig_ax
    _save_figure(fig=fig, filename="test")


def test_save_figure_png_and_svg(fig_ax, tmp_path):
    fig, _ = fig_ax

    with patch.object(fig, "savefig") as mock_save:
        _save_figure(
            fig=fig,
            image_path_png=str(tmp_path),
            image_path_svg=str(tmp_path),
            filename="plot",
        )

        assert mock_save.call_count == 2


def test_add_best_fit_with_legend(fig_ax):
    _, ax = fig_ax
    x = np.arange(10)
    y = 2 * x + 1

    _add_best_fit(
        ax=ax,
        x=x,
        y=y,
        linestyle="--",
        linecolor="blue",
        show_legend=True,
    )

    assert ax.legend_ is not None


def test_add_best_fit_without_legend(fig_ax):
    _, ax = fig_ax
    x = np.arange(10)
    y = 2 * x + 1

    _add_best_fit(
        ax=ax,
        x=x,
        y=y,
        linestyle="-",
        linecolor="red",
        show_legend=False,
    )

    assert ax.legend_ is None


def test_get_label_default():
    assert _get_label("x") == "x"


def test_get_label_with_mapping():
    assert _get_label("x", {"x": "X Label"}) == "X Label"


def test_get_palette_length():
    palette = _get_palette(3)
    assert len(palette) == 3


def test_resolve_density_colors_none():
    out = _resolve_density_colors(["kde", "norm"], None)
    assert out == {"kde": None, "norm": None}


def test_resolve_density_colors_str():
    out = _resolve_density_colors(["kde"], "red")
    assert out == {"kde": "red"}


def test_resolve_density_colors_list():
    out = _resolve_density_colors(["kde", "norm"], ["red", "blue"])
    assert out["norm"] == "blue"


def test_resolve_density_colors_list_length_mismatch():
    with pytest.raises(ValueError):
        _resolve_density_colors(["kde", "norm"], ["red"])


def test_resolve_density_colors_dict():
    out = _resolve_density_colors(["kde"], {"kde": "green"})
    assert out["kde"] == "green"


def test_resolve_density_colors_invalid_type():
    with pytest.raises(TypeError):
        _resolve_density_colors(["kde"], 123)


def test_fit_distribution_mle():
    data = np.random.normal(size=100)
    dist, params = _fit_distribution(data, "norm", fit_method="MLE")
    assert callable(dist.pdf)


def test_fit_distribution_mm():
    data = np.random.normal(size=100)
    dist, params = _fit_distribution(data, "norm", fit_method="MM")
    assert params is not None


def test_fit_distribution_invalid_method():
    with pytest.raises(ValueError):
        _fit_distribution(np.array([1, 2, 3]), "norm", fit_method="BAD")


def test_plot_density_overlays_kde(simple_data, fig_ax):
    _, ax = fig_ax

    _plot_density_overlays(
        ax=ax,
        data=simple_data,
        col="x",
        density_function=["kde"],
        density_fit="MLE",
        hue=None,
        log_scale=False,
        density_color=None,
    )


def test_plot_density_overlays_parametric(simple_data, fig_ax):
    _, ax = fig_ax

    _plot_density_overlays(
        ax=ax,
        data=simple_data,
        col="x",
        density_function=["norm"],
        density_fit="MLE",
        hue=None,
        log_scale=False,
        density_color="blue",
    )


def test_plot_density_overlays_invalid_dist(simple_data, fig_ax):
    _, ax = fig_ax

    with pytest.raises(ValueError):
        _plot_density_overlays(
            ax=ax,
            data=simple_data,
            col="x",
            density_function=["not_a_dist"],
            density_fit="MLE",
            hue=None,
            log_scale=False,
            density_color=None,
        )


def test_qq_plot_theoretical(fig_ax):
    _, ax = fig_ax
    data = np.random.normal(size=50)

    _qq_plot(
        ax=ax,
        data=data,
        dist_obj=stats.norm,
        params=stats.norm.fit(data),
        label="norm",
        scale="linear",
        label_fontsize=10,
        tick_fontsize=10,
    )


def test_qq_plot_empirical(fig_ax):
    _, ax = fig_ax
    data = np.random.normal(size=50)
    ref = np.random.normal(size=50)

    _qq_plot(
        ax=ax,
        data=data,
        dist_obj=None,
        params=(),
        label="emp",
        scale="linear",
        label_fontsize=10,
        tick_fontsize=10,
        qq_type="empirical",
        reference_data=ref,
    )


def test_qq_plot_too_few_points(fig_ax):
    _, ax = fig_ax
    with pytest.raises(ValueError):
        _qq_plot(
            ax=ax,
            data=np.array([1.0]),
            dist_obj=None,
            params=(),
            label="bad",
            scale="linear",
            label_fontsize=10,
            tick_fontsize=10,
        )


def test_cdf_exceedance_lower(fig_ax):
    _, ax = fig_ax
    data = np.random.normal(size=100)

    _cdf_exceedance_plot(
        ax=ax,
        data=data,
        dist_obj=stats.norm,
        params=stats.norm.fit(data),
        label="norm",
        scale="linear",
        tail="lower",
        label_fontsize=10,
    )


def test_cdf_exceedance_upper(fig_ax):
    _, ax = fig_ax
    data = np.random.normal(size=100)

    _cdf_exceedance_plot(
        ax=ax,
        data=data,
        dist_obj=stats.norm,
        params=stats.norm.fit(data),
        label="norm",
        scale="linear",
        tail="upper",
        label_fontsize=10,
    )


def test_cdf_exceedance_both(fig_ax):
    _, ax = fig_ax
    data = np.random.normal(size=100)

    _cdf_exceedance_plot(
        ax=ax,
        data=data,
        dist_obj=stats.norm,
        params=stats.norm.fit(data),
        label="norm",
        scale="linear",
        tail="both",
        label_fontsize=10,
    )
