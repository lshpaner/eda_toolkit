import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
from typing import Optional, List, Dict, Union, Tuple


################################################################################
# Figure Saving Utility
################################################################################


def _save_figure(
    *,
    fig: Optional[plt.Figure] = None,
    image_path_png: Optional[str] = None,
    image_path_svg: Optional[str] = None,
    filename: Optional[str] = None,
    bbox_inches: str = "tight",
    dpi: Optional[int] = None,
) -> None:
    """
    Save a matplotlib figure to PNG and/or SVG.

    This helper centralizes all figure-saving logic. Callers provide
    output directories and a base filename. Full file paths are
    constructed internally.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or None
        Figure to save. If None, uses the current active figure.

    image_path_png : str or None
        Directory path where the PNG file should be saved.
        If None, PNG output is skipped.

    image_path_svg : str or None
        Directory path where the SVG file should be saved.
        If None, SVG output is skipped.

    filename : str or None
        Base filename without extension. If None, no files are saved.

    bbox_inches : str, optional
        Bounding box option passed to savefig.

    dpi : int or None, optional
        DPI for raster outputs such as PNG.
    """
    # Nothing to do
    if filename is None or (image_path_png is None and image_path_svg is None):
        return

    fig = fig or plt.gcf()

    if image_path_png:
        png_path = os.path.join(image_path_png, f"{filename}.png")
        fig.savefig(
            png_path,
            bbox_inches=bbox_inches,
            dpi=dpi,
        )

    if image_path_svg:
        svg_path = os.path.join(image_path_svg, f"{filename}.svg")
        fig.savefig(
            svg_path,
            bbox_inches=bbox_inches,
        )


################################################################################
# Best-Fit Line Utilitys
################################################################################


def _add_best_fit(
    *,
    ax,
    x,
    y,
    linestyle,
    linecolor,
    show_legend: bool,
    legend_loc: str = "best",
) -> None:
    """
    Add a linear least-squares best-fit line to an existing Axes.

    This utility computes a first-order (linear) polynomial fit using
    ``numpy.polyfit`` and overlays the resulting line on the provided
    Matplotlib Axes. The fitted equation is added as the line label and
    the legend is optionally shown or removed.
    """

    m, b = np.polyfit(x, y, 1)

    ax.plot(
        x,
        m * x + b,
        color=linecolor,
        linestyle=linestyle,
        label=f"y = {m:.2f}x + {b:.2f}",
    )

    if show_legend:
        ax.legend(loc=legend_loc)
    else:
        if ax.legend_ is not None:
            ax.legend_.remove()


################################################################################
# Labeling and Palette Utilities
################################################################################


def _get_label(var: str, label_names: Optional[Dict[str, str]] = None) -> str:
    """
    Return a display label for a variable.

    If label_names is provided and contains the variable, return the mapped
    label. Otherwise return the variable name.
    """
    if not label_names:
        return var

    ## used, for example, in scatter_fit_plot
    return label_names.get(var, var)


def _get_palette(n_colors):
    """
    Returns a 'tab10' color palette with the specified number of colors.
    """
    return sns.color_palette("tab10", n_colors=n_colors)


################################################################################
# Density Overlay Plotting Utils
################################################################################


def _plot_density_overlays(
    *,
    ax,
    data: pd.DataFrame,
    col: str,
    density_function: List[str],
    density_fit: str,
    hue: Optional[str],
    log_scale: bool,
    density_color: Optional[Union[str, List[str], Dict[str, str]]],
    **kwargs,
) -> None:
    """
    Plot density overlays (KDE and/or parametric PDFs).

    Supports:
    - single color for all densities
    - list of colors aligned with density_function
    - dict mapping {density_name: color}
    """

    x = data[col].dropna().values
    if len(x) <= 1:
        return

    x_grid = np.linspace(x.min(), x.max(), 500)

    # ------------------------------------------------------------------
    # Normalize density colors
    # ------------------------------------------------------------------
    if density_color is None:
        color_map = {d: None for d in density_function}

    elif isinstance(density_color, str):
        color_map = {d: density_color for d in density_function}

    elif isinstance(density_color, list):
        if len(density_color) != len(density_function):
            raise ValueError(
                "When density_color is a list, its length must match "
                "density_function."
            )
        color_map = dict(zip(density_function, density_color))

    elif isinstance(density_color, dict):
        color_map = {d: density_color.get(d) for d in density_function}

    else:
        raise TypeError(
            "density_color must be a str, list[str], dict[str, str], or None."
        )

    # ------------------------------------------------------------------
    # Plot density overlays
    # ------------------------------------------------------------------
    for dist_name in density_function:
        curve_color = color_map.get(dist_name)

        # KDE
        if dist_name == "kde":
            sns.kdeplot(
                data=data,
                x=col,
                ax=ax,
                hue=hue,
                color=curve_color if hue is None else None,
                log_scale=log_scale,
                label="kde",
                **kwargs,
            )
            continue

        # Parametric distributions
        if not hasattr(stats, dist_name):
            raise ValueError(
                f"Unknown density_function '{dist_name}'. "
                "Use 'kde' or a valid scipy.stats distribution name "
                "(e.g., 'norm', 'lognorm', 'gamma')."
            )

        dist = getattr(stats, dist_name)

        try:
            params = dist.fit(x) if density_fit == "MLE" else dist.fit(x, method="MM")
            pdf = dist.pdf(x_grid, *params)

            ax.plot(
                x_grid,
                pdf,
                label=dist_name,
                color=curve_color,
            )

        except Exception as e:
            warnings.warn(
                f"Could not fit '{dist_name}' for '{col}': {e}",
                UserWarning,
            )


################################################################################
# Resolve Density Colors
################################################################################


def _resolve_density_colors(
    density_function: List[str],
    density_color: Optional[Union[str, List[str], Dict[str, str]]],
) -> Dict[str, Optional[str]]:

    if density_color is None:
        return {d: None for d in density_function}

    if isinstance(density_color, str):
        return {d: density_color for d in density_function}

    if isinstance(density_color, list):
        if len(density_color) != len(density_function):
            raise ValueError(
                "When density_color is a list, its length must match "
                "density_function."
            )
        return dict(zip(density_function, density_color))

    if isinstance(density_color, dict):
        return {d: density_color.get(d) for d in density_function}

    raise TypeError("density_color must be a str, list[str], dict[str, str], or None.")


################################################################################
# Distribution Fitting Utilities
################################################################################


def _fit_distribution(
    data: np.ndarray,
    dist_name: str,
    fit_method: str = "MLE",
):
    """
    Fit a scipy.stats distribution using MLE or Method of Moments.
    """
    dist = getattr(stats, dist_name)

    if fit_method == "MLE":
        params = dist.fit(data)
    elif fit_method == "MM":
        params = dist.fit(data, method="MM")
    else:
        raise ValueError("fit_method must be 'MLE' or 'MM'")

    return dist, params


################################################################################
# Quantile–Quantile Plotting Utilities
################################################################################


def _qq_plot(
    ax,
    data: np.ndarray,
    dist_obj,
    params: Tuple,
    label: str,
    scale: str,
    label_fontsize: int,
    tick_fontsize: int,
    qq_type: str = "theoretical",
    reference_data: Optional[np.ndarray] = None,
    show_reference: bool = True,
    color: Optional[str] = None,
):
    """
    Quantile–Quantile plot.

    qq_type:
        - "theoretical": sample vs fitted distribution
        - "empirical": sample vs reference_data
    """

    # ---------------------------
    # Validation
    # ---------------------------
    if data is None or len(data) < 2:
        raise ValueError("QQ plot requires at least 2 data points.")

    if qq_type == "empirical":
        if reference_data is None or len(reference_data) < 2:
            raise ValueError(
                "Empirical QQ plot requires reference_data with >= 2 values."
            )

    # ---------------------------
    # Quantiles
    # ---------------------------
    if qq_type == "theoretical":
        osm, osr = stats.probplot(
            data,
            dist=dist_obj,
            sparams=params,
            plot=None,
        )[0]
        xlabel = "Theoretical Quantiles"

    else:  # empirical
        q = np.linspace(0.01, 0.99, 100)
        osm = np.quantile(reference_data, q)
        osr = np.quantile(data, q)
        xlabel = "Reference Quantiles"

    # ---------------------------
    # Scatter
    # ---------------------------
    ax.scatter(
        osm,
        osr,
        s=15,
        alpha=0.7,
        color=color,
        label=label,
    )

    # ---------------------------
    # Reference line (theoretical only)
    # ---------------------------
    if qq_type == "theoretical" and show_reference:
        ax.plot(
            [osm.min(), osm.max()],
            [osm.min(), osm.max()],
            linestyle="--",
            color=color,
            alpha=0.5,
            label=f"{label} reference",
        )

    # ---------------------------
    # Formatting
    # ---------------------------
    if scale == "log":
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel("Sample Quantiles", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)


################################################################################
# CDF and Exceedance Plotting Utilities
################################################################################


def _cdf_exceedance_plot(
    ax,
    data: np.ndarray,
    dist_obj,
    params: Tuple,
    label: str,
    scale: str,
    tail: str,
    label_fontsize: int,
    color: Optional[str] = None,
):
    """
    Plot CDF or exceedance probability with optional log scaling.
    """
    x = np.sort(data)
    cdf = dist_obj.cdf(x, *params)

    # Determine which curves to plot and the y-axis label
    if tail == "lower":
        y = cdf
        ax.plot(x, y, label=label, color=color)
        ylabel = "CDF"

    elif tail == "upper":
        y = 1 - cdf
        ax.plot(x, y, label=label, color=color)
        ylabel = "Exceedance Probability"

    else:  # both
        ax.plot(x, cdf, label=f"{label} CDF", alpha=0.8, color=color)
        ax.plot(
            x,
            1 - cdf,
            label=f"{label} Exceedance",
            alpha=0.8,
            linestyle="--",
            color=color,
        )
        ylabel = "Probability"

    # Apply scale BEFORE setting labels
    if scale == "log":
        ax.set_yscale("log")

    # Set axis labels with explicit font size
    ax.set_xlabel("x", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
