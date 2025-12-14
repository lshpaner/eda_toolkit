import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import warnings

################################################################################
# Density Overlay Plotting Utils
################################################################################


def _plot_density_overlays(
    *,
    ax,
    data: pd.DataFrame,
    col: str,
    density_function: list[str],
    density_fit: str,
    hue: str | None,
    log_scale: bool,
    density_color: str | list[str] | dict[str, str] | None,
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
    density_function: list[str],
    density_color: str | list[str] | dict[str, str] | None,
) -> dict[str, str | None]:
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
    params: tuple,
    label: str,
    scale: str,
    label_fontsize: int,
    tick_fontsize: int,
    qq_type: str = "theoretical",
    reference_data: np.ndarray | None = None,
    show_reference: bool = True,
    color: str | None = None,
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
    params: tuple,
    label: str,
    scale: str,
    tail: str,
    label_fontsize: int,
    color: str | None = None,
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
