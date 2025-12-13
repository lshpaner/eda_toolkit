import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import warnings


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
