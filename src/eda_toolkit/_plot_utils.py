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
    density_color: str | None,
    **kwargs,
) -> None:
    """
    Plot density overlays (KDE and/or parametric PDFs) using a single
    consistent color for all density curves.
    """

    x = data[col].dropna().values
    if len(x) <= 1:
        return

    x_grid = np.linspace(x.min(), x.max(), 500)

    for dist_name in density_function:
        # KDE
        if dist_name == "kde":
            sns.kdeplot(
                data=data,
                x=col,
                ax=ax,
                hue=hue,
                color=density_color if hue is None else None,
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
                color=density_color,
            )

        except Exception as e:
            warnings.warn(
                f"Could not fit '{dist_name}' for '{col}': {e}",
                UserWarning,
            )
