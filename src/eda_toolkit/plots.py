################################################################################
############################### Library Imports ################################
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import itertools  # Import itertools for combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib.ticker as mticker  # Import for formatting
import seaborn as sns
import textwrap
import os
import warnings
from sklearn.preprocessing import PowerTransformer, RobustScaler
from tqdm import tqdm

################################################################################
############################ KDE Distribution Plots ############################
################################################################################


def kde_distributions(
    df,
    vars_of_interest=None,
    figsize=(5, 5),  # Unified figsize parameter
    subplot_figsize=None,  # Size of the overall subplot grid
    hist_color="#0000FF",  # Default color blue as hex code
    kde_color="#FF0000",  # Default color red as hex code
    mean_color="#000000",
    median_color="#000000",
    hist_edgecolor="#000000",  # Default edge color black as hex code
    hue=None,  # Added hue parameter
    fill=True,  # Added fill parameter
    fill_alpha=1,  # Transparency level for the fill
    n_rows=None,
    n_cols=None,
    w_pad=1.0,
    h_pad=1.0,
    image_path_png=None,
    image_path_svg=None,
    image_filename=None,
    bbox_inches=None,
    single_var_image_filename=None,
    y_axis_label="Density",  # Parameter to control y-axis label
    plot_type="both",  # To control plot type ('hist', 'kde', or 'both')
    log_scale_vars=None,  # To specify which variables to apply log scale
    bins="auto",  # Default to 'auto' as per sns
    binwidth=None,  # Parameter to control the width of bins
    label_fontsize=10,  # Fontsize control for labels
    tick_fontsize=10,  # Fontsize control for tick labels
    text_wrap=50,
    disable_sci_notation=False,  # Toggle for scientific notation
    stat="density",  # Control the aggregate statistic for histograms
    xlim=None,
    ylim=None,
    plot_mean=False,
    plot_median=False,
    std_dev_levels=None,  # Parameter to control how many stdev to plot
    std_color="#808080",
    label_names=None,
    show_legend=True,  # New parameter to toggle the legend
    custom_xlabels=None,  # New parameter to customize x-axis labels
    custom_titles=None,  # New parameter to customize plot titles
    **kwargs,  # To capture additional keyword arguments
):
    """
    Generate KDE and/or histogram distribution plots for columns in a DataFrame.

    This function provides a flexible way to visualize the distribution of
    data for specified columns in a DataFrame. It supports both kernel density
    estimation (KDE) and histograms, with options to customize various aspects
    of the plots, including colors, labels, binning, scaling, and statistical
    overlays.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.

    vars_of_interest : list of str, optional
        List of column names for which to generate distribution plots. If
        'all', plots will be generated for all numeric columns.

    figsize : tuple of int, optional (default=(5, 5))
        Size of each individual plot. This parameter is used when there is only
        one variable being plotted or when generating separate plots for each
        variable of interest.

    subplot_figsize : tuple of int, optional
        Size of the overall grid of subplots when there are multiple variables
        being plotted in a single subplot grid. This parameter is ignored when
        only one variable is plotted or when using `single_var_image_filename`.

    hist_color : str, optional (default='#0000FF')
        Color of the histogram bars.

    kde_color : str, optional (default='#FF0000')
        Color of the KDE plot.

    mean_color : str, optional (default='#000000')
        Color of the mean line if `plot_mean` is True.

    median_color : str, optional (default='#000000')
        Color of the median line if `plot_median` is True.

    hist_edgecolor : str, optional (default='#000000')
        Color of the histogram bar edges.

    hue : str, optional
        Column name to group data by, adding different colors for each group.

    fill : bool, optional (default=True)
        Whether to fill the histogram bars with color.

    fill_alpha : float, optional (default=1)
        Alpha transparency for the fill color of the histogram bars, where
        0 is fully transparent and 1 is fully opaque.

    n_rows : int, optional
        Number of rows in the subplot grid. If not provided, it will be
        calculated automatically.

    n_cols : int, optional
        Number of columns in the subplot grid. If not provided, it will be
        calculated automatically.

    w_pad : float, optional (default=1.0)
        Width padding between subplots.

    h_pad : float, optional (default=1.0)
        Height padding between subplots.

    image_path_png : str, optional
        Directory path to save the PNG image of the overall distribution plots.

    image_path_svg : str, optional
        Directory path to save the SVG image of the overall distribution plots.

    image_filename : str, optional
        Filename to use when saving the overall distribution plots.

    bbox_inches : str, optional
        Bounding box to use when saving the figure. For example, 'tight'.

    single_var_image_filename : str, optional
        Filename to use when saving the separate distribution plots. The
        variable name will be appended to this filename. When using this
        parameter, the `figsize` parameter is used to determine the size of the
        individual plots. The `subplot_figsize` parameter is ignored in this context.

    y_axis_label : str, optional (default='Density')
        The label to display on the y-axis. If set to `None`, no y-axis label
        will be displayed.

    plot_type : str, optional (default='both')
        The type of plot to generate ('hist', 'kde', or 'both').

    log_scale_vars : str or list of str, optional
        Variable name(s) to apply log scaling. Can be a single string or a
        list of strings.

    bins : int or sequence, optional (default='auto')
        Specification of histogram bins.

    binwidth : float, optional
        Width of each bin, overrides bins but can be used with binrange.

    label_fontsize : int, optional (default=10)
        Font size for axis labels, including xlabel, ylabel, and tick marks.

    tick_fontsize : int, optional (default=10)
        Font size for tick labels on the axes.

    text_wrap : int, optional (default=50)
        Maximum number of characters allowed in plot titles and axis labels
        before wrapping them onto the next line.

    disable_sci_notation : bool, optional (default=False)
        Toggle to disable scientific notation on axes.

    stat : str, optional (default='density')
        Aggregate statistic to compute in each bin (e.g., 'count', 'frequency',
        'probability', 'percent', 'density').

    xlim : tuple of (float, float), optional
        Limits for the x-axis.

    ylim : tuple of (float, float), optional
        Limits for the y-axis.

    plot_mean : bool, optional (default=False)
        Whether to plot the mean as a vertical line.

    plot_median : bool, optional (default=False)
        Whether to plot the median as a vertical line.

    std_dev_levels : list of int, optional
        Levels of standard deviation to plot around the mean.

    std_color : str or list of str, optional (default='#808080')
        Color(s) for the standard deviation lines.

    label_names : dict, optional
        Custom labels for the variables of interest. Keys should be column
        names, and values should be the corresponding labels to display.

    show_legend : bool, optional (default=True)
        Whether to show the legend on the plots.

    custom_xlabels : dict, optional
        Dictionary to customize x-axis labels. Keys are column names, and values
        are the desired labels. If a value is `None`, no x-axis label is displayed
        for the corresponding column.

    custom_titles : dict, optional
        Dictionary to customize plot titles. Keys are column names, and values
        are the desired titles. If a value is `None`, no title is displayed
        for the corresponding column.

    **kwargs : additional keyword arguments
        Additional keyword arguments passed to the Seaborn plotting function.

    Returns:
    --------
    None
        This function does not return any value but generates and optionally
        saves distribution plots for the specified columns in the DataFrame.

    Raises:
    -------
    ValueError
        If `plot_type` is not one of ['hist', 'kde', 'both'].

    ValueError
        If `stat` is not one of ['count', 'frequency', 'probability',
        'percent', 'density'].

    ValueError
        If any variable specified in `log_scale_vars` is not in the DataFrame.

    ValueError
        If `fill` is set to False but `hist_edgecolor` or `fill_alpha` is
        specified.

    ValueError
        If `bins` and `binwidth` are both set, which can affect performance.

    ValueError
        If `subplot_figsize` is provided when only one plot is being created.

    Warnings:
    ---------
    UserWarning
        If both `bins` and `binwidth` are set, a warning about performance
        impacts is raised.
    """

    # Handle the "all" option for vars_of_interest
    if vars_of_interest == "all":
        vars_of_interest = df.select_dtypes(include=np.number).columns.tolist()

    if vars_of_interest is None:
        print("Error: No variables of interest provided.")
        return

    # Set defaults for optional parameters
    if std_dev_levels is None:
        std_dev_levels = []  # Empty list if not provided

    # Dynamically calculate n_rows and n_cols if not provided
    num_vars = len(vars_of_interest)

    # If only one variable is being plotted
    if num_vars == 1:
        n_rows, n_cols = 1, 1
        if subplot_figsize is not None:
            raise ValueError(
                f"Cannot use subplot_figsize when there is only one "
                f"plot. Use figsize instead."
            )
    else:
        # Calculate columns based on square root
        if n_rows is None or n_cols is None:
            n_cols = int(np.ceil(np.sqrt(num_vars)))
            n_rows = int(np.ceil(num_vars / n_cols))

        # Adjust figsize for subplots if multiple plots
        if subplot_figsize is None:
            figsize = (figsize[0] * n_cols, figsize[1] * n_rows)
        else:
            figsize = subplot_figsize

    # Convert log_scale_vars to list if it's a single string
    if isinstance(log_scale_vars, str):
        log_scale_vars = [log_scale_vars]

    # Ensure std_dev_levels is a list if it's specified
    if isinstance(std_dev_levels, int):
        std_dev_levels = [std_dev_levels]

    # Ensure std_color is a list with enough colors
    if isinstance(std_color, str):
        std_color = [std_color] * len(std_dev_levels)
    elif isinstance(std_color, list) and len(std_color) < len(std_dev_levels):
        raise ValueError(
            f"Not enough colors specified in 'std_color'. "
            f"You have {len(std_color)} color(s) but {len(std_dev_levels)} "
            f"standard deviation level(s). "
            f"Please provide at least as many colors as standard deviation levels."
        )

    # Validate plot_type parameter
    valid_plot_types = ["hist", "kde", "both"]
    if plot_type.lower() not in valid_plot_types:
        raise ValueError(
            f"Invalid plot_type value. Expected one of {valid_plot_types}, "
            f"got '{plot_type}' instead."
        )

    # Validate stat parameter
    valid_stats = [
        "count",
        "frequency",
        "probability",
        "proportion",
        "percent",
        "density",
    ]
    if stat.lower() not in valid_stats:
        raise ValueError(
            f"Invalid stat value. Expected one of {valid_stats}, "
            f"got '{stat}' instead."
        )

    # Check if all log_scale_vars are in the DataFrame
    if log_scale_vars:
        invalid_vars = [var for var in log_scale_vars if var not in df.columns]
        if invalid_vars:
            raise ValueError(f"Invalid log_scale_vars: {invalid_vars}")

    # Check if edgecolor is being set while fill is False
    if not fill and hist_edgecolor != "#000000":
        raise ValueError("Cannot change edgecolor when fill is set to False")

    # Check if fill_alpha is being set while fill is False
    if not fill and fill_alpha != 0.6:
        raise ValueError("Cannot set fill_alpha when fill is set to False")

    # Warn if both bins and binwidth are set
    if bins != "auto" and binwidth is not None:
        warnings.warn(
            "Specifying both bins and binwidth may affect performance.",
            UserWarning,
        )

    # Create subplots grid
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    # Flatten the axes array to simplify iteration
    axes = np.atleast_1d(axes).flatten()

    def get_label(var):
        """
        Helper function to get the custom label or original column name.
        """
        return label_names[var] if label_names and var in label_names else var

    # Iterate over the provided column list and corresponding axes
    for ax, col in zip(axes[: len(vars_of_interest)], vars_of_interest):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # Determine if log scale should be applied to this variable
            log_scale = col in log_scale_vars if log_scale_vars else False

            # Filter out non-positive values if log_scale is True
            data = df[df[col] > 0] if log_scale else df

            if custom_xlabels and col in custom_xlabels:
                xlabel = custom_xlabels[col]
            else:
                # Add "(Log)" to the label if log_scale is applied
                xlabel = f"{get_label(col)} (Log)" if log_scale else get_label(col)

            # Determine custom title
            if custom_titles and col in custom_titles:
                title = custom_titles[col]
            else:
                # Modify the title to include "(Log Scaled)" if log_scale is applied
                title = f"Distribution of {get_label(col)} {'(Log Scaled)' if log_scale else ''}"

            # Calculate mean and median if needed
            mean_value = data[col].mean() if plot_mean or std_dev_levels else None
            median_value = data[col].median() if plot_median else None
            std_value = data[col].std() if std_dev_levels else None

            try:
                # Your existing plot code
                if plot_type == "hist":
                    sns.histplot(
                        data=data,
                        x=col,
                        kde=False,
                        ax=ax,
                        hue=hue,
                        color=hist_color if hue is None and fill else None,
                        edgecolor=hist_edgecolor,
                        stat=stat.lower(),
                        fill=fill,
                        alpha=fill_alpha,  # Apply for transparency
                        log_scale=log_scale,
                        bins=bins,
                        binwidth=binwidth,
                        **kwargs,
                    )
                elif plot_type == "kde":
                    sns.kdeplot(
                        data=data,
                        x=col,
                        ax=ax,
                        hue=hue,
                        color=kde_color,
                        fill=True,
                        log_scale=log_scale,
                        **kwargs,
                    )
                elif plot_type == "both":
                    sns.histplot(
                        data=data,
                        x=col,
                        kde=False,  # No need, since plot_type controls it
                        ax=ax,
                        hue=hue,
                        color=hist_color if hue is None and fill else None,
                        edgecolor=hist_edgecolor,
                        stat=stat.lower(),
                        fill=fill,
                        alpha=fill_alpha,  # Apply for transparency
                        log_scale=log_scale,
                        bins=bins,
                        binwidth=binwidth,
                        **kwargs,
                    )
                    sns.kdeplot(
                        data=data,
                        x=col,
                        ax=ax,
                        hue=hue,
                        color=kde_color if hue is None else None,
                        log_scale=log_scale,
                        label="KDE",
                        **kwargs,
                    )

                # Plot mean as a vertical dotted line if plot_mean is True
                if plot_mean and mean_value is not None:
                    ax.axvline(
                        mean_value,
                        color=mean_color,
                        linestyle="--",
                        label="Mean",
                    )

                # Plot median as a vertical dotted line if plot_median is True
                if plot_median and median_value is not None:
                    ax.axvline(
                        median_value,
                        color=median_color,
                        linestyle="--",
                        label="Median",
                    )

                # Plot standard deviation bands if std_dev_levels is specified
                if std_dev_levels and mean_value is not None and std_value is not None:
                    for level, color in zip(std_dev_levels, std_color):
                        ax.axvline(
                            mean_value + level * std_value,
                            color=color,
                            linestyle="--",
                            label=f"±{level} Std Dev",
                        )
                        ax.axvline(
                            mean_value - level * std_value,
                            color=color,
                            linestyle="--",
                        )

                # After plotting logic, before showing the legend
                handles, labels = ax.get_legend_handles_labels()

                if show_legend and len(handles) > 0:
                    ax.legend(loc="best")
                else:
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()

            except Exception as e:
                # Handle different Python versions or issues w/ legends & labels
                if "No artists with labels found to put in legend." in str(e):
                    print(f"Warning encountered while plotting '{col}': {str(e)}")
                    handles, labels = ax.get_legend_handles_labels()
                    if show_legend and len(handles) > 0 and len(labels) > 0:
                        ax.legend(loc="best")
                    else:
                        if ax.get_legend() is not None:
                            ax.get_legend().remove()

            finally:
                pass  # Ensures the try block is valid

            ax.set_xlabel(
                "\n".join(textwrap.wrap(xlabel, width=text_wrap)) if xlabel else None,
                fontsize=label_fontsize,
            )

            ax.set_ylabel(
                (
                    "\n".join(textwrap.wrap(y_axis_label.capitalize(), width=text_wrap))
                    if y_axis_label
                    else None
                ),
                fontsize=label_fontsize,
            )

            # Apply the title
            ax.set_title(
                "\n".join(textwrap.wrap(title, width=text_wrap)) if title else None,
                fontsize=label_fontsize,
            )

            ax.tick_params(
                axis="both", labelsize=tick_fontsize
            )  # Control tick fontsize separately

            # Set axis limits if specified
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            # Disable scientific notation if requested
            if disable_sci_notation:
                ax.xaxis.set_major_formatter(
                    mticker.ScalarFormatter(useMathText=False),
                )
                ax.yaxis.set_major_formatter(
                    mticker.ScalarFormatter(useMathText=False),
                )

    # Hide any remaining axes
    for ax in axes[len(vars_of_interest) :]:
        ax.axis("off")

    # Adjust layout with specified padding
    plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # Save files if paths are provided
    if image_path_png and image_filename:
        plt.savefig(
            os.path.join(image_path_png, f"{image_filename}.png"),
            bbox_inches=bbox_inches,
        )
    if image_path_svg and image_filename:
        plt.savefig(
            os.path.join(image_path_svg, f"{image_filename}.svg"),
            bbox_inches=bbox_inches,
        )
    plt.show()

    # Generate separate plots for each variable of interest if provided
    if vars_of_interest:
        for var in vars_of_interest:
            fig, ax = plt.subplots(figsize=figsize)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                title = f"Distribution of {var}"

                # Determine if log scale should be applied to this variable
                log_scale = var in log_scale_vars if log_scale_vars else False

                # Filter out non-positive values if log_scale is True
                data = df[df[var] > 0] if log_scale else df

                try:
                    if plot_type == "hist":
                        sns.histplot(
                            data=data,
                            x=var,
                            kde=False,
                            ax=ax,
                            hue=hue,
                            color=hist_color if hue is None and fill else None,
                            edgecolor=hist_edgecolor,
                            stat=stat.lower(),
                            fill=fill,
                            alpha=fill_alpha,  # Apply for transparency
                            log_scale=log_scale,
                            bins=bins,
                            binwidth=binwidth,
                            **kwargs,
                        )
                    elif plot_type == "kde":
                        sns.kdeplot(
                            data=data,
                            x=var,
                            ax=ax,
                            hue=hue,
                            color=kde_color,
                            fill=True,
                            log_scale=log_scale,
                            **kwargs,
                        )
                    elif plot_type == "both":
                        sns.histplot(
                            data=data,
                            x=var,
                            kde=False,  # No need, since plot_type controls this
                            ax=ax,
                            hue=hue,
                            color=hist_color if hue is None and fill else None,
                            edgecolor=hist_edgecolor,
                            stat=stat.lower(),
                            fill=fill,
                            alpha=fill_alpha,  # Apply for transparency
                            log_scale=log_scale,
                            bins=bins,
                            binwidth=binwidth,
                            **kwargs,
                        )
                        sns.kdeplot(
                            data=data,
                            x=var,
                            ax=ax,
                            hue=hue,
                            color=kde_color if hue is None else None,
                            log_scale=log_scale,
                            label="KDE",
                            **kwargs,
                        )

                    # Plot mean as a vertical dotted line if plot_mean is True
                    if plot_mean:
                        mean_value = data[var].mean()
                        ax.axvline(
                            mean_value,
                            color=mean_color,
                            linestyle="--",
                            label="Mean",
                        )

                    # Plot median as vertical dotted line if plot_median is True
                    if plot_median:
                        median_value = data[var].median()
                        ax.axvline(
                            median_value,
                            color=median_color,
                            linestyle="--",
                            label="Median",
                        )

                    # Plot std. deviation bands if std_dev_levels is specified
                    if std_dev_levels:
                        std_value = data[var].std()
                        for level, color in zip(std_dev_levels, std_color):
                            ax.axvline(
                                mean_value + level * std_value,
                                color=color,
                                linestyle="--",
                            )
                            ax.axvline(
                                mean_value - level * std_value,
                                color=color,
                                linestyle="--",
                                label=f"±{level} Std Dev",
                            )

                    # After plotting logic, before showing the legend
                    handles, labels = ax.get_legend_handles_labels()

                    if show_legend and len(handles) > 0 and len(labels) > 0:
                        ax.legend(loc="best")
                    else:
                        if ax.get_legend() is not None:
                            ax.get_legend().remove()

                except Exception as e:
                    # Handle different Python versions or issues w/ legends & labels
                    if "No artists with labels found to put in legend." in str(e):
                        print(f"Warning encountered while plotting '{col}': {str(e)}")
                        handles, labels = ax.get_legend_handles_labels()
                        if show_legend and len(handles) > 0 and len(labels) > 0:
                            ax.legend(loc="best")
                        else:
                            if ax.get_legend() is not None:
                                ax.get_legend().remove()

                finally:
                    pass  # Ensures the try block is valid

                ax.set_xlabel(
                    (
                        "\n".join(textwrap.wrap(xlabel, width=text_wrap))
                        if xlabel
                        else None
                    ),
                    fontsize=label_fontsize,
                )

                ax.set_ylabel(
                    (
                        "\n".join(
                            textwrap.wrap(y_axis_label.capitalize(), width=text_wrap)
                        )
                        if y_axis_label
                        else None
                    ),
                    fontsize=label_fontsize,
                )
                ax.set_title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)) if title else None,
                    fontsize=label_fontsize,
                )
                ax.tick_params(
                    axis="both", labelsize=tick_fontsize
                )  # Control tick fontsize separately

                # Set axis limits if specified~
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

                # Disable scientific notation if requested
                if disable_sci_notation:
                    ax.xaxis.set_major_formatter(
                        mticker.ScalarFormatter(useMathText=False)
                    )
                    ax.yaxis.set_major_formatter(
                        mticker.ScalarFormatter(useMathText=False)
                    )

            plt.tight_layout()

            # Save files for the variable of interest if paths are provided
            if image_path_png and single_var_image_filename:
                plt.savefig(
                    os.path.join(
                        image_path_png,
                        f"{single_var_image_filename}_{var}.png",
                    ),
                    bbox_inches=bbox_inches,
                )
            if image_path_svg and single_var_image_filename:
                plt.savefig(
                    os.path.join(
                        image_path_svg,
                        f"{single_var_image_filename}_{var}.svg",
                    ),
                    bbox_inches=bbox_inches,
                )
            plt.close(
                fig
            )  # Close figure after saving to avoid displaying it multiple times


################################################################################
###################### Stacked Bar Plots W/ Crosstab Options ###################
################################################################################


def stacked_crosstab_plot(
    df,
    col,
    func_col,
    legend_labels_list,
    title,
    kind="bar",
    width=0.9,
    rot=0,
    custom_order=None,
    image_path_png=None,
    image_path_svg=None,
    save_formats=None,
    color=None,
    output="both",
    return_dict=False,
    x=None,
    y=None,
    p=None,
    file_prefix=None,
    logscale=False,
    plot_type="both",
    show_legend=True,
    label_fontsize=12,
    tick_fontsize=10,
    text_wrap=50,
    remove_stacks=False,
    xlim=None,
    ylim=None,
):
    """
    Generates stacked or regular bar plots and crosstabs for specified columns.

    This function allows users to create stacked bar plots (or regular bar plots
    if stacks are removed) and corresponding crosstabs for specific columns
    in a DataFrame. It provides options to customize the appearance, including
    font sizes for axis labels, tick labels, and title text wrapping, and to
    choose between regular or normalized plots.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.

    col : str
        The name of the column in the DataFrame to be analyzed.

    func_col : list of str
        List of columns in the DataFrame that will be used to generate the
        crosstabs and stack the bars in the plot.

    legend_labels_list : list of list of str
        List of legend labels corresponding to each column in `func_col`.

    title : list of str
        List of titles for each plot generated.

    kind : str, optional (default='bar')
        The kind of plot to generate ('bar' or 'barh' for horizontal bars).

    width : float, optional (default=0.9)
        The width of the bars in the bar plot.

    rot : int, optional (default=0)
        The rotation angle of the x-axis labels.

    custom_order : list, optional
        Specifies a custom order for the categories in `col`.

    image_path_png : str, optional
        Directory path where generated PNG plot images will be saved.

    image_path_svg : str, optional
        Directory path where generated SVG plot images will be saved.

    save_formats : list of str, optional (default=None)
        List of file formats to save the plot images in. Valid formats are
        'png' and 'svg'. If not provided, defaults to an empty list and no
        images will be saved.

    color : list of str, optional
        List of colors to use for the plots. If not provided, a default
        color scheme is used.

    output : str, optional (default='both')
        Specify the output type: "plots_only", "crosstabs_only", or "both".

    return_dict : bool, optional (default=False)
        Specify whether to return the crosstabs dictionary.

    x : int, optional
        The width of the figure.

    y : int, optional
        The height of the figure.

    p : int, optional
        The padding between the subplots.

    file_prefix : str, optional
        Prefix for the filename when output includes plots.

    logscale : bool, optional (default=False)
        Apply log scale to the y-axis.

    plot_type : str, optional (default='both')
        Specify the type of plot to generate: "both", "regular", or "normalized".

    show_legend : bool, optional (default=True)
        Specify whether to show the legend.

    label_fontsize : int, optional (default=12)
        Font size for axis labels.

    tick_fontsize : int, optional (default=10)
        Font size for tick labels on the axes.

    text_wrap : int, optional (default=50)
        The maximum width of the title text before wrapping.

    remove_stacks : bool, optional (default=False)
        If True, removes stacks and creates a regular bar plot using only
        the `col` parameter. Only works when `plot_type` is set to 'regular'.

    xlim : tuple, optional
        Tuple specifying the limits of the x-axis.

    ylim : tuple, optional
        Tuple specifying the limits of the y-axis.

    Returns:
    --------
    crosstabs_dict : dict
        Dictionary of crosstabs DataFrames if `return_dict` is True.

    None
        If `return_dict` is False.

    Raises:
    -------
    ValueError
        If `remove_stacks` is used when `plot_type` is not set to "regular".

    ValueError
        If `output` is not one of ["both", "plots_only", "crosstabs_only"].

    ValueError
        If `plot_type` is not one of ["both", "regular", "normalized"].

    ValueError
        If the lengths of `title`, `func_col`, and `legend_labels_list` are not
        equal.

    KeyError
        If any columns in `col` or `func_col` are missing in the DataFrame.

    ValueError
        If an invalid save format is specified without providing the
        corresponding image path.
    """

    # Check if remove_stacks is used correctly
    if remove_stacks and plot_type != "regular":
        raise ValueError(
            "`remove_stacks` can only be used when `plot_type` is set to 'regular'."
        )

    # Check if the output parameter is valid
    valid_outputs = ["both", "plots_only", "crosstabs_only"]
    if output not in valid_outputs:
        raise ValueError(
            f"Invalid output type: {output}. Valid options are {valid_outputs}"
        )

    # Check if the plot_type parameter is valid
    valid_plot_types = ["both", "regular", "normalized"]
    if plot_type not in valid_plot_types:
        raise ValueError(
            f"Invalid plot type: {plot_type}. Valid options are {valid_plot_types}"
        )

    # Ensure save_formats is a list even if None, string, or tuple is passed
    save_formats = (
        save_formats or []
    )  # Modified line: Ensures save_formats is an empty list if None
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    elif isinstance(save_formats, tuple):
        save_formats = list(save_formats)

    # Initialize the dictionary to store crosstabs
    crosstabs_dict = {}
    # Default color settings
    if color is None:
        color = ["#00BFC4", "#F8766D"]  # Default colors
    elif isinstance(color, str):
        color = [color]  # Ensure a single color is passed as a list

    # Check if all required columns are present in the DataFrame
    missing_cols = [
        col_name for col_name in [col] + func_col if col_name not in df.columns
    ]
    if missing_cols:
        raise KeyError(f"Columns missing in DataFrame: {missing_cols}")

    if not (len(title) == len(func_col) == len(legend_labels_list)):
        raise ValueError(
            f"Length mismatch: Ensure that the lengths of `title`, `func_col`, "
            f"and `legend_labels_list` are equal. Current lengths are: "
            f"title={len(title)}, func_col={len(func_col)}, "
            f"legend_labels_list={len(legend_labels_list)}. "
            "Check for missing items or commas."
        )

    # Work on a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Setting custom order if provided
    if custom_order:
        df_copy[col] = pd.Categorical(
            df_copy[col], categories=custom_order, ordered=True
        )
        df_copy.sort_values(by=col, inplace=True)

    # Generate plots if output is "both" or "plots_only"
    if output in ["both", "plots_only"]:
        if file_prefix is None:
            file_prefix = f"{col}_{'_'.join(func_col)}"

        # Set default values for x, y, and p if not provided
        if x is None:
            x = 12
        if y is None:
            y = 8
        if p is None:
            p = 10

        # Determine the number of subplots based on the plot_type parameter
        if plot_type == "both":
            nrows = 2
        else:
            nrows = 1

        # Loop through each condition and create the plots
        for truth, legend, tit in zip(func_col, legend_labels_list, title):
            image_path = {}

            if image_path_png:
                func_col_filename_png = os.path.join(
                    image_path_png, f"{file_prefix}_{truth}.png"
                )
                image_path["png"] = func_col_filename_png

            if image_path_svg:
                func_col_filename_svg = os.path.join(
                    image_path_svg, f"{file_prefix}_{truth}.svg"
                )
                image_path["svg"] = func_col_filename_svg

            # Verify the DataFrame state before creating plots
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(x, y))
            fig.tight_layout(w_pad=5, pad=p, h_pad=5)

            if remove_stacks:
                # Create a regular bar plot using only the `col` parameter
                counts = df_copy[col].value_counts()
                title1 = f"Distribution of {col.replace('_', ' ').title()}"
                xlabel1 = f"{col.replace('_', ' ')}"
                ylabel1 = "Count"
                counts.plot(
                    kind=kind,
                    ax=axes[0] if plot_type == "both" else axes,
                    color=color[0],
                    width=width,
                    rot=rot,
                    fontsize=12,
                    logy=logscale,  # Apply log scale if logscale is True
                )
                ax0 = axes[0] if plot_type == "both" else axes

                if kind == "barh":
                    ax0.set_xlabel(ylabel1, fontsize=label_fontsize)
                    ax0.set_ylabel(xlabel1, fontsize=label_fontsize)
                else:
                    ax0.set_xlabel(xlabel1, fontsize=label_fontsize)
                    ax0.set_ylabel(ylabel1, fontsize=label_fontsize)

                # Set axis limits if specified
                if xlim:
                    ax0.set_xlim(xlim)
                if ylim:
                    ax0.set_ylim(ylim)

                ax0.set_title(
                    "\n".join(textwrap.wrap(title1, width=text_wrap)),
                    fontsize=label_fontsize,  # Ensure label_fontsize is applied
                )
                ax0.tick_params(axis="both", labelsize=tick_fontsize)

                if show_legend:
                    ax0.legend([col], fontsize=12)
                else:
                    ax0.legend().remove()

            else:
                # Define crosstabdest to avoid UnboundLocalError
                crosstabdest = pd.crosstab(df_copy[col], df_copy[truth])
                try:
                    crosstabdest.columns = legend  # Rename columns
                except ValueError:
                    raise ValueError(
                        f"Length mismatch: Crosstab columns "
                        f"({len(crosstabdest.columns)}) and legend "
                        f"({len(legend)}). Check the length of your "
                        f"`legend_labels_list`, `func_col`, and `title` to ensure "
                        f"you are not missing an item, comma, or have an extra "
                        f"item."
                    )

                if plot_type in ["both", "regular"]:
                    # Plot the first graph (absolute counts)
                    title1 = f"Prevalence of {tit} by {col.replace('_', ' ').title()}"
                    xlabel1 = f"{col.replace('_', ' ').title()}"
                    ylabel1 = "Count"
                    crosstabdest.plot(
                        kind=kind,
                        stacked=True,
                        ax=axes[0] if plot_type == "both" else axes,
                        color=color,
                        width=width,
                        rot=rot,
                        fontsize=label_fontsize,  # Apply label_fontsize here
                    )

                    # Explicitly set the title with the desired font size
                    ax0 = axes[0] if plot_type == "both" else axes
                    ax0.set_title(
                        "\n".join(textwrap.wrap(title1, width=text_wrap)),
                        # Ensure the title font size is consistent
                        fontsize=label_fontsize,
                    )

                    if kind == "barh":
                        ax0.set_xlabel(ylabel1, fontsize=label_fontsize)
                        ax0.set_ylabel(xlabel1, fontsize=label_fontsize)
                    else:
                        ax0.set_xlabel(xlabel1, fontsize=label_fontsize)
                        ax0.set_ylabel(ylabel1, fontsize=label_fontsize)

                    # Set axis limits if specified
                    if xlim:
                        ax0.set_xlim(xlim)
                    if ylim:
                        ax0.set_ylim(ylim)

                    # Set tick fontsize
                    ax0.tick_params(axis="both", labelsize=tick_fontsize)

                    # Set legend font size to match label_fontsize
                    if show_legend:
                        ax0.legend(legend, fontsize=label_fontsize)
                    else:
                        ax0.legend().remove()

                if plot_type in ["both", "normalized"]:
                    # Plotting the second, normalized stacked bar graph
                    title2 = (
                        f"Prevalence of {tit} by {col.replace('_', ' ').title()} "
                        f"(Normalized)"
                    )
                    xlabel2 = f"{col.replace('_', ' ').title()}"
                    ylabel2 = "Percentage"
                    crosstabdestnorm = crosstabdest.div(
                        crosstabdest.sum(1),
                        axis=0,
                    )
                    crosstabdestnorm.plot(
                        kind=kind,
                        stacked=True,
                        ylabel="Percentage",
                        ax=axes[1] if plot_type == "both" else axes,
                        color=color,
                        width=width,
                        rot=rot,
                        # This controls axis labels and ticks
                        fontsize=label_fontsize,
                        logy=logscale,
                    )

                    # Explicitly set the title with the desired font size
                    ax1 = axes[1] if plot_type == "both" else axes
                    ax1.set_title(
                        "\n".join(textwrap.wrap(title2, width=text_wrap)),
                        # This should now control the title font size
                        fontsize=label_fontsize,
                    )

                    if kind == "barh":
                        ax1.set_xlabel(ylabel2, fontsize=label_fontsize)
                        ax1.set_ylabel(xlabel2, fontsize=label_fontsize)
                    else:
                        ax1.set_xlabel(xlabel2, fontsize=label_fontsize)
                        ax1.set_ylabel(ylabel2, fontsize=label_fontsize)

                    # Set axis limits if specified
                    if xlim:
                        ax1.set_xlim(xlim)
                    if ylim:
                        ax1.set_ylim(ylim)

                    # Set tick fontsize
                    ax1.tick_params(axis="both", labelsize=tick_fontsize)

                    # Set legend font size to match label_fontsize
                    if show_legend:
                        ax1.legend(legend, fontsize=label_fontsize)
                    else:
                        ax1.legend().remove()

            fig.align_ylabels()

            # Ensure save_formats is a list even if a string or tuple is passed
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            elif isinstance(save_formats, tuple):
                save_formats = list(save_formats)

            # Check for invalid save formats
            valid_formats = []
            if image_path_png:
                valid_formats.append("png")
            if image_path_svg:
                valid_formats.append("svg")

            # Throw an error if an invalid format is specified
            for save_format in save_formats:
                if save_format not in valid_formats:
                    missing_path = f"image_path_{save_format}"
                    raise ValueError(
                        f"Invalid save format '{save_format}'. To save in this "
                        f"format, you must first pass input for '{missing_path}'. "
                        f"Valid options are: {valid_formats}"
                    )

            if save_formats and isinstance(image_path, dict):
                for save_format in save_formats:
                    if save_format in image_path:
                        full_path = image_path[save_format]
                        plt.savefig(full_path, bbox_inches="tight")
                        print(f"Plot saved as {full_path}")

            plt.show()

    # Always generate crosstabs if return_dict=True
    if return_dict:
        legend_counter = 0
        # First run of the crosstab, accounting for totals only
        for col_results in func_col:
            crosstab_df = pd.crosstab(
                df_copy[col],
                df_copy[col_results],
                margins=True,
                margins_name="Total",
            )
            # Rename columns
            crosstab_df.rename(
                columns={
                    **{
                        col: legend_labels_list[legend_counter][i]
                        for i, col in enumerate(crosstab_df.columns)
                        if col != "Total"
                    },
                    "Total": "Total",
                },
                inplace=True,
            )
            # Re-do the crosstab, this time, accounting for normalized data
            crosstab_df_norm = pd.crosstab(
                df_copy[col],
                df_copy[col_results],
                normalize="index",
                margins=True,
                margins_name="Total",
            )
            crosstab_df_norm = crosstab_df_norm.mul(100).round(2)
            crosstab_df_norm.rename(
                columns={
                    **{
                        col: f"{legend_labels_list[legend_counter][i]}_%"
                        for i, col in enumerate(crosstab_df_norm.columns)
                        if col != "Total"
                    },
                    "Total": "Total_%",
                },
                inplace=True,
            )
            crosstab_df = pd.concat([crosstab_df, crosstab_df_norm], axis=1)
            legend_counter += 1
            crosstabs_dict[col_results] = crosstab_df

    # Display crosstabs only if required by output
    if output in ["both", "crosstabs_only"]:
        for col_results, crosstab_df in crosstabs_dict.items():
            # Display results
            print()
            print("Crosstab for " + col_results)
            print()
            print(crosstab_df)
            print()
            # Store the crosstab in the dictionary
            # Use col_results as the key

    # Return the crosstabs_dict only if return_dict is True
    if return_dict:
        return crosstabs_dict


################################################################################
############################ Box and Violin Plots ##############################
################################################################################


def box_violin_plot(
    df,
    metrics_list,
    metrics_comp,
    n_rows=None,  # Allow users to define the number of rows
    n_cols=None,  # Allow users to define the number of columns
    image_path_png=None,  # Make image paths optional
    image_path_svg=None,  # Make image paths optional
    save_plots=False,  # Parameter to control saving plots
    show_legend=True,  # Parameter to toggle legend
    plot_type="boxplot",  # Parameter to specify plot type
    xlabel_rot=0,  # Parameter to rotate x-axis labels
    show_plot="both",  # Parameter to control plot display
    rotate_plot=False,  # Parameter to rotate (pivot) plots
    individual_figsize=(6, 4),
    subplot_figsize=None,  # Parameter to specify figure size for subplots
    label_fontsize=12,  # Parameter to control axis label fontsize
    tick_fontsize=10,  # Parameter to control tick label fontsize
    text_wrap=50,  # Add text_wrap parameter
    xlim=None,  # New parameter for setting x-axis limits
    ylim=None,  # New parameter for setting y-axis limits
    label_names=None,
    **kwargs,  # To allow passing additional parameters to Seaborn
):
    """
    Create and save individual or subplot-based boxplots or violin plots for
    specified metrics and comparisons.

    This function generates individual plots, subplots, or both for the
    specified metrics and comparison categories in a DataFrame. It provides
    extensive customization options for plot appearance, saving options, and
    display preferences, including support for axis limits, label customization,
    and rotated layouts.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.

    metrics_list : list of str
        List of column names representing the metrics to be plotted.

    metrics_comp : list of str
        List of column names representing the comparison categories.

    n_rows : int, optional
        Number of rows in the subplot grid. Calculated automatically if not
        provided.

    n_cols : int, optional
        Number of columns in the subplot grid. Calculated automatically if not
        provided.

    image_path_png : str, optional
        Directory path to save plots in PNG format. If not specified, plots will
        not be saved as PNG.

    image_path_svg : str, optional
        Directory path to save plots in SVG format. If not specified, plots will
        not be saved as SVG.

    save_plots : bool, optional
        If True, saves the plots specified by the `show_plot` parameter
        ("individual", "subplots", or "both"). Defaults to False.

    show_legend : bool, optional (default=True)
        Whether to display the legend on the plots.

    plot_type : str, optional (default='boxplot')
        Type of plot to generate. Options are "boxplot" or "violinplot".

    xlabel_rot : int, optional (default=0)
        Rotation angle for x-axis labels.

    show_plot : str, optional (default='both')
        Specify the type of plots to display. Options are "individual", "subplots",
        or "both".

    rotate_plot : bool, optional (default=False)
        If True, rotates the plots by swapping the x and y axes.

    individual_figsize : tuple of int, optional (default=(6, 4))
        Dimensions (width, height) for individual plots.

    subplot_figsize : tuple of int, optional
        Dimensions (width, height) of the subplots. Defaults to a size
        proportional to the number of rows and columns.

    label_fontsize : int, optional (default=12)
        Font size for axis labels.

    tick_fontsize : int, optional (default=10)
        Font size for axis tick labels.

    text_wrap : int, optional (default=50)
        Maximum number of characters allowed in plot titles and axis labels
        before wrapping them onto the next line.

    xlim : tuple of float, optional
        Limits for the x-axis as (min, max).

    ylim : tuple of float, optional
        Limits for the y-axis as (min, max).

    label_names : dict, optional
        Dictionary to map original column names to custom labels for display
        purposes.

    **kwargs : additional keyword arguments
        Additional parameters passed to the Seaborn plotting function.

    Returns:
    --------
    None
        This function does not return a value. It generates and optionally saves
        or displays the specified plots.

    Raises:
    -------
    ValueError
        - If `show_plot` is not one of "individual", "subplots", or "both".
        - If `save_plots` is True but `image_path_png` or `image_path_svg` is
          not specified.
        - If `rotate_plot` is not a boolean value.
        - If `individual_figsize` or `subplot_figsize` is not a tuple or list of
          two numbers.

    Notes:
    ------
    - Automatically calculates `n_rows` and `n_cols` if not provided based on
      the number of plots.
    - Supports rotating the plot layout using the `rotate_plot` parameter.
    - Saves plots to the specified paths if `save_plots` is True.
    - Handles axis label customization using `label_names` and supports text
      wrapping for plot titles.
    """

    # Check for valid show_plot values
    if show_plot not in ["individual", "subplots", "both"]:
        raise ValueError(
            "Invalid `show_plot` value selected. Choose from 'individual', "
            "'subplots', or 'both'."
        )

    # Check for valid save_plots value
    if not isinstance(save_plots, bool):
        raise ValueError("`save_plots` must be a boolean value (True or False).")

    # Check if save_plots is set without image paths
    if save_plots and not (image_path_png or image_path_svg):
        raise ValueError("To save plots, specify `image_path_png` or `image_path_svg`.")

    # Check for valid rotate_plot values
    if not isinstance(rotate_plot, bool):
        raise ValueError(
            "Invalid `rotate_plot` value selected. Choose from 'True' or 'False'."
        )

    # Check for valid individual_figsize values
    if not (
        isinstance(individual_figsize, (tuple, list))
        and len(individual_figsize) == 2
        and all(isinstance(x, (int, float)) for x in individual_figsize)
    ):
        raise ValueError(
            "Invalid `individual_figsize` value. It should be a tuple or list "
            "of two numbers (width, height)."
        )

    # Check for valid subplot_figsize values if specified
    if subplot_figsize is not None and not (
        isinstance(subplot_figsize, (tuple, list))
        and len(subplot_figsize) == 2
        and all(isinstance(x, (int, float)) for x in subplot_figsize)
    ):
        raise ValueError(
            "Invalid `subplot_figsize` value. It should be a tuple or list of two "
            "numbers (width, height)."
        )

    total_plots = len(metrics_list) * len(metrics_comp)

    # Ensure at least one plot is possible
    if total_plots == 0:
        raise ValueError(
            "No valid plots to generate. Ensure `metrics_list` and "
            "`metrics_comp` are not empty."
        )

    if n_cols is None:
        # Ensure at least 1 column
        n_cols = max(1, int(np.ceil(np.sqrt(total_plots))))

    if n_rows is None:
        # Ensure at least 1 row
        n_rows = max(1, int(np.ceil(total_plots / n_cols)))

    # Set default subplot figure size if not specified
    if subplot_figsize is None:
        subplot_figsize = (5 * n_cols, 5 * n_rows)

    # Determine saving options based on `show_plot`
    save_individual = save_plots and show_plot in ["individual", "both"]
    save_subplots = save_plots and show_plot in ["subplots", "both"]

    def get_palette(n_colors):
        """
        Returns a 'tab10' color palette with the specified number of colors.
        """
        return sns.color_palette("tab10", n_colors=n_colors)

    def get_label(var):
        """
        Helper function to get the custom label or original column name.
        """
        return label_names[var] if label_names and var in label_names else var

    # Map plot_type to the corresponding seaborn function
    plot_function = getattr(sns, plot_type)

    # Save and/or show individual plots if required
    if save_individual or show_plot in ["individual", "both"]:
        for met_comp in metrics_comp:
            unique_vals = df[met_comp].value_counts().count()
            palette = get_palette(unique_vals)
            for met_list in metrics_list:
                plt.figure(figsize=individual_figsize)  # Adjust size as needed
                # Use original column names for plotting
                ax = plot_function(
                    x=met_list if rotate_plot else met_comp,
                    y=met_comp if rotate_plot else met_list,
                    data=df,
                    hue=met_comp,
                    palette=palette,
                    dodge=False,
                    **kwargs,
                )

                # Use custom labels only for display purposes
                title = (
                    f"Distribution of {get_label(met_list)} by {get_label(met_comp)}"
                )
                plt.title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)),
                    fontsize=label_fontsize,
                )
                plt.xlabel(
                    "\n".join(
                        textwrap.wrap(
                            get_label(met_list) if rotate_plot else get_label(met_comp),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                plt.ylabel(
                    "\n".join(
                        textwrap.wrap(
                            get_label(met_comp) if rotate_plot else get_label(met_list),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)

                # Set x and y limits if specified
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

                # Toggle legend
                if not show_legend and ax.legend_:
                    ax.legend_.remove()

                if save_individual:
                    safe_met_list = (
                        met_list.replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("/", "_per_")
                    )
                    if image_path_png:
                        filename_png = (
                            f"{safe_met_list}_by_{met_comp}_" f"{plot_type}.png"
                        )
                        plt.savefig(
                            os.path.join(image_path_png, filename_png),
                            bbox_inches="tight",
                        )
                    if image_path_svg:
                        filename_svg = (
                            f"{safe_met_list}_by_{met_comp}_" f"{plot_type}.svg"
                        )
                        plt.savefig(
                            os.path.join(image_path_svg, filename_svg),
                            bbox_inches="tight",
                        )

                if show_plot in ["individual", "both", "subplots"]:
                    plt.show()  # Display the plot

    # Save and/or show the entire subplot grid if required
    if save_subplots or show_plot in ["subplots", "both"]:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=subplot_figsize)
        # Handle the case when axs is a single Axes object
        if n_rows * n_cols == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        for i, ax in enumerate(axs):
            if i < len(metrics_list) * len(metrics_comp):
                met_comp = metrics_comp[i // len(metrics_list)]
                met_list = metrics_list[i % len(metrics_list)]
                unique_vals = df[met_comp].value_counts().count()
                palette = get_palette(unique_vals)
                plot_function(
                    x=met_list if rotate_plot else met_comp,
                    y=met_comp if rotate_plot else met_list,
                    data=df,
                    hue=met_comp,
                    ax=ax,
                    palette=palette,
                    dodge=False,
                    **kwargs,
                )
                title = (
                    f"Distribution of {get_label(met_list)} by {get_label(met_comp)}"
                )

                ax.set_title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)),
                    fontsize=label_fontsize,
                )
                ax.set_xlabel(
                    "\n".join(
                        textwrap.wrap(
                            get_label(met_list) if rotate_plot else get_label(met_comp),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    "\n".join(
                        textwrap.wrap(
                            get_label(met_comp) if rotate_plot else get_label(met_list),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                # Set x and y limits if specified
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

                # Toggle legend
                if not show_legend and ax.legend_:
                    ax.legend_.remove()
            else:
                ax.set_visible(False)

        plt.tight_layout()
        if save_subplots:
            if image_path_png:
                fig.savefig(
                    os.path.join(
                        image_path_png,
                        f"all_plots_comparisons_{plot_type}.png",
                    ),
                    bbox_inches="tight",
                )
            if image_path_svg:
                fig.savefig(
                    os.path.join(
                        image_path_svg,
                        f"all_plots_comparisons_{plot_type}.svg",
                    ),
                    bbox_inches="tight",
                )

        if show_plot in ["subplots", "both"]:
            plt.show()  # Display the plot


################################################################################
########################## Multi-Purpose Scatter Plots #########################
################################################################################


def scatter_fit_plot(
    df,
    x_vars=None,
    y_vars=None,
    all_vars=None,
    exclude_combinations=None,
    n_rows=None,
    n_cols=None,
    max_cols=4,
    image_path_png=None,  # Make image paths optional
    image_path_svg=None,  # Make image paths optional
    save_plots=None,  # Parameter to control saving plots
    show_legend=True,  # Parameter to toggle legend
    xlabel_rot=0,  # Parameter to rotate x-axis labels
    show_plot="subplots",  # Parameter to control plot display
    rotate_plot=False,  # Parameter to rotate (pivot) plots
    individual_figsize=(6, 4),
    subplot_figsize=None,  # Parameter to specify figure size for subplots
    label_fontsize=12,  # Parameter to control axis label fontsize
    tick_fontsize=10,  # Parameter to control tick label fontsize
    text_wrap=50,  # Parameter to control wrapping of text in title
    add_best_fit_line=False,  # Parameter to add best fit line
    scatter_color="C0",  # Parameter to control the color of scattered points
    best_fit_linecolor="red",  # Parameter to control color of best fit line
    best_fit_linestyle="-",  # Parameter to control linestyle of best fit line
    hue=None,  # Parameter to add hue to scatterplot
    hue_palette=None,  # Parameter to specify colors for each hue level
    size=None,  # Parameter to control the size of scatter points
    sizes=None,  # Parameter to define a range of sizes for scatter points
    marker="o",  # Parameter to control the marker style
    show_correlation=True,  # Parameter to toggle showing correlation in title
    xlim=None,  # Parameter to set x-axis limits
    ylim=None,  # Parameter to set y-axis limits
    label_names=None,  # New parameter for custom column renames
    **kwargs,  # Additional keyword arguments to pass to sns.scatterplot
):
    """
    Create and save scatter plots or a grid of scatter subplots for given
    x_vars and y_vars, with an optional best fit line, customizable
    point color, size, markers, and axis limits.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    x_vars : list of str or str, optional
        List of variable names to plot on the x-axis. If a string is provided,
        it will be converted into a list with one element.

    y_vars : list of str or str, optional
        List of variable names to plot on the y-axis. If a string is provided,
        it will be converted into a list with one element.

    all_vars : list of str, optional
        If provided, automatically generates scatter plots for all combinations
        of variables in this list, overriding x_vars and y_vars.

    exclude_combinations : list of tuples, optional
        List of (x_var, y_var) combinations to exclude from the plots.

    n_rows : int, optional
        Number of rows in the subplot grid. If not specified, it will be
        calculated based on the number of plots and n_cols.

    n_cols : int, optional
        Number of columns in the subplot grid. If not specified, it will be
        calculated based on the number of plots and max_cols.

    max_cols : int, optional (default=4)
        Maximum number of columns in the subplot grid.

    image_path_png : str, optional
        Directory path to save PNG images of the scatter plots.

    image_path_svg : str, optional
        Directory path to save SVG images of the scatter plots.

    save_plots : str, optional
        Controls which plots to save: "all", "individual", or "subplots".
        If None, plots will not be saved.
        - "all": Saves both individual and subplots.
        - "individual": Saves each scatter plot separately with a progress bar
          (powered by `tqdm`) to track saving progress.
        - "subplots": Saves a single subplot grid of all combinations.

    show_legend : bool, optional (default=True)
        Whether to display the legend on the plots.

    xlabel_rot : int, optional (default=0)
        Rotation angle for x-axis labels.

    show_plot : str, optional (default="subplots")
        Controls plot display: "individual", "subplots", or "both".

    rotate_plot : bool, optional (default=False)
        Whether to rotate (pivot) the plots.

    individual_figsize : tuple or list, optional (default=(6, 4))
        Width and height of the figure for individual plots.

    subplot_figsize : tuple or list, optional
        Width and height of the figure for subplots.
        If not specified, defaults to a calculated size based on the number of
        rows and columns.

    label_fontsize : int, optional (default=12)
        Font size for axis labels.

    tick_fontsize : int, optional (default=10)
        Font size for axis tick labels.

    text_wrap : int, optional (default=50)
        The maximum width of text (in characters) before wrapping. This applies
        to the plot title, x-axis labels, and y-axis labels.

    add_best_fit_line : bool, optional (default=False)
        Whether to add a best fit line to the scatter plots.

    scatter_color : str, optional (default="C0")
        Color code for the scattered points.

    best_fit_linecolor : str, optional (default="red")
        Color code for the best fit line.

    best_fit_linestyle : str, optional (default="-")
        Linestyle for the best fit line.

    hue : str, optional
        Column name for the grouping variable that will produce points with
        different colors.

    hue_palette : dict, list, or str, optional
        Specifies colors for each hue level. Can be a dictionary mapping hue
        levels to colors, a list of colors, or the name of a seaborn color
        palette.

    size : str, optional
        Column name for the grouping variable that will produce points with
        different sizes.

    sizes : dict, optional
        Dictionary mapping sizes (smallest and largest) to min and max values.

    marker : str, optional (default="o")
        Marker style used for the scatter points.

    show_correlation : bool, optional (default=True)
        Whether to display the Pearson correlation coefficient in the plot title.

    xlim : tuple or list, optional
        Limits for the x-axis as a tuple or list of (min, max).
        If not provided, the limits are determined automatically.

    ylim : tuple or list, optional
        Limits for the y-axis as a tuple or list of (min, max).
        If not provided, the limits are determined automatically.

    label_names : dict, optional
        A dictionary to rename columns for display in the plot titles and labels.

    **kwargs : dict, optional
        Additional keyword arguments to pass to sns.scatterplot.

    Returns:
    --------
    None
        This function does not return any value but generates and optionally
        saves scatter plots for the specified x_vars and y_vars.

    Raises:
    -------
    ValueError
        If `all_vars` is provided and either `x_vars` or `y_vars` is also
        provided.

    ValueError
        If neither `all_vars` nor both `x_vars` and `y_vars` are provided.

    ValueError
        If `hue_palette` is specified without `hue`.

    ValueError
        If `show_plot` is not one of ["individual", "subplots", "both", "combinations"].

    ValueError
        If `save_plots` is not one of [None, "all", "individual", "subplots"].

    ValueError
        If `save_plots` is set without specifying either `image_path_png` or
        `image_path_svg`.

    ValueError
        If `rotate_plot` is not a boolean value.

    ValueError
        If `individual_figsize` is not a tuple or list of two numbers
        (width, height).

    ValueError
        If `subplot_figsize` is provided and is not a tuple or list of two numbers
        (width, height).

    ValueError
        If `exclude_combinations` contains invalid entries (i.e., items that
        are not tuples of exactly two elements).

    ValueError
        If any column names in `exclude_combinations` do not exist in the
        DataFrame.
    """

    # Ensure x_vars and y_vars are lists
    if isinstance(x_vars, str):
        x_vars = [x_vars]
    if isinstance(y_vars, str):
        y_vars = [y_vars]

    # Check for conflicting inputs of variable assignments
    if all_vars is not None and (x_vars is not None or y_vars is not None):
        raise ValueError(
            f"Cannot pass `all_vars` and still choose `x_vars` "
            f"and/or `y_vars`. Must choose either `x_vars` and "
            f"`y_vars` as inputs or `all_vars`."
        )

    # Check if hue_palette is provided without hue
    if hue_palette is not None and hue is None:
        raise ValueError(
            f"Cannot specify `hue_palette` without specifying `hue`. "
            f"Please provide the `hue` parameter or remove `hue_palette`."
        )

    # Generate combinations of x_vars and y_vars or use all_vars
    if all_vars:
        combinations = list(itertools.combinations(all_vars, 2))
    elif x_vars is not None and y_vars is not None:
        combinations = [(x_var, y_var) for x_var in x_vars for y_var in y_vars]
    else:
        raise ValueError(
            f"Either `all_vars` or both `x_vars` and `y_vars` must be provided."
        )

    # Validate exclude_combinations
    if exclude_combinations:
        # Validate exclude_combinations without modifying the original input
        normalized_exclude_combinations = {
            tuple(sorted(pair)) for pair in exclude_combinations
        }

        # Check if all columns in exclude_combinations exist in the DataFrame
        invalid_columns = {
            col
            for pair in normalized_exclude_combinations
            for col in pair
            if col not in df.columns
        }
        if invalid_columns:
            raise ValueError(
                f"Invalid column names in `exclude_combinations`: {invalid_columns}. "
                "Please ensure all columns exist in the DataFrame."
            )

        # Use normalized_exclude_combinations to filter combinations
        combinations = [
            (x, y)
            for (x, y) in combinations
            if tuple(sorted((x, y))) not in normalized_exclude_combinations
        ]

    # Handle show_plot="combinations"
    if show_plot == "combinations":
        return combinations

    # Calculate the number of plots
    num_plots = len(combinations)

    # Ensuring that `n_cols` and `n_rows` never become zero, preventing a
    # division by zero error.
    if n_cols is None or n_cols == 0:
        n_cols = min(num_plots, max_cols) if num_plots > 0 else 1
    if n_rows is None:
        n_rows = max(1, math.ceil(num_plots / n_cols))

    # Set default subplot grid figure size if not specified
    if subplot_figsize is None:
        subplot_figsize = (5 * n_cols, 5 * n_rows)

    # Validate the show_plot input
    valid_show_plot_values = ["individual", "subplots", "both", "combinations"]
    if show_plot not in valid_show_plot_values:
        raise ValueError(f"Invalid `show_plot`. Choose from {valid_show_plot_values}.")

    # Validate the save_plots input
    if save_plots not in [None, "all", "individual", "subplots"]:
        raise ValueError(
            "Invalid `save_plots` value. Choose from 'all', "
            "'individual', 'subplots', or None."
        )

    # Check if save_plots is set without image paths
    if save_plots and not (image_path_png or image_path_svg):
        raise ValueError("To save plots, specify `image_path_png` or `image_path_svg`.")

    # Validate the rotate_plot input
    if not isinstance(rotate_plot, bool):
        raise ValueError("Invalid `rotate_plot`. Choose True or False.")

    # Validate the individual_figsize input
    if not (
        isinstance(individual_figsize, (tuple, list))
        and len(individual_figsize) == 2
        and all(isinstance(x, (int, float)) for x in individual_figsize)
    ):
        raise ValueError(
            "Invalid `individual_figsize` value. It should be a tuple or list "
            "of two numbers (width, height)."
        )

    # Validate the subplot_figsize input if specified
    if subplot_figsize is not None and not (
        isinstance(subplot_figsize, (tuple, list))
        and len(subplot_figsize) == 2
        and all(isinstance(x, (int, float)) for x in subplot_figsize)
    ):
        raise ValueError(
            "Invalid `subplot_figsize` value. It should be a tuple or list of two "
            "numbers (width, height)."
        )

    # Validation checks (already present)
    def get_label(var):
        return label_names.get(var, var) if label_names else var

    def add_best_fit(ax, x, y, linestyle, linecolor):
        m, b = np.polyfit(x, y, 1)
        ax.plot(
            x,
            m * x + b,
            color=linecolor,
            linestyle=linestyle,
            label=f"y = {m:.2f}x + {b:.2f}",
        )
        if show_legend:
            ax.legend(loc="best")
        elif ax.legend_:
            ax.legend_.remove()

    # Create grid for individual or subplots plotting
    if num_plots == 1:
        _, ax = plt.subplots(figsize=subplot_figsize)
        axes = [ax]  # Wrap single axis in a list for consistency
    else:
        _, axes = plt.subplots(n_rows, n_cols, figsize=subplot_figsize)
        axes = axes.flatten()

    # Render and show individual plots
    if show_plot in ["individual", "both"]:
        for x_var, y_var in combinations:
            plt.figure(figsize=individual_figsize)
            ax = sns.scatterplot(
                x=x_var if not rotate_plot else y_var,
                y=y_var if not rotate_plot else x_var,
                data=df,
                color=scatter_color if hue is None else None,
                hue=hue,
                palette=hue_palette,
                size=size,
                sizes=sizes,
                marker=marker,
                **kwargs,
            )

            if add_best_fit_line:
                x_data = df[x_var] if not rotate_plot else df[y_var]
                y_data = df[y_var] if not rotate_plot else df[x_var]
                add_best_fit(
                    ax,
                    x_data,
                    y_data,
                    best_fit_linestyle,
                    best_fit_linecolor,
                )

            r_value = df[x_var].corr(df[y_var])
            title = f"{get_label(x_var)} vs. {get_label(y_var)}"
            if show_correlation:
                title += f" ($r$ = {r_value:.2f})"
            plt.title(
                "\n".join(textwrap.wrap(title, width=text_wrap)),
                fontsize=label_fontsize,
            )
            plt.xlabel(
                "\n".join(
                    textwrap.wrap(
                        get_label(x_var) if not rotate_plot else get_label(y_var),
                        width=text_wrap,
                    )
                ),
                fontsize=label_fontsize,
            )
            plt.ylabel(
                "\n".join(
                    textwrap.wrap(
                        get_label(y_var) if not rotate_plot else get_label(x_var),
                        width=text_wrap,
                    )
                ),
                fontsize=label_fontsize,
            )
            ax.tick_params(axis="x", rotation=xlabel_rot)
            ax.tick_params(axis="both", labelsize=tick_fontsize)

            # Apply xlim and ylim if provided
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            plt.show()

    # Render and show subplots
    if show_plot in ["subplots", "both"]:

        for i, ax in enumerate(axes):
            if i < num_plots:
                x_var, y_var = combinations[i]
                sns.scatterplot(
                    x=x_var if not rotate_plot else y_var,
                    y=y_var if not rotate_plot else x_var,
                    data=df,
                    ax=ax,
                    color=scatter_color,
                    hue=hue,
                    palette=hue_palette,
                    size=size,
                    sizes=sizes,
                    marker=marker,
                    **kwargs,
                )

                if add_best_fit_line:
                    x_data = df[x_var] if not rotate_plot else df[y_var]
                    y_data = df[y_var] if not rotate_plot else df[x_var]
                    add_best_fit(
                        ax,
                        x_data,
                        y_data,
                        best_fit_linestyle,
                        best_fit_linecolor,
                    )

                r_value = df[x_var].corr(df[y_var])
                title = f"{get_label(x_var)} vs. {get_label(y_var)}"
                if show_correlation:
                    title += f" ($r$ = {r_value:.2f})"
                ax.set_title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)),
                    fontsize=label_fontsize,
                )
                ax.set_xlabel(
                    "\n".join(
                        textwrap.wrap(
                            get_label(x_var) if not rotate_plot else get_label(y_var),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    "\n".join(
                        textwrap.wrap(
                            get_label(y_var) if not rotate_plot else get_label(x_var),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)

                # Apply xlim and ylim if provided
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

            else:
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    # Save individual plots with progress bar
    if save_plots in ["all", "individual"]:
        with tqdm(total=len(combinations), desc="Saving scatter plot(s)") as pbar:
            for x_var, y_var in combinations:
                fig_individual, ax = plt.subplots(
                    figsize=individual_figsize
                )  # Use distinct figure for each plot
                sns.scatterplot(
                    x=x_var if not rotate_plot else y_var,
                    y=y_var if not rotate_plot else x_var,
                    data=df,
                    ax=ax,
                    color=scatter_color if hue is None else None,
                    hue=hue,
                    palette=hue_palette,
                    size=size,
                    sizes=sizes,
                    marker=marker,
                    **kwargs,
                )

                if add_best_fit_line:
                    add_best_fit(
                        ax,
                        df[x_var] if not rotate_plot else df[y_var],
                        df[y_var] if not rotate_plot else df[x_var],
                        best_fit_linestyle,
                        best_fit_linecolor,
                    )

                ax.set_title(
                    f"{get_label(x_var)} vs. {get_label(y_var)}",
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)

                safe_x_var = x_var.replace(" ", "_").replace("/", "_per_")
                safe_y_var = y_var.replace(" ", "_").replace("/", "_per_")
                if image_path_png:
                    fig_individual.savefig(
                        os.path.join(
                            image_path_png,
                            f"scatter_{safe_x_var}_vs_{safe_y_var}.png",
                        ),
                        bbox_inches="tight",
                    )
                if image_path_svg:
                    fig_individual.savefig(
                        os.path.join(
                            image_path_svg,
                            f"scatter_{safe_x_var}_vs_{safe_y_var}.svg",
                        ),
                        bbox_inches="tight",
                    )
                plt.close(fig_individual)  # Clear memory
                pbar.update(1)  # Update progress bar

    # Save subplots
    if save_plots == "subplots":
        # Render the subplots
        fig_grid, axes = plt.subplots(n_rows, n_cols, figsize=subplot_figsize)
        axes = axes.flatten()  # Flatten axes for consistent handling

        for i, ax in enumerate(axes):
            if i < num_plots:
                x_var, y_var = combinations[i]
                sns.scatterplot(
                    x=x_var if not rotate_plot else y_var,
                    y=y_var if not rotate_plot else x_var,
                    data=df,
                    ax=ax,
                    color=scatter_color,
                    hue=hue,
                    palette=hue_palette,
                    size=size,
                    sizes=sizes,
                    marker=marker,
                    **kwargs,
                )

                if add_best_fit_line:
                    x_data = df[x_var] if not rotate_plot else df[y_var]
                    y_data = df[y_var] if not rotate_plot else df[x_var]
                    add_best_fit(
                        ax,
                        x_data,
                        y_data,
                        best_fit_linestyle,
                        best_fit_linecolor,
                    )

                r_value = df[x_var].corr(df[y_var])
                title = f"{get_label(x_var)} vs. {get_label(y_var)}"
                if show_correlation:
                    title += f" ($r$ = {r_value:.2f})"
                ax.set_title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)),
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
            else:
                ax.axis("off")  # Turn off unused axes

        plt.tight_layout()

        # Save the subplots without a progress bar
        subplots_filename_png = "scatter_plots_subplots.png"
        subplots_filename_svg = "scatter_plots_subplots .svg"
        if image_path_png:
            fig_grid.savefig(
                os.path.join(image_path_png, subplots_filename_png),
                bbox_inches="tight",
            )
        if image_path_svg:
            fig_grid.savefig(
                os.path.join(image_path_svg, subplots_filename_svg),
                bbox_inches="tight",
            )

        plt.close(fig_grid)  # Clear memory


################################################################################
######################### Correlation Matrices #################################
################################################################################


def flex_corr_matrix(
    df,
    cols=None,
    annot=True,
    cmap="coolwarm",
    save_plots=False,
    image_path_png=None,
    image_path_svg=None,
    figsize=(10, 10),
    title=None,
    label_fontsize=12,
    tick_fontsize=10,
    xlabel_rot=45,
    ylabel_rot=0,
    xlabel_alignment="right",
    ylabel_alignment="center_baseline",
    text_wrap=50,
    vmin=-1,
    vmax=1,
    cbar_label="Correlation Index",
    triangular=True,  # New parameter to control triangular vs full matrix
    label_names=None,
    cbar_padding=0.8,
    cbar_width_ratio=0.05,
    show_colorbar=True,
    **kwargs,
):
    """
    Creates a correlation heatmap with extensive customization options, including
    triangular masking, alignment adjustments, title wrapping, and dynamic
    colorbar scaling. Users can save the plot in PNG and SVG formats.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    cols : list of str, optional
        List of column names to include in the correlation matrix. If None, all
        columns are included.

    annot : bool, optional (default=True)
        Whether to annotate the heatmap with correlation coefficients.

    cmap : str, optional (default='coolwarm')
        The colormap to use for the heatmap.

    save_plots : bool, optional (default=False)
        Whether to save the heatmap as an image.

    image_path_png : str, optional
        Directory path to save the heatmap as a PNG image.

    image_path_svg : str, optional
        Directory path to save the heatmap as an SVG image.

    figsize : tuple, optional (default=(10, 10))
        Width and height of the heatmap figure.

    title : str, optional
        Title of the heatmap.

    label_fontsize : int, optional (default=12)
        Font size for axis labels and title.

    tick_fontsize : int, optional (default=10)
        Font size for tick labels and colorbar labels.

    xlabel_rot : int, optional (default=45)
        Rotation angle for x-axis labels.

    ylabel_rot : int, optional (default=0)
        Rotation angle for y-axis labels.

    xlabel_alignment : str, optional (default="right")
        Horizontal alignment for x-axis labels (e.g., "center", "right").

    ylabel_alignment : str, optional (default="center_baseline")
        Vertical alignment for y-axis labels (e.g., "center", "top").

    text_wrap : int, optional (default=50)
        Maximum width of the text before wrapping. This applies to the plot
        title, x-axis labels, and y-axis labels, ensuring that long text is
        neatly displayed without overflow or truncation.

    vmin : float, optional (default=-1)
        Minimum value for the heatmap color scale.

    vmax : float, optional (default=1)
        Maximum value for the heatmap color scale.

    cbar_label : str, optional (default='Correlation Index')
        Label for the colorbar.

    triangular : bool, optional (default=True)
        Whether to show only the upper triangle of the correlation matrix.

    label_names : dict, optional
        A dictionary to map original column names to custom labels.

    cbar_padding : float, optional (default=0.8)
        Padding between the heatmap and the colorbar.

    cbar_width_ratio : float, optional (default=0.05)
        Relative width of the colorbar compared to the heatmap.

    show_colorbar : bool, optional (default=True)
        Whether to display the colorbar. If False, no colorbar will be shown.

    **kwargs : dict, optional
        Additional keyword arguments to pass to `sns.heatmap()`.

    Returns:
    --------
    None
        This function does not return any value but generates and optionally
        saves a correlation heatmap.

    Raises:
    -------
    ValueError
        If `annot`, `save_plots`, or `triangular` is not a boolean value.

    ValueError
        If `cols` is provided but is not a list of column names.

    ValueError
        If `save_plots` is True but neither `image_path_png` nor `image_path_svg`
        is specified.

    Notes:
    ------
    - If `triangular=True`, the heatmap will display only the upper triangle
    of the correlation matrix, excluding the diagonal.
    - Custom labels specified in `label_names` will replace the default column
    names in the heatmap's axes.
    - Save formats are determined by the paths provided for PNG and SVG files.
    - The colorbar width dynamically scales based on the grid square size for
    consistent proportions.
    - The `cbar_padding` and `cbar_width_ratio` parameters control the relative
    spacing and size of the colorbar, providing additional layout customization.
    - The `show_colorbar` parameter allows users to optionally exclude the
    colorbar for a cleaner plot.
    """

    # Validation: Ensure annot is a boolean
    if not isinstance(annot, bool):
        raise ValueError(
            "Invalid value for `annot`. Please enter either True or False."
        )

    # Validation: Ensure cols is a list if provided
    if cols is not None and not isinstance(cols, list):
        raise ValueError("The `cols` parameter must be a list of column names.")

    # Validation: Ensure save_plots is a boolean
    if not isinstance(save_plots, bool):
        raise ValueError("Invalid `save_plots` value. Enter True or False.")

    # Validation: Ensure triangular is a boolean
    if not isinstance(triangular, bool):
        raise ValueError(
            "Invalid `triangular` value. Please enter either True or False."
        )

    # Validate paths are specified if save_plots is True
    if save_plots and not (image_path_png or image_path_svg):
        raise ValueError(
            f"You must specify `image_path_png` or `image_path_svg` "
            f"when `save_plots` is True."
        )

    # Filter DataFrame if cols are specified
    if cols is not None:
        df = df[cols]

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Generate a mask for the upper triangle, excluding the diagonal
    mask = None
    if triangular:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Set up the matplotlib figure
    fig, ax_heatmap = plt.subplots(figsize=figsize)

    # Draw the heatmap
    heatmap = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar=False,  # Disable the default colorbar
        vmin=vmin,
        vmax=vmax,
        annot_kws={"fontsize": label_fontsize},
        ax=ax_heatmap,
        **kwargs,
    )

    # Add the colorbar
    if show_colorbar:
        # Use make_axes_locatable to add the colorbar
        divider = make_axes_locatable(ax_heatmap)
        cax = divider.append_axes(
            "right", size=f"{cbar_width_ratio*100}%", pad=cbar_padding
        )

        # Add the colorbar
        cbar = fig.colorbar(
            heatmap.collections[0],
            cax=cax,
            orientation="vertical",
        )

        # Customize the colorbar
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_label(cbar_label, fontsize=label_fontsize)

        # Align the colorbar's height to the heatmap
        pos_heatmap = ax_heatmap.get_position()
        pos_cax = cax.get_position()
        cax.set_position(
            [
                pos_cax.x0,  # Keep x position
                pos_heatmap.y0,  # Align bottom with heatmap
                pos_cax.width,  # Keep the width set by `cbar_width_ratio`
                pos_heatmap.height,  # Match height with heatmap
            ]
        )

        # Remove the spines (despine the colorbar)
        for spine in cax.spines.values():
            spine.set_visible(False)

    # Set the title if provided
    if title:
        plt.title(
            "\n".join(textwrap.wrap(title, width=text_wrap)),
            fontsize=label_fontsize,  # Now using label_fontsize instead
        )

    # Apply custom labels if label_names is provided
    if label_names:
        heatmap.set_xticklabels(
            [
                "\n".join(
                    textwrap.wrap(
                        label_names.get(label.get_text(), label.get_text()),
                        width=text_wrap,
                    )
                )
                for label in heatmap.get_xticklabels()
            ],
            rotation=xlabel_rot,
            fontsize=tick_fontsize,
            ha=xlabel_alignment,
            rotation_mode="anchor",
        )
        heatmap.set_yticklabels(
            [
                "\n".join(
                    textwrap.wrap(
                        label_names.get(label.get_text(), label.get_text()),
                        width=text_wrap,
                    )
                )
                for label in heatmap.get_yticklabels()
            ],
            rotation=ylabel_rot,
            fontsize=tick_fontsize,
            va=ylabel_alignment,
        )
    else:
        # Directly set rotation and alignment for x-axis labels
        heatmap.set_xticklabels(
            [
                "\n".join(textwrap.wrap(label.get_text(), width=text_wrap))
                for label in heatmap.get_xticklabels()
            ],
            rotation=xlabel_rot,
            ha=xlabel_alignment,
            fontsize=tick_fontsize,
            rotation_mode="anchor",
        )

        # Directly set rotation and alignment for y-axis labels with wrapping
        heatmap.set_yticklabels(
            [
                "\n".join(textwrap.wrap(label.get_text(), width=text_wrap))
                for label in heatmap.get_yticklabels()
            ],
            rotation=ylabel_rot,
            va=ylabel_alignment,
            fontsize=tick_fontsize,
        )

    # Adjust layout to prevent overlap
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.1,
        wspace=cbar_padding,
    )

    # Determine the filename title for saving, using the default if None
    filename_title = title or "Correlation Matrix"

    # Set the plot title only if a title is explicitly provided
    if title:
        plt.title(
            "\n".join(textwrap.wrap(title, width=text_wrap)),
            fontsize=label_fontsize,
        )

    # Save the plot if save_plots is True
    if save_plots:
        safe_title = filename_title.replace(" ", "_").replace(":", "").lower()

        if image_path_png:
            filename_png = f"{safe_title}.png"
            plt.savefig(
                os.path.join(image_path_png, filename_png),
                bbox_inches="tight",
            )
        if image_path_svg:
            filename_svg = f"{safe_title}.svg"
            plt.savefig(
                os.path.join(image_path_svg, filename_svg),
                bbox_inches="tight",
            )

    plt.show()


################################################################################
############################## Data Doctor #####################################
################################################################################


def data_doctor(
    df,
    feature_name,
    data_fraction=1,
    scale_conversion=None,
    scale_conversion_kws=None,
    apply_cutoff=False,
    lower_cutoff=None,
    upper_cutoff=None,
    show_plot=True,
    plot_type="all",
    xlim=None,
    kde_ylim=None,
    hist_ylim=None,
    box_violin_ylim=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    apply_as_new_col_to_df=False,
    kde_kws=None,
    hist_kws=None,
    box_violin_kws=None,
    box_violin="boxplot",
    label_fontsize=12,
    tick_fontsize=10,
    random_state=None,
    figsize=(18, 6),
):
    """
    Analyze and transform a specific feature in a DataFrame, with options for
    scaling, applying cutoffs, and visualizing the results. This function also
    allows for the creation of a new column with the transformed data if
    specified. Plots can be saved in PNG or SVG format with filenames that
    incorporate the `plot_type`, `feature_name`, `scale_conversion`, and
    `cutoff` if cutoffs are applied.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the feature to analyze.

    feature_name : str
        The name of the feature (column) to analyze.

    data_fraction : float, optional (default=1)
        Fraction of the data to analyze. Useful for large datasets where a
        sample can represent the population. If `apply_as_new_col_to_df=True`,
        the full dataset is used (data_fraction=1).

    scale_conversion : str, optional
        Type of conversion to apply to the feature. Options include:
            - 'abs': Absolute values
            - 'log': Natural logarithm
            - 'sqrt': Square root
            - 'cbrt': Cube root
            - 'reciprocal': Reciprocal transformation
            - 'stdrz': Standardized (z-score)
            - 'minmax': Min-Max scaling
            - 'boxcox': Box-Cox transformation (positive values only; supports
                        `lmbda` for specific lambda or `alpha` for confidence
                        interval)
            - 'robust': Robust scaling (median and IQR)
            - 'maxabs': Max-abs scaling
            - 'exp': Exponential transformation
            - 'logit': Logit transformation (values between 0 and 1)
            - 'arcsinh': Inverse hyperbolic sine
            - 'square': Squaring the values
            - 'power': Power transformation (Yeo-Johnson).
        Defaults to None (no conversion).

    scale_conversion_kws : dict, optional
        Additional keyword arguments to pass to the scaling functions, such as:
            - 'alpha' for Box-Cox transformation (returns a confidence interval
               for lambda)
            - 'lmbda' for a specific Box-Cox transformation value
            - 'quantile_range' for robust scaling.

    apply_cutoff : bool, optional (default=False)
        Whether to apply upper and/or lower cutoffs to the feature.

    lower_cutoff : float, optional
        Lower bound to apply if `apply_cutoff=True`. Defaults to None.

    upper_cutoff : float, optional
        Upper bound to apply if `apply_cutoff=True`. Defaults to None.

    show_plot : bool, optional (default=True)
        Whether to display plots of the transformed feature: KDE, histogram, and
        boxplot/violinplot.

    plot_type : str, list, or tuple, optional (default="all")
        Specifies the type of plot(s) to produce. Options are:
            - 'all': Generates KDE, histogram, and boxplot/violinplot.
            - 'kde': KDE plot only.
            - 'hist': Histogram plot only.
            - 'box_violin': Boxplot or violin plot only (specified by
                            `box_violin`).
        If a list or tuple is provided (e.g., `plot_type=["kde", "hist"]`),
        the specified plots are displayed in a single row with sufficient
        spacing. A `ValueError` is raised if an invalid plot type is included.

    figsize : tuple or list, optional (default=(18, 6))
        Specifies the figure size for the plots. This applies to all plot types,
        including single plots (when `plot_type` is set to "kde", "hist", or
        "box_violin") and multi-plot layout when `plot_type` is "all".

    xlim : tuple or list, optional
        Limits for the x-axis in all plots, specified as (xmin, xmax).

    kde_ylim : tuple or list, optional
        Limits for the y-axis in the KDE plot, specified as (ymin, ymax).

    hist_ylim : tuple or list, optional
        Limits for the y-axis in the histogram plot, specified as (ymin, ymax).

    box_violin_ylim : tuple or list, optional
        Limits for the y-axis in the boxplot or violin plot, specified as
        (ymin, ymax).

    save_plot : bool, optional (default=False)
        Whether to save the plots as PNG and/or SVG images. If `True`, the user
        must specify at least one of `image_path_png` or `image_path_svg`,
        otherwise a `ValueError` is raised.

    image_path_png : str, optional
        Directory path to save the plot as a PNG file. Only used if
        `save_plot=True`.

    image_path_svg : str, optional
        Directory path to save the plot as an SVG file. Only used if
        `save_plot=True`.

    apply_as_new_col_to_df : bool, optional (default=False)
        Whether to create a new column in the DataFrame with the transformed
        values. If True, the new column name is generated based on the
        feature name and the transformation applied:
            - `<feature_name>_<scale_conversion>`: If a transformation is
            applied.
            - `<feature_name>_w_cutoff`: If only cutoffs are applied.
        For Box-Cox transformation, if `alpha` is specified, the confidence
        interval for lambda is displayed. If `lmbda` is specified, the lambda
        value is displayed.

    kde_kws : dict, optional
        Additional keyword arguments to pass to the KDE plot (seaborn.kdeplot).

    hist_kws : dict, optional
        Additional keyword arguments to pass to the histogram plot
        (seaborn.histplot).

    box_violin_kws : dict, optional
        Additional keyword arguments to pass to either boxplot or violinplot.

    box_violin : str, optional (default="boxplot")
        Specifies whether to plot a 'boxplot' or 'violinplot' if `plot_type` is
        set to 'box_violin'.

    label_fontsize : int, optional (default=12)
        Font size for the axis labels and plot titles.

    tick_fontsize : int, optional (default=10)
        Font size for the tick labels on both axes.

    random_state : int, optional
        Seed for reproducibility when sampling the data.

    Returns:
    --------
    None
        Displays the feature name, descriptive statistics, quartile information,
        and outlier details. If a new column is created, confirms the new
        column's addition to the DataFrame. Also, for Box-Cox transformation,
        prints the lambda value (if provided) or confidence interval for lambda
        (if `alpha` is provided).

    Raises:
    -------
    ValueError
        If an invalid `scale_conversion` is provided.

    ValueError
        If Box-Cox transformation is applied to non-positive values.

    ValueError
        If `save_plot=True` but neither `image_path_png` nor `image_path_svg` is
        provided.

    ValueError
        If an invalid option is provided for `box_violin`.

    ValueError
        If an invalid option is provided for `plot_type`.

    ValueError
        If the length of transformed data does not match the original feature
        length.

    Notes:
    ------
    - When saving plots, the filename will include the `feature_name`,
      `scale_conversion`, and each selected `plot_type` to allow easy
      identification. If `plot_type` includes 'box_violin', the filename will
      reflect the user's specific choice of either 'boxplot' or 'violinplot' as
      set in `box_violin`. Additionally, if `apply_cutoff=True`, "cutoff" is
      appended to the filename. For example, if `feature_name` is "age",
      `scale_conversion` is "boxcox", and `plot_type` is ["kde", "hist",
      "box_violin"] with `box_violin` set to "boxplot", the filename will be:
      `age_boxcox_kde_hist_boxplot_cutoff.png` or
      `age_boxcox_kde_hist_boxplot_cutoff.svg`.

    - The cutoff values (if applied) are displayed as text at the bottom of
      the figure, with thousands separators for readability. If `plot_type="all"`,
      the text is displayed in a separate row. For custom plot lists, the text
      appears centered below the figure.
    """

    # Suppress warnings for division by zero, or invalid values in subtract
    np.seterr(divide="ignore", invalid="ignore")

    # If the user specifies apply_as_new_col_to_df, check for valid conditions
    if apply_as_new_col_to_df:
        if scale_conversion is None and not apply_cutoff:
            raise ValueError(
                "When applying a new column with `apply_as_new_col_to_df=True`, "
                "you must specify either a `scale_conversion` or set "
                "`apply_cutoff=True`."
            )

    # If conversion will be applied to a new column, set sample_frac to 1
    if apply_as_new_col_to_df:
        data_fraction = 1  # change the sample fraction value to 100 percent, to

    # Define valid scale conversions
    valid_conversions = [
        "logit",
        "abs",
        "log",
        "sqrt",
        "cbrt",
        "reciprocal",
        "stdrz",
        "minmax",
        "robust",
        "maxabs",
        "exp",
        "arcsinh",
        "square",
        "boxcox",
        "power",
        None,
    ]

    # Check if scale_conversion is valid
    if scale_conversion not in valid_conversions:
        raise ValueError(
            f"Invalid scale_conversion '{scale_conversion}'. "
            f"Valid options are: {valid_conversions[:-1]}"
        )

    new_col_name = feature_name

    # Sample the data once to ensure consistency in transformations
    # Convert data according to scale_conversion selection

    sampled_feature = df.sample(
        frac=data_fraction,
        random_state=random_state,
    )[feature_name]

    # New column name options when apply_as_new_col_to_df
    if apply_as_new_col_to_df:
        if scale_conversion is None and apply_cutoff:
            new_col_name = feature_name + "_w_cutoff"
        elif scale_conversion is not None:
            new_col_name = feature_name + "_" + scale_conversion

    # Initialize scale_conversion_kws if None
    scale_conversion_kws = scale_conversion_kws or {}

    # Transformation logic
    if scale_conversion == "logit":
        if np.any((sampled_feature <= 0) | (sampled_feature >= 1)):
            raise ValueError(
                "Logit transformation requires values to be between 0 and 1. "
                "Consider using a scaling method such as min-max scaling first."
            )
        from scipy.special import logit

        feature_ = logit(sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "abs":
        feature_ = np.abs(sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "log":
        feature_ = np.log(sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "sqrt":
        feature_ = np.sqrt(sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "cbrt":
        feature_ = np.cbrt(sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "reciprocal":
        feature_ = np.divide(1, sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "stdrz":
        mean = np.mean(sampled_feature)
        std = np.std(sampled_feature)
        feature_ = np.divide(
            sampled_feature - mean,
            std,
            **scale_conversion_kws,
        )

    elif scale_conversion == "minmax":
        min_val = np.min(sampled_feature)
        max_val = np.max(sampled_feature)
        feature_ = np.divide(
            sampled_feature - min_val,
            max_val - min_val,
            **scale_conversion_kws,
        )

    elif scale_conversion == "robust":
        # Extract optional keyword arguments for RobustScaler
        # Default to True (center by median)
        with_centering = scale_conversion_kws.get("with_centering", True)
        # Default to IQR
        quantile_range = scale_conversion_kws.get(
            "quantile_range",
            (25.0, 75.0),
        )

        # Apply RobustScaler with optional kwargs
        scaler = RobustScaler(
            with_centering=with_centering,
            quantile_range=quantile_range,
        )
        feature_ = scaler.fit_transform(
            sampled_feature.values.reshape(-1, 1),
        ).flatten()

    elif scale_conversion == "maxabs":
        max_abs = np.max(np.abs(sampled_feature))
        # This directly divides by max abs value
        feature_ = sampled_feature / max_abs

    elif scale_conversion == "exp":
        feature_ = np.exp(
            sampled_feature,
            **scale_conversion_kws,
        )

    elif scale_conversion == "arcsinh":
        feature_ = np.arcsinh(
            sampled_feature,
            **scale_conversion_kws,
        )

    elif scale_conversion == "square":
        feature_ = np.square(
            sampled_feature,
            **scale_conversion_kws,
        )

    elif scale_conversion == "boxcox":
        if np.any(sampled_feature <= 0):
            raise ValueError(
                "Box-Cox transformation requires strictly positive values."
            )

        # Initialize scale_conversion_kws if None
        scale_conversion_kws = scale_conversion_kws or {}

        # Apply Box-Cox transformation
        try:
            if "alpha" in scale_conversion_kws:
                # Apply Box-Cox transformation and get confidence interval
                feature_array, box_cox_lmbda, conf_interval = stats.boxcox(
                    sampled_feature, **scale_conversion_kws
                )
            else:
                # Apply Box-Cox transformation and get single lambda
                feature_array, box_cox_lmbda = stats.boxcox(
                    sampled_feature, **scale_conversion_kws
                )

        except ValueError as ve:
            raise ValueError(f"Error during Box-Cox transformation: {ve}")

        # Ensure feature_array is a 1D array
        feature_array = np.asarray(feature_array).flatten()

        # Check if the length of feature_array matches the sampled feature
        if len(feature_array) != len(sampled_feature):
            raise ValueError(
                f"Length of transformed data ({len(feature_array)}) does not match "
                f"the length of the sampled feature ({len(sampled_feature)})"
            )

        # Convert the feature_array into a pandas Series with the correct index
        feature_ = pd.Series(feature_array, index=sampled_feature.index)

    elif scale_conversion == "power":
        pt = PowerTransformer(**scale_conversion_kws)
        feature_array = pt.fit_transform(
            sampled_feature.values.reshape(-1, 1)
        ).flatten()
        feature_ = pd.Series(feature_array)  # Do not specify index

    else:
        feature_ = sampled_feature.copy()
        if scale_conversion is None:
            scale_conversion = "None"

    # Apply cutoffs if specified
    if apply_cutoff:
        if lower_cutoff is not None:
            feature_ = np.where(
                feature_ < lower_cutoff,
                lower_cutoff,
                feature_,
            )
        if upper_cutoff is not None:
            feature_ = np.where(
                feature_ > upper_cutoff,
                upper_cutoff,
                feature_,
            )

    # Ensure feature_ is a pandas Series
    if not isinstance(feature_, pd.Series):
        feature_ = pd.Series(feature_)

    print("DATA DOCTOR SUMMARY REPORT".center(52))

    # ASCII table for statistical data
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Feature':<28} | {feature_name:<18} |")
    print(f"+{'-'*30}+{'-'*20}+")

    # Header for the statistical section
    print(f"| {'Statistic':<28} | {'Value':<18} |")
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Min':<28} | {np.min(feature_):>18,.4f} |")
    print(f"| {'Max':<28} | {np.max(feature_):>18,.4f} |")
    print(f"| {'Mean':<28} | {np.mean(feature_):>18,.4f} |")
    print(f"| {'Median':<28} | {np.median(feature_):>18,.4f} |")
    print(f"| {'Std Dev':<28} | {np.std(feature_):>18,.4f} |")

    # Header for the quartiles section
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Quartile':<28} | {'Value':<18} |")
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Q1 (25%)':<28} | {np.quantile(feature_, 0.25):>18,.4f} |")
    print(f"| {'Q2 (50% = Median)':<28} | {np.quantile(feature_, 0.50):>18,.4f} |")
    print(f"| {'Q3 (75%)':<28} | {np.quantile(feature_, 0.75):>18,.4f} |")
    print(
        f"| {'IQR':<28} | "
        f"{(np.quantile(feature_, 0.75) - np.quantile(feature_, 0.25)):>18,.4f} |"
    )

    # Calculate the first quartile (25th percentile) of the feature's values.
    Q1 = np.quantile(feature_, 0.25)

    # Calculate the third quartile (75th percentile) of the feature's values.
    Q3 = np.quantile(feature_, 0.75)

    # Compute the Interquartile Range (IQR) by subtracting Q1 from Q3.
    # The IQR represents the middle 50% of the data, providing a measure of
    # variability.
    IQR = Q3 - Q1

    # Calculate the lower bound for outliers using the IQR method.
    # This boundary is set at 1.5 * IQR below the first quartile (Q1).
    # Values below this bound are considered potential outliers.
    outlier_lower_bound = Q1 - 1.5 * IQR

    # Calculate the upper bound for outliers using the IQR method.
    # This boundary is set at 1.5 * IQR above the third quartile (Q3).
    # Values above this bound are considered potential outliers.
    outlier_upper_bound = Q3 + 1.5 * IQR

    # Header for the outlier section
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Outlier Bound':<28} | {'Value':<18} |")
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Lower Bound':<28} | {outlier_lower_bound:>18,.4f} |")
    print(f"| {'Upper Bound':<28} | {outlier_upper_bound:>18,.4f} |")
    print(f"+{'-'*30}+{'-'*20}+")

    # Add column to dataframe along with data
    if apply_as_new_col_to_df and data_fraction == 1:
        # Ensure feature_ has the same length as df
        if len(feature_) != len(df):
            raise ValueError(
                "Length of transformed feature does not match length of DataFrame."
            )

        # Assign values directly
        df[new_col_name] = feature_.values
        print()
        print("New Column Name:", new_col_name)
        # Print lambda if 'lmbda' was specified
        if "lmbda" in scale_conversion_kws:
            print(f"Box-Cox Lambda (provided): {scale_conversion_kws['lmbda']:.4f}")

        # Print the lambda or confidence interval based on the transformation
        elif scale_conversion == "boxcox":
            if "alpha" in scale_conversion_kws:
                # Confidence interval is available, so print it
                if "conf_interval" in locals():
                    print(
                        f"Box-Cox C.I. for Lambda: ({conf_interval[0]:.4f}, "
                        f"{conf_interval[1]:.4f})"
                    )
            else:
                # No alpha, so we print the single lambda value
                if "box_cox_lmbda" in locals():
                    print(f"Box-Cox Lambda: {round(box_cox_lmbda, 4)}")

    elif apply_as_new_col_to_df and data_fraction != 1:
        print("New Column Name:", new_col_name)
        print(
            "NOTE: Column was not added to dataframe as sample_frac is set to "
            + str(data_fraction)
            + " and not to 1, representing 100 percent."
        )

    # Update lower_cutoff and upper_cutoff values to represent any value updates
    # made in steps above...to ensure the xlabel reflects these values

    lower_cutoff = round(np.min(feature_), 4)
    upper_cutoff = round(np.max(feature_), 4)

    # Convert plot_type to a list if it’s a single string
    if isinstance(plot_type, str):
        if plot_type == "all":
            plot_type = ["kde", "hist", "box_violin"]
        else:
            plot_type = [plot_type]
    elif not isinstance(plot_type, (list, tuple)):
        raise ValueError("plot_type must be a string, list, or tuple.")

    # Verify that all plot types are valid
    valid_plot_types = ["kde", "hist", "box_violin"]
    invalid_plots = [ptype for ptype in plot_type if ptype not in valid_plot_types]
    if invalid_plots:
        raise ValueError(
            f"Invalid plot type(s) {invalid_plots}. "
            f"Valid options are: {valid_plot_types}"
        )

    # Determine layout based on the number of specified plot types
    n_plots = len(plot_type)
    n_rows = 1 if n_plots <= 3 else 2
    n_cols = n_plots if n_plots <= 3 else (n_plots + 1) // 2

    # Increase figure size slightly to create more space
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figsize[0] + 2, figsize[1] + 2), squeeze=False
    )

    # Flatten axes for easy indexing if there are multiple rows/columns
    axes = axes.flatten()

    # Plot based on specified plot types
    if show_plot:
        for i, ptype in enumerate(plot_type):
            ax = axes[i]

            # Check if x-axis values are large (in the hundreds of thousands or above)
            if feature_.max() >= 100_000:
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(
                    style="sci", axis="x", scilimits=(5, 5)
                )  # Set only for 100,000+

            if ptype == "kde":
                sns.kdeplot(
                    x=feature_,
                    ax=ax,
                    clip=(lower_cutoff, upper_cutoff),
                    warn_singular=False,
                    **(kde_kws or {}),
                )
                ax.set_title(
                    f"KDE Plot: {feature_name} (Scale: {scale_conversion})",
                    fontsize=label_fontsize,
                    pad=25,  # Increased padding between title and plot
                )
                ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize)
                ax.set_ylabel("Density", fontsize=label_fontsize)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                if xlim:
                    ax.set_xlim(xlim)
                if kde_ylim:
                    ax.set_ylim(kde_ylim)

            elif ptype == "hist":
                sns.histplot(x=feature_, ax=ax, **(hist_kws or {}))
                ax.set_title(
                    f"Histplot: {feature_name} (Scale: {scale_conversion})",
                    fontsize=label_fontsize,
                    pad=25,  # Increased padding
                )
                ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize)
                ax.set_ylabel("Count", fontsize=label_fontsize)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                if xlim:
                    ax.set_xlim(xlim)
                if hist_ylim:
                    ax.set_ylim(hist_ylim)

            elif ptype == "box_violin":
                if box_violin == "boxplot":
                    sns.boxplot(
                        x=feature_,
                        ax=ax,
                        **(box_violin_kws or {}),
                    )
                    ax.set_title(
                        f"Boxplot: {feature_name} (Scale: {scale_conversion})",
                        fontsize=label_fontsize,
                        pad=25,  # Increased padding
                    )
                elif box_violin == "violinplot":
                    sns.violinplot(
                        x=feature_,
                        ax=ax,
                        **(box_violin_kws or {}),
                    )
                    ax.set_title(
                        f"Violinplot: {feature_name} (Scale: {scale_conversion})",
                        fontsize=label_fontsize,
                        pad=25,  # Increased padding
                    )
                ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize)
                ax.set_ylabel("", fontsize=label_fontsize)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                if xlim:
                    ax.set_xlim(xlim)
                if box_violin_ylim:
                    ax.set_ylim(box_violin_ylim)

    # Display the cutoff text universally
    if plot_type == ["kde", "hist", "box_violin"]:  # When using "all"
        # configuration. Check if we have a 2D grid of axes or a single row
        if n_rows > 1:
            # Use dedicated row for text when in subplot mode with multiple rows
            axes[1, 0].text(
                0,
                0.5,
                f"Lower cutoff: {round(lower_cutoff, 2)} |"
                f"Upper cutoff: {round(upper_cutoff, 2)}",
                fontsize=label_fontsize,
                ha="left",
                va="center",
            )
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")
            axes[1, 2].axis("off")
        else:
            # If only a single row of plots, add text below the plots in that row
            fig.text(
                0.5,
                -0.05,  # Position below the plot
                f"Lower cutoff: {lower_cutoff:,.0f} | "
                f"Upper cutoff: {upper_cutoff:,.0f}",
                ha="center",
                fontsize=label_fontsize,
            )
    else:
        # Display text below the single plot row when using custom configuration
        fig.text(
            0.5,
            -0.05,  # Position below the plot
            f"Lower cutoff: {lower_cutoff:,.0f} | "
            f"Upper cutoff: {upper_cutoff:,.0f}",
            ha="center",
            fontsize=label_fontsize,
        )
    # Apply layout adj. to prevent overlaps without affecting inter-plot spacing
    plt.tight_layout(pad=3.0, w_pad=1.5, h_pad=3.0)

    # Check if the user-specified plot type is valid
    if box_violin not in ["boxplot", "violinplot"]:
        raise ValueError(
            f"Invalid plot type '{box_violin}'. "
            f"Valid options are 'boxplot' or 'violinplot'."
        )

    # Check if save_plots=True but no image path is provided
    if save_plot and not image_path_png and not image_path_svg:
        raise ValueError(
            "You must provide either 'image_path_png' or 'image_path_svg' "
            "when 'save_plots=True'."
        )

    # Save the plots if save_plot is True and the paths are provided
    if save_plot:
        # Adjust plot_type for custom labeling of boxplot or violinplot
        adjusted_plot_type = [
            box_violin if ptype == "box_violin" else ptype for ptype in plot_type
        ]

        # Generate a filename based on the feature name, scale conversion, and
        # selected plot types
        plot_type_str = (
            "_".join(adjusted_plot_type)
            if isinstance(plot_type, (list, tuple))
            else adjusted_plot_type[0]
        )

        # Add 'cutoff' to filename if cutoff is applied
        cutoff_str = "_cutoff" if apply_cutoff else ""

        default_filename = (
            f"{feature_name}_"
            f"{scale_conversion if scale_conversion else 'original'}_"
            f"{plot_type_str}{cutoff_str}"
        )

        # Save as PNG if path is provided
        if image_path_png:
            png_filename = f"{image_path_png}/{default_filename}.png"
            plt.savefig(
                png_filename,
                format="png",
                bbox_inches="tight",
            )
            print(f"Plot saved as PNG at {png_filename}")

        # Save as SVG if path is provided
        if image_path_svg:
            svg_filename = f"{image_path_svg}/{default_filename}.svg"
            plt.savefig(
                svg_filename,
                format="svg",
                bbox_inches="tight",
            )
            print(f"Plot saved as SVG at {svg_filename}")


#################################################################################
######################### Crosstab Plotting Function ############################
#################################################################################


def outcome_crosstab_plot(
    df,
    list_name,
    outcome,
    bbox_to_anchor=(0.5, -0.25),
    w_pad=4,
    h_pad=4,
    figsize=(12, 8),
    label_fontsize=12,
    tick_fontsize=10,
    n_rows=None,
    n_cols=None,
    label_0=None,
    label_1=None,
    normalize=False,
    show_value_counts=False,
    color_schema=None,
    save_plots=False,
    image_path_png=None,
    image_path_svg=None,
    string=None,
):
    """
    Generates crosstab bar plots visualizing the relationship between an outcome
    variable and multiple categorical variables.

    Color control (via color_schema):
      - None: default two-tone plus extras.
      - list/tuple: applies to all subplots, repeating if needed.
      - dict: map column names to specific color lists.

    Parameters:
    - df: pandas.DataFrame to plot.
    - list_name: list of str, categorical columns to include.
    - outcome: str, name of the outcome column.
    - bbox_to_anchor: tuple, legend anchor coordinates.
    - w_pad, h_pad: floats, padding for tight_layout.
    - figsize: tuple, figure size.
    - label_fontsize: int, font size for axis labels, titles, legend.
    - tick_fontsize: int, font size for tick labels.
    - n_rows, n_cols: ints, optional grid rows/columns; auto-calculated if omitted.
    - label_0, label_1: str, custom x-axis labels for outcome levels.
    - normalize: bool, False for raw counts, True for normalized proportions.
    - show_value_counts: bool, append counts or percentages to legend entries.
    - color_schema: None, list, or dict for color customization.
    - save_plots: bool, save figures if True.
    - image_path_png, image_path_svg: str paths for saving.
    - string: base filename for saving (sanitized).
    """
    n_vars = len(list_name)

    # Determine grid size
    if n_rows is None and n_cols is None:
        n_cols = math.ceil(math.sqrt(n_vars))
        n_rows = math.ceil(n_vars / n_cols)
    elif n_rows is None:
        n_rows = math.ceil(n_vars / n_cols)
    elif n_cols is None:
        n_cols = math.ceil(n_vars / n_rows)

    # Default color schemes
    default_colors = ["#00BFC4", "#F8766D"]
    extra_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for item, ax in zip(list_name, axes[:n_vars]):
        # Compute crosstab
        if not normalize:
            ylabel = "Count"
            ctab = pd.crosstab(df[outcome], df[item])
        else:
            ylabel = "Percentage"
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
            )
            ctab = pd.crosstab(df[outcome], df[item], normalize="index")

        # Select colors
        n_cats = len(ctab.columns)
        if isinstance(color_schema, dict) and item in color_schema:
            scheme = color_schema[item]
        elif isinstance(color_schema, (list, tuple)):
            scheme = list(color_schema)
        else:
            scheme = default_colors.copy()
        if len(scheme) < n_cats:
            needed = n_cats - len(scheme)
            pool = extra_colors if color_schema is None else scheme
            scheme = (scheme + pool * math.ceil(needed / len(pool)))[:n_cats]

        # Plot
        ctab.plot(kind="bar", stacked=True, rot=0, ax=ax, color=scheme)

        # Formatting
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.set_xlabel("Outcome", fontsize=label_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        if label_0 and label_1:
            ax.set_xticklabels([label_0, label_1], fontsize=tick_fontsize)
        ax.set_title(f"Outcome vs. {item}", fontsize=label_fontsize)

        # Legend labels: counts or percentages
        labels = [str(col) for col in ctab.columns]
        if show_value_counts:
            if not normalize:
                counts = df[item].value_counts()
                labels = [
                    f"{lbl} (n={counts.get(col, 0)})"
                    for lbl, col in zip(labels, ctab.columns)
                ]
            else:
                proportions = df[item].value_counts(normalize=True)
                labels = [
                    f"{lbl} ({proportions.get(col, 0)*100:.0f}%)"
                    for lbl, col in zip(labels, ctab.columns)
                ]

        handles, _ = ax.get_legend_handles_labels()
        leg = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,
            fontsize=label_fontsize,
        )
        for text in leg.get_texts():
            text.set_fontsize(label_fontsize)

    # Hide extra subplots
    for ax in axes[n_vars:]:
        ax.axis("off")

    plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # Save logic
    if save_plots:
        base = string or "crosstab_plot"
        safe = base.replace(" ", "_").replace(":", "").lower()
        if image_path_png:
            plt.savefig(
                os.path.join(image_path_png, f"{safe}.png"), bbox_inches="tight"
            )
        if image_path_svg:
            plt.savefig(
                os.path.join(image_path_svg, f"{safe}.svg"), bbox_inches="tight"
            )

    plt.show()
