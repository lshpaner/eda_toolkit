################################################################################
############################### Library Imports ################################
from .main import *

################################################################################
########################## 3D Partial Dependence Plots #########################
################################################################################


def plot_3d_pdp(
    model,
    dataframe,
    feature_names_list,
    x_label,
    y_label,
    z_label,
    title,
    html_file_path,
    html_file_name,
    image_filename=None,
    x_label_plotly=None,
    y_label_plotly=None,
    z_label_plotly=None,
    plot_type="both",
    matplotlib_colormap=None,
    plotly_colormap="Viridis",
    zoom_out_factor=None,
    wireframe_color=None,
    view_angle=(22, 70),
    figsize=(7, 4.5),
    text_wrap=50,
    horizontal=3,
    depth=5,
    vertical=0.8,
    image_path_png=None,
    image_path_svg=None,
    show_cbar=True,
    grid_resolution=20,
):
    """
    Generate 3D partial dependence plots for two features of a machine learning
    model.

    This function supports both static (Matplotlib) and interactive (Plotly)
    visualizations. It is compatible with both older and newer versions of
    scikit-learn, making it versatile for various environments.

    Parameters
    ----------
    model : estimator object
        A trained machine learning model that implements the `predict`,
        `predict_proba`, or `decision_function` method.

    dataframe : pandas.DataFrame or numpy.ndarray
        The dataset on which the model was trained or a representative sample.
        If a DataFrame is provided, the `feature_names_list` should correspond
        to the column names. If a numpy array is provided, `feature_names_list`
        should correspond to the indices of the columns.

    feature_names_list : list of str
        A list of two feature names or indices corresponding to the features for
        which partial dependence plots are generated.

    x_label : str
        Label for the x-axis in the plots.

    y_label : str
        Label for the y-axis in the plots.

    z_label : str
        Label for the z-axis in the plots.

    title : str
        Title for the plots.

    html_file_path : str
        Path to save the interactive Plotly HTML file.

    html_file_name : str
        Name of the HTML file to save the interactive Plotly plot.

    image_filename : str, optional
        Base filename for saving static Matplotlib plots as PNG and/or SVG.

    x_label_plotly : str, optional
        Custom x-axis label for  interactive Plotly plot. Defaults to `x_label`.

    y_label_plotly : str, optional
        Custom y-axis label for  interactive Plotly plot. Defaults to `y_label`.

    z_label_plotly : str, optional
        Custom z-axis label for  interactive Plotly plot. Defaults to `z_label`.

    plot_type : str, optional, default="both"
        Type of plots to generate. Options are:
        - "static": Generate only static Matplotlib plots.
        - "interactive": Generate only interactive Plotly plots.
        - "both": Generate both static and interactive plots.

    matplotlib_colormap : matplotlib.colors.Colormap, optional
        Custom colormap for the Matplotlib plot. If not provided, a default
        colormap is used.

    plotly_colormap : str, optional, default="Viridis"
        Colormap for the Plotly plot.

    zoom_out_factor : float, optional
        Factor to adjust the zoom level of the Plotly plot.

    wireframe_color : str, optional
        Color for the wireframe in the Matplotlib plot. If `None`, no wireframe
        is plotted.

    view_angle : tuple, optional, default=(22, 70)
        Elevation and azimuthal angles for the Matplotlib plot view.

    figsize : tuple, optional, default=(7, 4.5)
        Figure size for the Matplotlib plot.

    text_wrap : int, optional, default=50
        Maximum width of the title text before wrapping.

    horizontal : float, optional, default=3
        Horizontal camera position for the Plotly plot.

    depth : float, optional, default=5
        Depth camera position for the Plotly plot.

    vertical : float, optional, default=0.8
        Vertical camera position for the Plotly plot.

    image_path_png : str, optional
        Directory path to save the PNG file of the Matplotlib plot.

    image_path_svg : str, optional
        Directory path to save the SVG file of the Matplotlib plot.

    show_cbar : bool, optional, default=True
        Whether to display the color bar in the Matplotlib plot.

    grid_resolution : int, optional, default=20
        The resolution of the grid for computing partial dependence.

    Raises
    ------
    ValueError
        If `plot_type` is not one of "static", "interactive", or "both".

    Notes
    -----
    - This function handles warnings related to scikit-learn's
      `partial_dependence` function, specifically a `FutureWarning` related to
      non-tuple sequences for multidimensional indexing. This warning is
      suppressed as it stems from the internal workings of scikit-learn in
      Python versions like 3.7.4.
    - To maintain compatibility with different versions of scikit-learn, the
      function attempts to use `"values"` for grid extraction in newer versions
      and falls back to `"grid_values"` for older versions.

    """

    ############# Suppress specific FutureWarnings from sklearn ################
    ############################################################################

    # Typically, it is best practice to avoid setting warnings like this inside
    # the function/method itself. However, the following logic mandates this
    # specific use case for the following reasons.
    #
    # There exists an unresolvable future warning for earlier
    # versions of python (i.e., 3.7.4) as it stems from the internal workings of
    # the partial_dependence function in scikit-learn, where non-tuple sequences
    # are being used for multidimensional indexing. Since this is happening
    # within the scikit-learn library and not in this code base. Sklearn cannot
    # be updated from 1.0.2 to 1.3.2 in this Python version, hence this
    # mandated suppression.
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    # Handle both pandas DataFrame and NumPy array inputs
    if isinstance(dataframe, np.ndarray):
        feature_indices = [
            feature_names_list.index(name) for name in feature_names_list
        ]
    else:
        feature_indices = [
            list(dataframe.columns).index(feature_names_list[0]),
            list(dataframe.columns).index(feature_names_list[1]),
        ]

    # Computing the partial dependence
    pdp_results = partial_dependence(
        model,
        X=dataframe,
        features=[(feature_indices[0], feature_indices[1])],
        grid_resolution=grid_resolution,
        kind="average",
    )

    # Attempt to extract grid values and partial dependence values
    try:
        # Newer versions of scikit-learn (post 0.24) return grid values using
        # the "values" key. This applies to versions where the data is stored in
        # a 'Bunch' object, making it necessary to access grid values using the
        # key "values".
        XX, YY = np.meshgrid(pdp_results["values"][0], pdp_results["values"][1])
    except KeyError:
        # Older versions of scikit-learn (pre 0.24) store the grid values using
        # the key "grid_values". In these versions, "values" might not exist,
        # causing a KeyError. We catch this error and fall back to using
        # "grid_values" to maintain compatibility.
        XX, YY = np.meshgrid(
            pdp_results["grid_values"][0], pdp_results["grid_values"][1]
        )

    ZZ = pdp_results["average"].reshape(XX.shape)

    # Check if the plot_type is valid
    if plot_type not in ["static", "interactive", "both"]:
        raise ValueError(
            "Invalid plot_type. Choose from 'static', 'interactive', or 'both'."
        )

    if plot_type in ["both", "interactive"]:
        # Plotly Interactive Plot
        plotly_fig = go.Figure(
            data=[
                go.Surface(
                    z=ZZ,
                    x=XX,
                    y=YY[::-1],
                    colorscale=plotly_colormap,
                    colorbar=dict(
                        len=0.65,
                        thickness=25,
                        yanchor="middle",
                        y=0.5,
                    ),
                )
            ]
        )

        plotly_fig.update_layout(
            title=textwrap.fill(title, text_wrap),
            scene=dict(
                xaxis_title=x_label_plotly or x_label,
                yaxis_title=y_label_plotly or y_label,
                zaxis_title=z_label_plotly or z_label,
                camera=dict(
                    eye=dict(
                        x=horizontal * zoom_out_factor,
                        y=depth * zoom_out_factor,
                        z=vertical * zoom_out_factor,
                    )
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="darkgrey",
                    gridwidth=2,
                    title=dict(text=x_label_plotly or x_label),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="darkgrey",
                    gridwidth=2,
                    title=dict(text=y_label_plotly or y_label),
                ),
                zaxis=dict(
                    showgrid=True,
                    gridcolor="darkgrey",
                    gridwidth=2,
                    title=dict(text=z_label_plotly or z_label),
                ),
            ),
            autosize=False,
            width=900,
            height=750,
            margin=dict(l=65, r=65, b=65, t=100),
        )

        pyo.iplot(plotly_fig)

        full_html_file_path = os.path.join(html_file_path, html_file_name)
        pyo.plot(
            plotly_fig,
            filename=full_html_file_path,
            auto_open=False,
            config={"modeBarButtonsToRemove": ["resetCameraDefault3d"]},
        )

    if plot_type in ["both", "static"]:
        # Prepare custom colormap
        if matplotlib_colormap is None:
            N = 256
            vals = np.ones((N, 4))
            vals[:, 0] = np.linspace(1, 0, N)
            vals[:, 1] = np.linspace(0, 1, N)
            vals[:, 2] = np.linspace(1, 1, N)
            matplotlib_colormap = ListedColormap(vals)

        fig = plt.figure(figsize=figsize, dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xlabel(x_label, fontsize=8, labelpad=-1)
        ax.set_ylabel(y_label, fontsize=8, labelpad=1)
        ax.set_zlabel(z_label, fontsize=8, labelpad=-1)
        ax.xaxis.line.set_color("gray")
        ax.yaxis.line.set_color("gray")
        ax.zaxis.line.set_color("gray")
        ax.view_init(*view_angle)

        ax.set_ylim(YY.max(), YY.min())

        for e in ax.get_yticklabels() + ax.get_xticklabels() + ax.get_zticklabels():
            e.set_fontsize(7)

        surf = ax.plot_surface(XX, YY, ZZ, cmap=matplotlib_colormap, shade=False)

        if wireframe_color:
            ax.plot_wireframe(XX, YY, ZZ, color=wireframe_color, linewidth=0.5)

        if show_cbar:
            cbar = fig.colorbar(surf, shrink=0.3, aspect=20, pad=0.05)
            cbar.ax.tick_params(labelsize=7)

        plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=-0.8)

        plt.title(textwrap.fill(title, text_wrap), fontsize=10, y=0.95, pad=10)

        if image_path_png and image_filename:
            plt.savefig(
                os.path.join(image_path_png, f"{image_filename}.png"),
                bbox_inches="tight",
            )
        if image_path_svg and image_filename:
            plt.savefig(
                os.path.join(image_path_svg, f"{image_filename}.svg"),
                bbox_inches="tight",
            )

        plt.show()


################################################################################
########################   2D Partial Dependence Plots #########################
################################################################################


def plot_2d_pdp(
    model,
    X_train,
    feature_names,
    features,
    title="Partial dependence plot",
    grid_resolution=50,
    plot_type="grid",  # Input to choose between "grid", "individual", or "both"
    grid_figsize=(12, 8),  # Default figure size for grid layout
    individual_figsize=(6, 4),  # Default figure size for individual plots
    label_fontsize=12,  # Parameter to control axis label and title fontsize
    tick_fontsize=10,  # Parameter to control tick label fontsize
    text_wrap=50,  # Add text_wrap parameter for title wrapping
    image_path_png=None,  # Path to save PNG file
    image_path_svg=None,  # Path to save SVG file
    save_plots=None,  # Parameter to control saving plots
):
    """
    Generate partial dependence plots for the specified features using the
    given model.

    Parameters:
    - model: Trained machine learning model to use for generating partial
      dependence plots.
    - X_train: Training data (features) used to generate partial dependence.
    - feature_names: List of feature names corresponding to columns in X_train.
    - features: List of feature indices or tuples of feature indices for which to
      generate partial dependence plots.
    - title: Title for the entire plot.
    - grid_resolution: Resolution of the grid for partial dependence plots
      (default is 50).
    - plot_type: Choose between "grid" for all plots in a grid layout,
      "individual" for separate plots, or "both" for both layouts.
    - grid_figsize: Tuple specifying the figure size for the grid layout
      (default is (12, 8)).
    - individual_figsize: Tuple specifying the figure size for individual plots
      (default is (6, 4)).
    - label_fontsize: Font size for the axis labels and titles (default is 12).
    - tick_fontsize: Font size for the tick labels (default is 10).
    - text_wrap: Maximum width of the title text before wrapping (default is 50).
    - image_path_png: Directory to save PNG files.
    - image_path_svg: Directory to save SVG files.
    - save_plots: String, "all", "individual", "grid", or None to control
      saving plots.
    """

    # Validate save_plots input
    if save_plots not in [None, "all", "individual", "grid"]:
        raise ValueError(
            f"Invalid save_plots value selected. Choose from 'all',"
            f"'individual', 'grid', or None."
        )

    # Check if save_plots is set without image paths
    if save_plots and not (image_path_png or image_path_svg):
        raise ValueError(
            "To save plots, specify 'image_path_png' or " "'image_path_svg'."
        )

    n_features = len(features)

    if plot_type not in ["grid", "individual", "both"]:
        raise ValueError(
            f"Invalid plot_type '{plot_type}'. Choose 'grid', 'individual',"
            f"or 'both'."
        )

    # Determine saving options based on save_plots value
    save_individual = save_plots in ["all", "individual"]
    save_grid = save_plots in ["all", "grid"]

    if plot_type in ["grid", "both"]:
        # Determine grid size based on the number of features
        n_cols = 3  # You can adjust this to change the number of columns
        n_rows = (n_features + n_cols - 1) // n_cols  # Calc. no. of rows needed

        # Create a custom layout with the required grid size
        fig, ax = plt.subplots(n_rows, n_cols, figsize=grid_figsize)
        ax = ax.flatten()  # Flatten the axes array

        # Remove any extra axes if the grid has more slots than needed
        for i in range(n_features, len(ax)):
            fig.delaxes(ax[i])

        # Generate partial dependence plots
        PartialDependenceDisplay.from_estimator(
            model,
            X_train,
            features,
            grid_resolution=grid_resolution,
            feature_names=feature_names,
            ax=ax[:n_features],
        )

        # Set font sizes for labels, ticks, and title
        for axis in ax[:n_features]:
            axis.set_xlabel(axis.get_xlabel(), fontsize=label_fontsize)
            axis.set_ylabel(axis.get_ylabel(), fontsize=label_fontsize)
            axis.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        # Add title with text wrapping and fontsize
        fig.suptitle(
            "\n".join(textwrap.wrap(title, text_wrap)), fontsize=label_fontsize
        )

        # Adjust the spacing between plots
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save grid plot if specified
        if save_grid:
            if image_path_png:
                fig.savefig(
                    os.path.join(image_path_png, "partial_dependence_grid.png"),
                    bbox_inches="tight",
                )
            if image_path_svg:
                fig.savefig(
                    os.path.join(image_path_svg, "partial_dependence_grid.svg"),
                    bbox_inches="tight",
                )

        plt.show()

    if plot_type in ["individual", "both"]:
        # Generate individual plots for each feature
        for i, feature in enumerate(features):
            fig, ax = plt.subplots(figsize=individual_figsize)

            PartialDependenceDisplay.from_estimator(
                model,
                X_train,
                [feature],
                grid_resolution=grid_resolution,
                feature_names=feature_names,
                ax=ax,
            )
            plt.title(
                "\n".join(
                    textwrap.wrap(
                        f"Partial dependence of {feature_names[i]}", text_wrap
                    )
                ),
                fontsize=label_fontsize,
            )

            # Set font sizes for the axis labels and ticks
            plt.xlabel(ax.get_xlabel(), fontsize=label_fontsize)
            plt.ylabel(ax.get_ylabel(), fontsize=label_fontsize)
            plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)

            # Save individual plots if specified
            if save_individual:
                safe_feature_name = (
                    feature_names[i].replace(" ", "_").replace("/", "_per_")
                )
                if image_path_png:
                    plt.savefig(
                        os.path.join(image_path_png, f"{safe_feature_name}.png"),
                        bbox_inches="tight",
                    )
                if image_path_svg:
                    plt.savefig(
                        os.path.join(image_path_svg, f"{safe_feature_name}.svg"),
                        bbox_inches="tight",
                    )

            plt.show()
