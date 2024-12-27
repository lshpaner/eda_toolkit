.. _eda_plots:   

.. _target-link:

.. raw:: html

   <div class="no-click">

.. image:: ../assets/eda_toolkit_logo.svg
   :alt: EDA Toolkit Logo
   :align: left
   :width: 300px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 100px;"></div>

Creating Effective Visualizations
==================================

This section focuses on practical strategies and techniques for designing clear 
and impactful visualizations using the diverse plotting tools provided in the EDA Toolkit.

Heuristics for Visualizations
------------------------------

When creating visualizations, there are several key heuristics to keep in mind:

- **Clarity**: The visualization should clearly convey the intended information without 
  ambiguity.
- **Simplicity**: Avoid overcomplicating visualizations with unnecessary elements; focus on 
  the data and insights.
- **Consistency**: Ensure consistent use of colors, shapes, and scales across visualizations 
  to facilitate comparisons.

Methodologies
-------------

The EDA Toolkit supports the following methodologies for creating effective visualizations:

- **KDE and Histograms Plots**: Useful for showing the distribution of a single variable. 
  When combined, these can provide a clearer picture of data density and distribution.

- **Feature Scaling and Outliers**: Identifying outliers is critical for understanding the data's distribution and potential anomalies. 
  The EDA Toolkit offers various methods for outlier detection, including enhanced visualizations using box plots and scatter plots.

- **Stacked Crosstab Plots**: These are used to display multiple data series on the same chart, comparing 
  cumulative quantities across categories. In addition to the visual stacked bar plots, the corresponding 
  crosstab table is printed alongside the visualization, providing detailed numerical insight into how the 
  data is distributed across different categories. This combination allows for both a visual and tabular 
  representation of categorical data, enhancing interpretability.

- **Box and Violin Plots**: Useful for visualizing the distribution of data points, identifying outliers, 
  and understanding the spread of the data. Box plots are particularly effective when visualizing multiple categories 
  side by side, enabling comparisons across groups. Violin plots provide additional insights by showing the 
  distribution's density, giving a fuller picture of the data's distribution shape.

- **Scatter Plots and Best Fit Lines**: Effective for visualizing relationships between two continuous variables. 
  Scatter plots can also be enhanced with regression lines or trend lines to identify relationships more clearly.

- **Correlation Matrices**: Helpful for visualizing the strength of relationships between multiple variables. 
  Correlation heatmaps use color gradients to indicate the degree of correlation, with options for annotating the values directly on the heatmap.


- **Partial Dependence Plots**: Useful for visualizing the relationship between a target variable and one or more features 
  after accounting for the average effect of the other features. These plots are often used in model interpretability 
  to understand how specific variables influence predictions.


KDE and Histogram Distribution Plots
=======================================


.. raw:: html

    <a id="kde_hist_plots"></a>

KDE Distribution Function
-----------------------------

**Generate KDE or histogram distribution plots for specified columns in a DataFrame.**

The ``kde_distributions`` function is a versatile tool designed for generating 
Kernel Density Estimate (KDE) plots, histograms, or a combination of both for 
specified columns within a DataFrame. This function is particularly useful for 
visualizing the distribution of numerical data across various categories or groups. 
It leverages the powerful seaborn library [2]_ for plotting, which is built on top of 
matplotlib [3]_ and provides a high-level interface for drawing attractive and informative 
statistical graphics.


**Key Features and Parameters**

- **Flexible Plotting**: The function supports creating histograms, KDE plots, or a combination of both for specified columns, allowing users to visualize data distributions effectively.
- **Leverages Seaborn Library**: The function is built on the `seaborn` library, which provides high-level, attractive visualizations, making it easy to create complex plots with minimal code.
- **Customization**: Users have control over plot aesthetics, such as colors, fill options, grid sizes, axis labels, tick marks, and more, allowing them to tailor the visualizations to their needs.
- **Scientific Notation Control**: The function allows disabling scientific notation on the axes, providing better readability for certain types of data.
- **Log Scaling**: The function includes an option to apply logarithmic scaling to specific variables, which is useful when dealing with data that spans several orders of magnitude.
- **Output Options**: The function supports saving plots as PNG or SVG files, with customizable filenames and output directories, making it easy to integrate the plots into reports or presentations.

.. function:: kde_distributions(df, vars_of_interest=None, figsize=(5, 5), grid_figsize=None, hist_color="#0000FF", kde_color="#FF0000", mean_color="#000000", median_color="#000000", hist_edgecolor="#000000", hue=None, fill=True, fill_alpha=1, n_rows=None, n_cols=None, w_pad=1.0, h_pad=1.0, image_path_png=None, image_path_svg=None, image_filename=None, bbox_inches=None, single_var_image_filename=None, y_axis_label="Density", plot_type="both", log_scale_vars=None, bins="auto", binwidth=None, label_fontsize=10, tick_fontsize=10, text_wrap=50, disable_sci_notation=False, stat="density", xlim=None, ylim=None, plot_mean=False, plot_median=False, std_dev_levels=None, std_color="#808080", label_names=None, show_legend=True, **kwargs)

    :param df: The DataFrame containing the data to plot.
    :type df: pandas.DataFrame
    :param vars_of_interest: List of column names for which to generate distribution plots. If 'all', plots will be generated for all numeric columns.
    :type vars_of_interest: list of str, optional
    :param figsize: Size of each individual plot, default is ``(5, 5)``. Used when only one plot is being generated or when saving individual plots.
    :type figsize: tuple of int, optional
    :param grid_figsize: Size of the overall grid of plots when multiple plots are generated in a grid. Ignored when only one plot is being generated or when saving individual plots. If not specified, it is calculated based on ``figsize``, ``n_rows``, and ``n_cols``.
    :type grid_figsize: tuple of int, optional
    :param hist_color: Color of the histogram bars, default is ``'#0000FF'``.
    :type hist_color: str, optional
    :param kde_color: Color of the KDE plot, default is ``'#FF0000'``.
    :type kde_color: str, optional
    :param mean_color: Color of the mean line if ``plot_mean`` is True, default is ``'#000000'``.
    :type mean_color: str, optional
    :param median_color: Color of the median line if ``plot_median`` is True, default is ``'#000000'``.
    :type median_color: str, optional
    :param hist_edgecolor: Color of the histogram bar edges, default is ``'#000000'``.
    :type hist_edgecolor: str, optional
    :param hue: Column name to group data by, adding different colors for each group.
    :type hue: str, optional
    :param fill: Whether to fill the histogram bars with color, default is ``True``.
    :type fill: bool, optional
    :param fill_alpha: Alpha transparency for the fill color of the histogram bars, where ``0`` is fully transparent and ``1`` is fully opaque. Default is ``1``.
    :type fill_alpha: float, optional
    :param n_rows: Number of rows in the subplot grid. If not provided, it will be calculated automatically.
    :type n_rows: int, optional
    :param n_cols: Number of columns in the subplot grid. If not provided, it will be calculated automatically.
    :type n_cols: int, optional
    :param w_pad: Width padding between subplots, default is ``1.0``.
    :type w_pad: float, optional
    :param h_pad: Height padding between subplots, default is ``1.0``.
    :type h_pad: float, optional
    :param image_path_png: Directory path to save the PNG image of the overall distribution plots.
    :type image_path_png: str, optional
    :param image_path_svg: Directory path to save the SVG image of the overall distribution plots.
    :type image_path_svg: str, optional
    :param image_filename: Filename to use when saving the overall distribution plots.
    :type image_filename: str, optional
    :param bbox_inches: Bounding box to use when saving the figure. For example, ``'tight'``.
    :type bbox_inches: str, optional
    :param single_var_image_filename: Filename to use when saving the separate distribution plots. The variable name will be appended to this filename. This parameter uses ``figsize`` for determining the plot size, ignoring ``grid_figsize``.
    :type single_var_image_filename: str, optional
    :param y_axis_label: The label to display on the ``y-axis``, default is ``'Density'``.
    :type y_axis_label: str, optional
    :param plot_type: The type of plot to generate, options are ``'hist'``, ``'kde'``, or ``'both'``. Default is ``'both'``.
    :type plot_type: str, optional
    :param log_scale_vars: Variable name(s) to apply log scaling. Can be a single string or a list of strings.
    :type log_scale_vars: str or list of str, optional
    :param bins: Specification of histogram bins, default is ``'auto'``.
    :type bins: int or sequence, optional
    :param binwidth: Width of each bin, overrides bins but can be used with binrange.
    :type binwidth: float, optional
    :param label_fontsize: Font size for axis labels, including xlabel, ylabel, and tick marks, default is ``10``.
    :type label_fontsize: int, optional
    :param tick_fontsize: Font size for tick labels on the axes, default is ``10``.
    :type tick_fontsize: int, optional
    :param text_wrap: Maximum width of the title text before wrapping, default is ``50``.
    :type text_wrap: int, optional
    :param disable_sci_notation: Toggle to disable scientific notation on axes, default is ``False``.
    :type disable_sci_notation: bool, optional
    :param stat: Aggregate statistic to compute in each bin (e.g., ``'count'``, ``'frequency'``, ``'probability'``, ``'percent'``, ``'density'``), default is ``'density'``.
    :type stat: str, optional
    :param xlim: Limits for the ``x-axis`` as a tuple or list of (``min``, ``max``).
    :type xlim: tuple or list, optional
    :param ylim: Limits for the ``y-axis`` as a tuple or list of (``min``, ``max``).
    :type ylim: tuple or list, optional
    :param plot_mean: Whether to plot the mean as a vertical line, default is ``False``.
    :type plot_mean: bool, optional
    :param plot_median: Whether to plot the median as a vertical line, default is ``False``.
    :type plot_median: bool, optional
    :param std_dev_levels: Levels of standard deviation to plot around the mean.
    :type std_dev_levels: list of int, optional
    :param std_color: Color(s) for the standard deviation lines, default is ``'#808080'``.
    :type std_color: str or list of str, optional
    :param label_names: Custom labels for the variables of interest. Keys should be column names, and values should be the corresponding labels to display.
    :type label_names: dict, optional
    :param show_legend: Whether to show the legend on the plots, default is ``True``.
    :type show_legend: bool, optional
    :param kwargs: Additional keyword arguments passed to the Seaborn plotting function.
    :type kwargs: additional keyword arguments
    
    :raises ValueError: 
        - If ``plot_type`` is not one of ``'hist'``, ``'kde'``, or ``'both'``.
        - If ``stat`` is not one of ``'count'``, ``'density'``, ``'frequency'``, ``'probability'``, ``'proportion'``, ``'percent'``.
        - If ``log_scale_vars`` contains variables that are not present in the DataFrame.
        - If ``fill`` is set to ``False`` and ``hist_edgecolor`` is not the default.
        - If ``grid_figsize`` is provided when only one plot is being created.
    
    :raises UserWarning:
        - If both ``bins`` and ``binwidth`` are specified, which may affect performance.

    :returns: ``None``

.. admonition:: Notes

    If you do not set ``n_rows`` or ``n_cols`` to any values, the function will 
    automatically calculate and create a grid based on the number of variables being 
    plotted, ensuring an optimal arrangement of the plots.

    To save images, the paths for ``image_path_png`` or ``image_path_svg`` must be specified. 
    The trigger for saving plots is the presence of ``image_filename`` as a string.

\

.. raw:: html
    
    <br>



KDE and Histograms Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the below example, the ``kde_distributions`` function is used to generate 
histograms for several variables of interest: ``"age"``, ``"education-num"``, and
``"hours-per-week"``. These variables represent different demographic and 
financial attributes from the dataset. The ``plot_type="both"`` parameter ensures that a 
Kernel Density Estimate (KDE) plot is overlaid on the histograms, providing a 
smoothed representation of the data's probability density.

The visualizations are arranged in a single row of four columns, as specified 
by ``n_rows=1`` and ``n_cols=3``, respectively. The overall size of the grid 
figure is set to `14 inches` wide and `4 inches tall` (``grid_figsize=(14, 4)``), 
while each individual plot is configured to be `4 inches` by `4 inches` 
(``single_figsize=(4, 4)``). The ``fill=True`` parameter fills the histogram 
bars with color, and the spacing between the subplots is managed using 
``w_pad=1`` and ``h_pad=1``, which add `1 inch` of padding both horizontally and 
vertically.

To handle longer titles, the ``text_wrap=50`` parameter ensures that the title 
text wraps to a new line after `50 characters`. The ``bbox_inches="tight"`` setting 
is used when saving the figure, ensuring that it is cropped to remove any excess 
whitespace around the edges. The variables specified in ``vars_of_interest`` are 
passed directly to the function for visualization.

Each plot is saved individually with filenames that are prefixed by 
``"kde_density_single_distribution"``, followed by the variable name. The ```y-axis```
for all plots is labeled as "Density" (``y_axis_label="Density"``), reflecting that 
the height of the bars or KDE line represents the data's density. The histograms 
are divided into `10 bins` (``bins=10``), offering a clear view of the distribution 
of each variable.

Additionally, the font sizes for the axis labels and tick labels 
are set to `16 points` (``label_fontsize=16``) and `14 points` (``tick_fontsize=14``), 
respectively, ensuring that all text within the plots is legible and well-formatted.


.. code-block:: python

    from eda_toolkit import kde_distributions

    vars_of_interest = [
        "age",
        "education-num",
        "hours-per-week",
    ]

    kde_distributions(
        df=df,
        n_rows=1,
        n_cols=3,
        grid_figsize=(14, 4),  
        fill=True,
        fill_alpha=0.60,
        text_wrap=50,
        bbox_inches="tight",
        vars_of_interest=vars_of_interest,
        y_axis_label="Density",
        bins=10,
        plot_type="both", 
        label_fontsize=16,  
        tick_fontsize=14,  
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/kde_density_distributions.svg
   :alt: KDE Distributions - KDE (+) Histograms (Density)
   :align: center
   :width: 950px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Histogram Example (Density)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, the ``kde_distributions()`` function is used to generate histograms for 
the variables ``"age"``, ``"education-num"``, and ``"hours-per-week"`` but with 
``plot_type="hist"``, meaning no KDE plots are included—only histograms are displayed. 
The plots are arranged in a single row of four columns (``n_rows=1, n_cols=3``), 
with a grid size of `14x4 inches` (``grid_figsize=(14, 4)``). The histograms are 
divided into `10 bins` (``bins=10``), and the ``y-axis`` is labeled "Density" (``y_axis_label="Density"``).
Font sizes for the axis labels and tick labels are set to `16` and `14` points, 
respectively, ensuring clarity in the visualizations. This setup focuses on the 
histogram representation without the KDE overlay.


.. code-block:: python

    from eda_toolkit import kde_distributions

    vars_of_interest = [
        "age",
        "education-num",
        "hours-per-week",
    ]

    kde_distributions(
        df=df,
        n_rows=1,
        n_cols=3,
        grid_figsize=(14, 4), 
        fill=True,
        text_wrap=50,
        bbox_inches="tight",
        vars_of_interest=vars_of_interest,
        y_axis_label="Density",
        bins=10,
        plot_type="hist",
        label_fontsize=16, 
        tick_fontsize=14,  
    )


.. raw:: html

   <div class="no-click">

.. image:: ../assets/hist_density_distributions.svg
   :alt: KDE Distributions - Histograms (Density)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Histogram Example (Count)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, the ``kde_distributions()`` function is modified to generate histograms 
with a few key changes. The ``hist_color`` is set to `"orange"`, changing the color of the 
histogram bars. The ``y-axis`` label is updated to "Count" (``y_axis_label="Count"``), 
reflecting that the histograms display the count of observations within each bin. 
Additionally, the stat parameter is set to ``"Count"`` to show the actual counts instead of 
densities. The rest of the parameters remain the same as in the previous example, 
with the plots arranged in a single row of four columns (``n_rows=1, n_cols=3``), 
a grid size of `14x4 inches`, and a bin count of `10`. This setup focuses on 
visualizing the raw counts in the dataset using orange-colored histograms.

.. code-block:: python

    from eda_toolkit import kde_distributions

    vars_of_interest = [
        "age",
        "education-num",
        "hours-per-week",
    ]

    kde_distributions(
        df=df,
        n_rows=1,
        n_cols=3,
        grid_figsize=(14, 4),  
        text_wrap=50,
        hist_color="orange",
        bbox_inches="tight",
        vars_of_interest=vars_of_interest,
        y_axis_label="Count",
        bins=10,
        plot_type="hist",
        stat="Count",
        label_fontsize=16, 
        tick_fontsize=14, 
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/count_hist_distributions.svg
   :alt: KDE Distributions - Histograms (Count)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>

Histogram Example - (Mean and Median) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, the ``kde_distributions()`` function is customized to generate 
histograms that include mean and median lines. The ``mean_color`` is set to ``"blue"`` 
and the ``median_color`` is set to ``"black"``, allowing for a clear distinction
between the two statistical measures. The function parameters are adjusted to 
ensure that both the mean and median lines are plotted ``(plot_mean=True, plot_median=True)``. 
The ``y_axis_label`` remains ``"Density"``, indicating that the histograms 
represent the density of observations within each bin. The histogram bars are 
colored using ``hist_color="brown"``, with a ``fill_alpha=0.60`` while the s
tatistical overlays enhance the interpretability of the data. The layout is 
configured with a single row and multiple columns ``(n_rows=1, n_cols=3)``, and 
the grid size is set to `15x5 inches`. This example highlights how to visualize 
central tendencies within the data using a histogram that prominently displays 
the mean and median.

.. code-block:: python

    from eda_toolkit import kde_distributions

    vars_of_interest = [
        "age",
        "education-num",
        "hours-per-week",
    ]

    kde_distributions(
        df=df,
        n_rows=1,
        n_cols=3,
        grid_figsize=(14, 4), 
        text_wrap=50,
        hist_color="brown",
        bbox_inches="tight",
        vars_of_interest=vars_of_interest,
        y_axis_label="Density",
        bins=10,
        fill_alpha=0.60,
        plot_type="hist",
        stat="Density",
        label_fontsize=16,  
        tick_fontsize=14,  
        plot_mean=True,
        plot_median=True,
        mean_color="blue",
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/density_hist_dist_mean_median.svg
   :alt: KDE Distributions - Histograms (Count)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>



Histogram Example - (Mean, Median, and Std. Deviation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, the ``kde_distributions()`` function is customized to generate 
a histogram that include mean, median, and 3 standard deviation lines. The 
``mean_color`` is set to ``"blue"`` and the median_color is set to ``"black"``, 
allowing for a clear distinction between these two central tendency measures. 
The function parameters are adjusted to ensure that both the mean and median lines 
are plotted ``(plot_mean=True, plot_median=True)``. The ``y_axis_label`` remains
``"Density"``, indicating that the histograms represent the density of observations 
within each bin. The histogram bars are colored using ``hist_color="brown"``, 
with a ``fill_alpha=0.40``, which adjusts the transparency of the fill color. 
Additionally, standard deviation bands are plotted using colors ``"purple"``, 
``"green"``, and ``"silver"`` for one, two, and three standard deviations, respectively.

The layout is configured with a single row and multiple columns ``(n_rows=1, n_cols=3)``, 
and the grid size is set to `15x5 inches`. This setup is particularly useful for 
visualizing the central tendencies within the data while also providing a clear 
view of the distribution and spread through the standard deviation bands. The 
configuration used in this example showcases how histograms can be enhanced with 
statistical overlays to provide deeper insights into the data.

.. note::

    You have the freedom to choose whether to plot the mean, median, and 
    standard deviation lines. You can display one, none, or all of these simultaneously.

.. code-block:: python

    from eda_toolkit import kde_distributions

    vars_of_interest = [
        "age",
    ]

    kde_distributions(
        df=df,
        figsize=(10, 6),
        text_wrap=50,
        hist_color="brown",
        bbox_inches="tight",
        vars_of_interest=vars_of_interest,
        y_axis_label="Density",
        bins=10,
        fill_alpha=0.40,
        plot_type="both",
        stat="Density",
        label_fontsize=16, 
        tick_fontsize=14,  
        plot_mean=True,
        plot_median=True,
        mean_color="blue",
        std_dev_levels=[
            1,
            2,
            3,
        ],
        std_color=[
            "purple",
            "green",
            "silver",
        ],
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/density_hist_dist_age.svg
   :alt: KDE Distributions - Histograms (Count)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Feature Scaling and Outliers
=============================

.. function:: data_doctor(df, feature_name, data_fraction=1, scale_conversion=None, scale_conversion_kws=None, apply_cutoff=False, lower_cutoff=None, upper_cutoff=None, show_plot=True, plot_type="all", figsize=(18, 6), xlim=None, kde_ylim=None, hist_ylim=None, box_violin_ylim=None, save_plot=False, image_path_png=None, image_path_svg=None, apply_as_new_col_to_df=False, kde_kws=None, hist_kws=None, box_violin_kws=None, box_violin="boxplot", label_fontsize=12, tick_fontsize=10, random_state=None)

    Analyze and transform a specific feature in a DataFrame, with options for
    scaling, applying cutoffs, and visualizing the results. This function also
    allows for the creation of a new column with the transformed data if
    specified. Plots can be saved in PNG or SVG format with filenames that
    incorporate the ``plot_type``, ``feature_name``, ``scale_conversion``, and
    ``cutoff`` if cutoffs are applied.

    :param df: The DataFrame containing the feature to analyze.
    :type df: pandas.DataFrame

    :param feature_name: The name of the feature (column) to analyze.
    :type feature_name: str

    :param data_fraction: Fraction of the data to analyze. Default is ``1`` (full dataset). Useful for large datasets where a sample can represent the population. If ``apply_as_new_col_to_df=True``, the full dataset is used (``data_fraction=1``).
    :type data_fraction: float, optional

    :param scale_conversion: Type of conversion to apply to the feature. Options include:
    
        - ``'abs'``: Absolute values
        - ``'log'``: Natural logarithm
        - ``'sqrt'``: Square root
        - ``'cbrt'``: Cube root
        - ``'reciprocal'``: Reciprocal transformation
        - ``'stdrz'``: Standardized (z-score)
        - ``'minmax'``: Min-Max scaling
        - ``'boxcox'``: Box-Cox transformation (positive values only; supports
          ``lmbda`` for specific lambda or ``alpha`` for confidence interval)
        - ``'robust'``: Robust scaling (median and IQR)
        - ``'maxabs'``: Max-abs scaling
        - ``'exp'``: Exponential transformation
        - ``'logit'``: Logit transformation (values between 0 and 1)
        - ``'arcsinh'``: Inverse hyperbolic sine
        - ``'square'``: Squaring the values
        - ``'power'``: Power transformation (Yeo-Johnson).
    :type scale_conversion: str, optional

    :param scale_conversion_kws: Additional keyword arguments to pass to the scaling functions, such as:
    
        - ``'alpha'`` for Box-Cox transformation (returns a confidence interval
          for lambda)
        - ``'lmbda'`` for a specific Box-Cox transformation value
        - ``'quantile_range'`` for robust scaling.
    :type scale_conversion_kws: dict, optional

    :param apply_cutoff: Whether to apply upper and/or lower cutoffs to the feature.
    :type apply_cutoff: bool, optional (default=False)

    :param lower_cutoff: Lower bound to apply if ``apply_cutoff=True``.
    :type lower_cutoff: float, optional

    :param upper_cutoff: Upper bound to apply if ``apply_cutoff=True``.
    :type upper_cutoff: float, optional

    :param show_plot: Whether to display plots of the transformed feature: KDE, histogram, and boxplot/violinplot.
    :type show_plot: bool, optional (default=True)

    :param plot_type: Specifies the type of plot(s) to produce. Options are:
    
        - ``'all'``: Generates KDE, histogram, and boxplot/violinplot.
        - ``'kde'``: KDE plot only.
        - ``'hist'``: Histogram plot only.
        - ``'box_violin'``: Boxplot or violin plot only (specified by
          ``box_violin``).

        If a list or tuple is provided (e.g., ``plot_type=["kde", "hist"]``),
        the specified plots are displayed in a single row with sufficient
        spacing. A ``ValueError`` is raised if an invalid plot type is included.
    :type plot_type: str, list, or tuple, optional (default="all")

    :param figsize: Specifies the figure size for the plots. This applies to all plot types, including single plots (when ``plot_type`` is set to "kde", "hist", or "box_violin") and multi-plot layout when ``plot_type`` is "all".
    :type figsize: tuple or list, optional (default=(18, 6))

    :param xlim: Limits for the x-axis in all plots, specified as ``(xmin, xmax)``.
    :type xlim: tuple or list, optional

    :param kde_ylim: Limits for the y-axis in the KDE plot, specified as ``(ymin, ymax)``.
    :type kde_ylim: tuple or list, optional

    :param hist_ylim: Limits for the y-axis in the histogram plot, specified as ``(ymin, ymax)``.
    :type hist_ylim: tuple or list, optional

    :param box_violin_ylim: Limits for the y-axis in the boxplot or violin plot, specified as ``(ymin, ymax)``.
    :type box_violin_ylim: tuple or list, optional

    :param save_plot: Whether to save the plots as PNG and/or SVG images. If ``True``, the user must specify at least one of ``image_path_png`` or ``image_path_svg``, otherwise a ``ValueError`` is raised.
    :type save_plot: bool, optional (default=False)

    :param image_path_png: Directory path to save the plot as a PNG file. Only used if ``save_plot=True``.
    :type image_path_png: str, optional

    :param image_path_svg: Directory path to save the plot as an SVG file. Only used if ``save_plot=True``.
    :type image_path_svg: str, optional

    :param apply_as_new_col_to_df: Whether to create a new column in the DataFrame with the transformed values. If ``True``, the new column name is generated based on the feature name and the transformation applied:
    
        - ``<feature_name>_<scale_conversion>``: If a transformation is applied.
        - ``<feature_name>_w_cutoff``: If only cutoffs are applied.
        
        For Box-Cox transformation, if ``alpha`` is specified, the confidence interval for lambda is displayed. If ``lmbda`` is specified, the lambda value is displayed.
    :type apply_as_new_col_to_df: bool, optional (default=False)

    :param kde_kws: Additional keyword arguments to pass to the KDE plot (``seaborn.kdeplot``).
    :type kde_kws: dict, optional

    :param hist_kws: Additional keyword arguments to pass to the histogram plot (``seaborn.histplot``).
    :type hist_kws: dict, optional

    :param box_violin_kws: Additional keyword arguments to pass to either boxplot or violinplot.
    :type box_violin_kws: dict, optional

    :param box_violin: Specifies whether to plot a ``boxplot`` or ``violinplot`` if ``plot_type`` is set to ``box_violin``.
    :type box_violin: str, optional (default="boxplot")

    :param label_fontsize: Font size for the axis labels and plot titles.
    :type label_fontsize: int, optional (default=12)

    :param tick_fontsize: Font size for the tick labels on both axes.
    :type tick_fontsize: int, optional (default=10)

    :param random_state: Seed for reproducibility when sampling the data.
    :type random_state: int, optional

    :returns: ``None`` 
        Displays the feature's descriptive statistics, quartile information,
        and outlier details. If a new column is created, confirms the addition to the DataFrame. For Box-Cox, either the lambda or its confidence interval is displayed.

    :raises ValueError: 
        - If an invalid ``scale_conversion`` is provided.
        - If Box-Cox transformation is applied to non-positive values.
        - If ``save_plot=True`` but neither ``image_path_png`` nor ``image_path_svg`` is provided.
        - If an invalid option is provided for ``box_violin``.
        - If an invalid option is provided for ``plot_type``.
        - If the length of transformed data does not match the original feature length.

    .. note::  
        
        When saving plots, the filename will include the ``feature_name``, ``scale_conversion``, each selected ``plot_type``, and, if cutoffs are applied, ``"_cutoff"``. For example, if ``feature_name`` is ``"age"``, ``scale_conversion`` is ``"boxcox"``, and ``plot_type`` is ``"kde"``, with cutoffs applied, the filename will be: ``age_boxcox_kde_cutoff.png`` or ``age_boxcox_kde_cutoff.svg``.


Available Scale Conversions
-----------------------------

The ``scale_conversion`` parameter accepts several options for data scaling, providing flexibility in how you preprocess your data. Each option addresses specific transformation needs, such as normalizing data, stabilizing variance, or adjusting data ranges. Below is the exhaustive list of available scale conversions:

- ``'abs'``: Takes the absolute values of the data, removing any negative signs.
- ``'log'``: Applies the natural logarithm to the data, useful for compressing large ranges and reducing skewness.
- ``'sqrt'``: Applies the square root transformation, often used to stabilize variance.
- ``'cbrt'``: Takes the cube root of the data, which can be useful for transforming both positive and negative values symmetrically.
- ``'stdrz'``: Standardizes the data to have a mean of 0 and a standard deviation of 1, also known as z-score normalization.
- ``'minmax'``: Rescales the data to a specified range, defaulting to [0, 1], ensuring that all values fall within this range.
- ``'boxcox'``: Applies the Box-Cox transformation to stabilize variance and make the data more normally distributed. Only works with positive values and supports passing ``lmbda`` or ``alpha`` for flexibility.
- ``'robust'``: Scales the data based on percentiles (such as the interquartile range), which reduces the influence of outliers.
- ``'maxabs'``: Scales the data by dividing it by its maximum absolute value, preserving the sign of the data while constraining it to the range [-1, 1].
- ``'reciprocal'``: Transforms the data by taking the reciprocal (1/x), which is useful when handling values that are far from zero.
- ``'exp'``: Applies the exponential function to the data, which is useful for modeling exponential growth or increasing the impact of large values.
- ``'logit'``: Applies the logit transformation to data, which is only valid for values between 0 and 1. This is typically used in logistic regression models.
- ``'arcsinh'``: Applies the inverse hyperbolic sine transformation, which is similar to the logarithm but can handle both positive and negative values.
- ``'square'``: Squares the values of the data, effectively emphasizing larger values while downplaying smaller ones.
- ``'power'``: Applies the power transformation (Yeo-Johnson), which is similar to Box-Cox but works for both positive and negative values.

``boxcox`` is just one of the many options available for transforming data in the ``data_doctor`` function, providing versatility to handle different scaling needs.

.. _Box_Cox_Example_1:

Box-Cox Transformation Example 1
----------------------------------

In this example from the US Census dataset [1]_, we demonstrate the usage of the ``data_doctor`` 
function to apply a **Box-Cox transformation** to the ``age`` column in a DataFrame. 
The ``data_doctor`` function provides a flexible way to preprocess data by applying 
various scaling techniques. In this case, we apply the Box-Cox transformation **without any tuning** 
of the ``alpha`` or ``lambda`` parameters, allowing the function to handle the transformation in a 
barebones approach. You can also choose other scaling conversions from the list of available 
options (such as ``'minmax'``, ``'standard'``, ``'robust'``, etc.), depending on your needs.


.. code-block:: python

   from eda_toolkit import data_doctor

   data_doctor(
       df=df,
       feature_name="age",
       data_fraction=0.6,
       scale_conversion="boxcox",
       apply_cutoff=False,
       lower_cutoff=None,
       upper_cutoff=None,
       show_plot=True,
       apply_as_new_col_to_df=True,
       random_state=111,
   )

.. code-block:: text

                DATA DOCTOR SUMMARY REPORT             
    +------------------------------+--------------------+
    | Feature                      | age                |
    +------------------------------+--------------------+
    | Statistic                    | Value              |
    +------------------------------+--------------------+
    | Min                          |             3.6664 |
    | Max                          |             6.8409 |
    | Mean                         |             5.0163 |
    | Median                       |             5.0333 |
    | Std Dev                      |             0.6761 |
    +------------------------------+--------------------+
    | Quartile                     | Value              |
    +------------------------------+--------------------+
    | Q1 (25%)                     |             4.5219 |
    | Q2 (50% = Median)            |             5.0333 |
    | Q3 (75%)                     |             5.5338 |
    | IQR                          |             1.0119 |
    +------------------------------+--------------------+
    | Outlier Bound                | Value              |
    +------------------------------+--------------------+
    | Lower Bound                  |             3.0040 |
    | Upper Bound                  |             7.0517 |
    +------------------------------+--------------------+

    New Column Name: age_boxcox
    Box-Cox Lambda: 0.1748

.. raw:: html

   <div class="no-click">

.. image:: ../assets/age_boxcox_kde_hist_boxplot.svg
   :alt: Box-Cox Transformation W/ Data Doctor
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>

.. code-block:: python

    df.head()

.. raw:: html

    <style type="text/css">
    .tg-wrap {
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    .tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:monospace, sans-serif !important;font-size:11px !important;
      overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:monospace, sans-serif !important;font-size:11px !important;
      font-weight:normal;overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg .tg-zv4m{border-color:#ffffff;text-align:left;vertical-align:top}
    .tg .tg-8jgo{border-color:#ffffff;text-align:center;vertical-align:top}
    .tg .tg-aw21{border-color:#ffffff;font-weight:bold;text-align:center;vertical-align:top}
    .tg .tg-lightpink{background-color:#FFCCCC; border-width: 0px;} /* Remove borders and apply solid pink color */
    </style>
    <div class="tg-wrap">
    <table class="tg">
      <thead>
        <tr>
          <th class="tg-zv4m"></th>
          <th class="tg-aw21">age</th>
          <th class="tg-aw21">workclass</th>
          <th class="tg-aw21">education</th>
          <th class="tg-aw21">education-num</th>
          <th class="tg-aw21">marital-status</th>
          <th class="tg-aw21">occupation</th>
          <th class="tg-aw21">relationship</th>
          <th class="tg-aw21 tg-lightpink">age_boxcox</th> <!-- Highlighted column -->
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="tg-aw21">census_id</td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo tg-lightpink"></td>
        </tr>
        <tr>
          <td class="tg-zv4m">582248222</td>
          <td class="tg-8jgo">39</td>
          <td class="tg-8jgo">State-gov</td>
          <td class="tg-8jgo">Bachelors</td>
          <td class="tg-8jgo">13</td>
          <td class="tg-8jgo">Never-married</td>
          <td class="tg-8jgo">Adm-clerical</td>
          <td class="tg-8jgo">Not-in-family</td>
          <td class="tg-8jgo tg-lightpink">5.180807</td>
        </tr>
        <tr>
          <td class="tg-zv4m">561810758</td>
          <td class="tg-8jgo">50</td>
          <td class="tg-8jgo">Self-emp-not-inc</td>
          <td class="tg-8jgo">Bachelors</td>
          <td class="tg-8jgo">13</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Exec-managerial</td>
          <td class="tg-8jgo">Husband</td>
          <td class="tg-8jgo tg-lightpink">5.912323</td>
        </tr>
        <tr>
          <td class="tg-zv4m">598098459</td>
          <td class="tg-8jgo">38</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">HS-grad</td>
          <td class="tg-8jgo">9</td>
          <td class="tg-8jgo">Divorced</td>
          <td class="tg-8jgo">Handlers-cleaners</td>
          <td class="tg-8jgo">Not-in-family</td>
          <td class="tg-8jgo tg-lightpink">5.227960</td>
        </tr>
        <tr>
          <td class="tg-zv4m">776705221</td>
          <td class="tg-8jgo">53</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">11th</td>
          <td class="tg-8jgo">7</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Handlers-cleaners</td>
          <td class="tg-8jgo">Husband</td>
          <td class="tg-8jgo tg-lightpink">6.389562</td>
        </tr>
        <tr>
          <td class="tg-zv4m">479262902</td>
          <td class="tg-8jgo">28</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">Bachelors</td>
          <td class="tg-8jgo">13</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Prof-specialty</td>
          <td class="tg-8jgo">Wife</td>
          <td class="tg-8jgo tg-lightpink">3.850675</td>
        </tr>
      </tbody>
    </table>
    </div>

\

.. note::

    Notice that the :ref:`unique identifiers <Adding_Unique_Identifiers>` function was also applied on the dataframe to generate randomized census IDs for the rows of the data.

**Explanation**

- ``df=df``: We are passing ``df`` as the input DataFrame.
- ``feature_name="age"``: The feature we are transforming is ``age``.
- ``data_fraction=1``: We are using 100% of the data in the ``age`` column. You can adjust this if you want to perform the operation on a subset of the data.
- ``scale_conversion="boxcox"``: This parameter defines the type of scaling we want to apply. In this case, we are using the Box-Cox transformation. You can change ``boxcox`` to any supported scale conversion method.
- ``apply_cutoff=False``: We are not applying any outlier cutoff in this example.
- ``lower_cutoff=None`` and ``upper_cutoff=None``: These are left as ``None`` since we are not applying outlier cutoffs in this case.
- ``show_plot=True``: This option will generate a plot to visualize the distribution of the ``age`` column before and after the transformation.
- ``apply_as_new_col_to_df=True``: This tells the function to apply the transformation and create a new column in the DataFrame. The new column will be named ``age_boxcox``, where ``"boxcox"`` indicates the type of transformation applied.

1. **Box-Cox Transformation**: This transformation normalizes the data by making the distribution more Gaussian-like, which can be beneficial for certain statistical models.
   
2. **No Outlier Handling**: In this example, we are not applying any cutoffs to remove or modify outliers. This means the function will process the entire range of values in the ``age`` column without making adjustments for extreme values.

3. **New Column Creation**: By setting ``apply_as_new_col_to_df=True``, a new column named ``age_boxcox`` will be created in the ``df`` DataFrame, where the transformed values will be stored. This allows us to keep the original ``age`` column intact while adding the transformed data as a new feature.

4. The ``show_plot=True`` parameter will generate a plot that visualizes the distribution of the original ``age`` data alongside the transformed ``age_boxcox`` data. This can help you assess how the Box-Cox transformation has affected the data distribution.


.. _Box_Cox_Example_2:

Box-Cox Transformation Example 2
----------------------------------

In this second example from the US Census dataset [1]_, we apply the Box-Cox 
transformation to the ``age`` column in a DataFrame, but this time with custom 
keyword arguments passed through the ``scale_conversion_kws``. Specifically, we 
provide an ``alpha`` value of `0.8`, :ref:`influencing the confidence interval for the 
transformation <Confidence_Intervals_for_Lambda>`. Additionally, we customize the 
visual appearance of the plots by specifying keyword arguments for the violinplot, 
KDE, and histogram plots. These customizations allow for greater control over the 
visual output.


.. code-block:: python 

    from eda_toolkit import data_doctor

    data_doctor(
        df=df,
        feature_name="age",
        data_fraction=1,
        scale_conversion="boxcox",
        apply_cutoff=False,
        lower_cutoff=None,
        upper_cutoff=None,
        show_plot=True,
        apply_as_new_col_to_df=True,
        scale_conversion_kws={"alpha": 0.8},
        box_violin="violinplot",
        box_violin_kws={"color": "lightblue"},
        kde_kws={"fill": True, "color": "blue"},
        hist_kws={"color": "green"},
        random_state=111,
    )

.. code-block:: text

                DATA DOCTOR SUMMARY REPORT             
    +------------------------------+--------------------+
    | Feature                      | age                |
    +------------------------------+--------------------+
    | Statistic                    | Value              |
    +------------------------------+--------------------+
    | Min                          |             3.6664 |
    | Max                          |             6.8409 |
    | Mean                         |             5.0163 |
    | Median                       |             5.0333 |
    | Std Dev                      |             0.6761 |
    +------------------------------+--------------------+
    | Quartile                     | Value              |
    +------------------------------+--------------------+
    | Q1 (25%)                     |             4.5219 |
    | Q2 (50% = Median)            |             5.0333 |
    | Q3 (75%)                     |             5.5338 |
    | IQR                          |             1.0119 |
    +------------------------------+--------------------+
    | Outlier Bound                | Value              |
    +------------------------------+--------------------+
    | Lower Bound                  |             3.0040 |
    | Upper Bound                  |             7.0517 |
    +------------------------------+--------------------+

    New Column Name: age_boxcox
    Box-Cox C.I. for Lambda: (0.1717, 0.1779)


.. raw:: html

   <div class="no-click">

.. image:: ../assets/age_boxcox_kde_hist_violinplot.svg
   :alt: Box-Cox Transformation W/ Data Doctor
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html

   <div style="height: 50px;"></div>


.. note::

    Note that this example specifies The theoretical overview section provides a 
    detailed framework for a :ref:`Box-Cox transformation <Box_Cox_Transformation>`.  

.. code-block:: python

    df.head()

.. raw:: html

    <style type="text/css">
    .tg-wrap {
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    .tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:monospace, sans-serif !important;font-size:11px !important;
      overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:monospace, sans-serif !important;font-size:11px !important;
      font-weight:normal;overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg .tg-zv4m{border-color:#ffffff;text-align:left;vertical-align:top}
    .tg .tg-8jgo{border-color:#ffffff;text-align:center;vertical-align:top}
    .tg .tg-aw21{border-color:#ffffff;font-weight:bold;text-align:center;vertical-align:top}
    .tg .tg-lightpink{background-color:#FFCCCC; border-width: 0px;} /* Remove borders and apply solid pink color */
    </style>
    <div class="tg-wrap">
    <table class="tg">
      <thead>
        <tr>
          <th class="tg-zv4m"></th>
          <th class="tg-aw21">age</th>
          <th class="tg-aw21">workclass</th>
          <th class="tg-aw21">education</th>
          <th class="tg-aw21">education-num</th>
          <th class="tg-aw21">marital-status</th>
          <th class="tg-aw21">occupation</th>
          <th class="tg-aw21">relationship</th>
          <th class="tg-aw21 tg-lightpink">age_boxcox</th> <!-- Highlighted column -->
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="tg-aw21">census_id</td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo tg-lightpink"></td>
        </tr>
        <tr>
          <td class="tg-zv4m">582248222</td>
          <td class="tg-8jgo">39</td>
          <td class="tg-8jgo">State-gov</td>
          <td class="tg-8jgo">Bachelors</td>
          <td class="tg-8jgo">13</td>
          <td class="tg-8jgo">Never-married</td>
          <td class="tg-8jgo">Adm-clerical</td>
          <td class="tg-8jgo">Not-in-family</td>
          <td class="tg-8jgo tg-lightpink">3.936876</td>
        </tr>
        <tr>
          <td class="tg-zv4m">561810758</td>
          <td class="tg-8jgo">50</td>
          <td class="tg-8jgo">Self-emp-not-inc</td>
          <td class="tg-8jgo">Bachelors</td>
          <td class="tg-8jgo">13</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Exec-managerial</td>
          <td class="tg-8jgo">Husband</td>
          <td class="tg-8jgo tg-lightpink">4.019590</td>
        </tr>
        <tr>
          <td class="tg-zv4m">598098459</td>
          <td class="tg-8jgo">38</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">HS-grad</td>
          <td class="tg-8jgo">9</td>
          <td class="tg-8jgo">Divorced</td>
          <td class="tg-8jgo">Handlers-cleaners</td>
          <td class="tg-8jgo">Not-in-family</td>
          <td class="tg-8jgo tg-lightpink">4.521908</td>
        </tr>
        <tr>
          <td class="tg-zv4m">776705221</td>
          <td class="tg-8jgo">53</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">11th</td>
          <td class="tg-8jgo">7</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Handlers-cleaners</td>
          <td class="tg-8jgo">Husband</td>
          <td class="tg-8jgo tg-lightpink">5.033257</td>
        </tr>
        <tr>
          <td class="tg-zv4m">479262902</td>
          <td class="tg-8jgo">28</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">Bachelors</td>
          <td class="tg-8jgo">13</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Prof-specialty</td>
          <td class="tg-8jgo">Wife</td>
          <td class="tg-8jgo tg-lightpink">5.614411	</td>
        </tr>
      </tbody>
    </table>
    </div>

\

In this example, you can see how the ``data_doctor`` function supports further 
flexibility with customizable plot aesthetics and scaling techniques. The 
Box-Cox transformation is still applied without any tuning of the ``lambda`` parameter, 
while the ``alpha`` value provides a confidence interval for the resulting transformation:

.. code-block:: text

    Box-Cox C.I. for Lambda: (0.1717, 0.1779)

This allows for tailored visualizations with consistent styling across multiple plot types.

Some of the keyword arguments, such as those passed in ``box_violin_kws``, are 
specific to Python version 3.7. For example, in this version, we remove the fill 
color from the boxplot using ``boxprops``. 

.. code-block:: python

    box_violin_kws={
        "boxprops": dict(facecolor="none", edgecolor="blue")
    },

In later Python versions (e.g., 3.11), 
this can be done more easily with ``fill=True``. Therefore, it is important to pass 
any desired keyword arguments based on the correct version of Python you're using.

**Explanation**

- ``df=df``: We are passing ``df`` as the input DataFrame.
- ``feature_name="age"``: The feature we are transforming is ``age``.
- ``data_fraction=1``: We are using 100% of the data in the ``age`` column. You can adjust this if you want to perform the operation on a subset of the data.
- ``scale_conversion="boxcox"``: This parameter defines the type of scaling we want to apply. In this case, we are using the Box-Cox transformation.
- ``apply_cutoff=False``: We are not applying any outlier cutoff in this example.
- ``lower_cutoff=None`` and ``upper_cutoff=None``: These are left as ``None`` since we are not applying outlier cutoffs in this case.
- ``show_plot=True``: This option will generate a plot to visualize the distribution of the ``age`` column before and after the transformation.
- ``apply_as_new_col_to_df=True``: This tells the function to apply the transformation and create a new column in the DataFrame. The new column will be named ``age_boxcox_alpha`` to indicate that an alpha parameter was used in the transformation.
- ``scale_conversion_kws={"alpha":0.8}``: The ``alpha`` keyword argument specifies the confidence interval for the Box-Cox transformation's lambda value, ensuring a confidence interval is returned instead of a single lambda value.
- ``box_violin_kws={"boxprops": dict(facecolor='none', edgecolor="blue")}``: This keyword argument customizes the appearance of the boxplot by removing the fill color and setting the edge color to blue. This syntax is specific to Python 3.7. In later versions (i.e., 3.11+), the ``fill=True`` argument can be used to control this behavior.
- ``kde_kws={"fill":True, "color":"blue"}``: This fills the area under the KDE plot with a blue color, enhancing the plot's visual presentation.
- ``hist_kws={"color":"blue"}``: This colors the histogram bars in blue for visual consistency across plots.
- ``image_path_svg=image_path_svg``: This parameter specifies the path where the resulting plot will be saved as an SVG file.
- ``save_plot=True``: This tells the function to save the plot, and since an image path is provided, the plot will be saved as an SVG file.

1. **Box-Cox Transformation with Confidence Interval**: In this example, we use the Box-Cox transformation with the ``alpha`` parameter set to 0.8, which returns a confidence interval for the lambda value rather than a single value.
   
2. **No Outlier Handling**: Similar to Example 1, no outliers are handled in this transformation.
   
3. **New Column Creation**: The transformed data is added to the DataFrame in a new column named ``age_boxcox_alpha``, where "alpha" indicates the confidence interval applied in the Box-Cox transformation.

4. **Custom Plot Visuals**: The KDE, histogram, and boxplot are customized with blue colors, and specific keyword arguments are provided for the boxplot appearance based on Python version. These changes allow for finer control over the visual aesthetics of the resulting plots.

5. **Plot Saving**: The ``save_plot`` parameter is set to ``True``, and the plot will be saved as an SVG file at the specified location.


Data Fraction Usage
----------------------

In the **Box-Cox transformation** examples, you may notice a difference in the values for ``data_fraction``:

- In :ref:`Box-Cox Example 1 <Box_Cox_Example_1>`, we set ``data_fraction=0.6``.

- In :ref:`Box-Cox Example 2 <Box_Cox_Example_2>`, we used the full data with ``data_fraction=1``.

Despite using a ``data_fraction`` of `0.6` in Example 1, the function still processed 
the entire dataset. The purpose of the ``data_fraction`` parameter is to allow 
users to select a smaller subset of the data for sampling and transformation while 
ensuring the final operation is applied to the full scope of data.

This behavior is intentional, as it serves to:

1. **Ensure Reproducibility**: By using a consistent ``random_state``, the sampled 
subset can reliably represent the dataset, regardless of ``data_fraction``.

2. **Preserve Sampling Assumptions**: Applying the desired operation (e.g., transformations) 
on the full data aligns the sample with the larger population and allows a seamless projection 
of the sample properties to the entire dataset.

Thus, while ``data_fraction`` provides a way to adjust the percentage of data 
used for sampling, the function will always apply the transformation across the 
full dataset, balancing performance efficiency with statistical integrity.


Retaining a Sample for Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To sample the exact subset used in the ``data_fraction=0.6`` calculation, you 
can directly sample from the DataFrame with a consistent random state for 
reproducibility. This method allows you to work with a representative subset of 
the data while preserving the original distribution characteristics.

To sample 60% of the data using the exact logic of the ``data_doctor`` function, 
use the following code:

.. code-block:: python

    sampled_df = df.sample(frac=0.6, random_state=111)

The ``random_state`` parameter ensures that the sampled data remains consistent 
across runs. After creating this subset, you can apply the ``data_doctor`` 
function to ``sampled_df`` as shown below to perform the Box-Cox transformation 
on the ``age`` column:

.. code-block:: python

    from eda_toolkit import data_doctor

    data_doctor(
        df=sampled_df,
        feature_name="age",
        data_fraction=1,
        scale_conversion="boxcox",
        apply_cutoff=False,
        lower_cutoff=None,
        upper_cutoff=None,
        show_plot=True,
        apply_as_new_col_to_df=True,
        random_state=111,
    )

By setting ``data_fraction=1`` within the ``data_doctor`` function, you ensure 
that it operates on the entire ``sampled_df``, which now consists of the selected 
60% subset. To confirm that the sampled data is indeed 60% of the original 
DataFrame, you can print the shape of ``sampled_df`` as follows:

.. code-block:: python

    print(
        f"The sampled dataframe has {sampled_df.shape[0]} rows and {sampled_df.shape[1]} columns."
    )

.. code-block:: bash

    The sampled dataframe has 29305 rows and 16 columns.


We can also inspect the first five rows of the ``sampled_df`` dataframe below:

.. raw:: html

    <style type="text/css">
    .tg-wrap {
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    .tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:monospace, sans-serif !important;font-size:11px !important;
      overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:monospace, sans-serif !important;font-size:11px !important;
      font-weight:normal;overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg .tg-zv4m{border-color:#ffffff;text-align:left;vertical-align:top}
    .tg .tg-8jgo{border-color:#ffffff;text-align:center;vertical-align:top}
    .tg .tg-aw21{border-color:#ffffff;font-weight:bold;text-align:center;vertical-align:top}
    .tg .tg-lightpink{background-color:#FFCCCC; border-width: 0px;} /* Remove borders and apply solid pink color */
    </style>
    <div class="tg-wrap">
    <table class="tg">
      <thead>
        <tr>
          <th class="tg-zv4m"></th>
          <th class="tg-aw21">age</th>
          <th class="tg-aw21">workclass</th>
          <th class="tg-aw21">education</th>
          <th class="tg-aw21">education-num</th>
          <th class="tg-aw21">marital-status</th>
          <th class="tg-aw21">occupation</th>
          <th class="tg-aw21">relationship</th>
          <th class="tg-aw21 tg-lightpink">age_boxcox</th> <!-- Highlighted column -->
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="tg-aw21">census_id</td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo tg-lightpink"></td>
        </tr>
        <tr>
          <td class="tg-zv4m">408117383</td>
          <td class="tg-8jgo">40</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">Some-college</td>
          <td class="tg-8jgo">10</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Machine-op-inspct</td>
          <td class="tg-8jgo">Husband</td>
          <td class="tg-8jgo tg-lightpink">4.355015</td>
        </tr>
        <tr>
          <td class="tg-zv4m">669717925</td>
          <td class="tg-8jgo">58</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">HS-grad</td>
          <td class="tg-8jgo">9</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Exec-managerial</td>
          <td class="tg-8jgo">Husband</td>
          <td class="tg-8jgo tg-lightpink">5.086108</td>
        </tr>
        <tr>
          <td class="tg-zv4m">399428377</td>
          <td class="tg-8jgo">41</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">HS-grad</td>
          <td class="tg-8jgo">9</td>
          <td class="tg-8jgo">Separated</td>
          <td class="tg-8jgo">Machine-op-inspct</td>
          <td class="tg-8jgo">Not-in-family</td>
          <td class="tg-8jgo tg-lightpink">5.037743</td>
        </tr>
        <tr>
          <td class="tg-zv4m">961427355</td>
          <td class="tg-8jgo">73</td>
          <td class="tg-8jgo">NaN</td>
          <td class="tg-8jgo">Some-college</td>
          <td class="tg-8jgo">10</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">NaN</td>
          <td class="tg-8jgo">Husband</td>
          <td class="tg-8jgo tg-lightpink">4.216561</td>
        </tr>
        <tr>
          <td class="tg-zv4m">458295720</td>
          <td class="tg-8jgo">19</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">HS-grad</td>
          <td class="tg-8jgo">9</td>
          <td class="tg-8jgo">Never-married</td>
          <td class="tg-8jgo">Farming-fishing</td>
          <td class="tg-8jgo">Not-in-family</td>
          <td class="tg-8jgo tg-lightpink">5.520438</td>
        </tr>
      </tbody>
    </table>
    </div>


\


Logit Transformation Example
-------------------------------

In this example, we demonstrate the usage of the ``data_doctor`` function to 
apply a **logit transformation** to a feature in a DataFrame. The **logit transformation** 
is used when dealing with data bounded between 0 and 1, as it maps values within 
this range to an unbounded scale in log-odds terms, making it particularly useful 
in fields such as logistic regression.

.. note::

    The ``data_doctor`` function provides a range of scaling options, and in this case, 
    we use the **logit transformation** to illustrate how the transformation is applied. 
    However, it’s important to note that if the feature contains values outside the (0, 1) 
    range, the function will raise a ``ValueError``. This is because the :ref:`logit function 
    is undefined for values less than or equal to 0 and greater than or equal to 1 <Logit_Assumptions>`. 

.. code-block:: python

    from eda_toolkit import data_doctor

    data_doctor(
        df=df,
        feature_name="age",
        data_fraction=1,
        scale_conversion="logit",
        apply_cutoff=False,
        lower_cutoff=None,
        upper_cutoff=None,
        show_plot=True,
        apply_as_new_col_to_df=True,
        random_state=111,
    )

.. error::

    ``ValueError: Logit transformation requires values to be between 0 and 1. Consider using a scaling method such as min-max scaling first.``

If you attempt to apply this transformation to data outside the (0, 1) range, 
such as an unscaled numerical feature, the function will halt and display an 
error message advising you to use an appropriate scaling method first.

If you encounter this error, it is recommended to first scale your data using a 
method like **min-max scaling** to bring it within the (0, 1) range before 
applying the logit transformation.


In this example:

- ``df=df``: Specifies the DataFrame containing the feature.
- ``feature_name="feature_proportion"``: The feature we are transforming should be bounded between 0 and 1.
- ``scale_conversion="logit"``: Sets the transformation to logit. Ensure that ``feature_proportion`` values are within (0, 1) before applying.
- ``show_plot=True``: Generates a plot of the transformed feature.


Plain Outliers Example
--------------------------------

Observed Outliers Sans Cutoffs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we examine the final weight (``fnlwgt``) feature from the US Census 
dataset [1]_, focusing on detecting outliers without applying any scaling 
transformations. The ``data_doctor`` function is used with minimal configuration 
to visualize where outliers are present in the raw data.

By enabling ``apply_cutoff=True`` and selecting ``plot_type=["box_violin", "hist"]``, 
we can clearly identify outliers both visually and numerically. This basic setup 
highlights the outliers without altering the data distribution, making it easy 
to see extreme values that could affect further analysis.

The following code demonstrates this:

.. code-block:: python

    from eda_toolkit import data_doctor

    data_doctor(
        df=df,
        feature_name="fnlwgt",
        data_fraction=0.6,
        plot_type=["box_violin", "hist"],
        hist_kws={"color": "gray"},
        figsize=(8, 4),
        image_path_svg=image_path_svg,
        save_plot=True,
        random_state=111,
    )

.. code-block:: text

                DATA DOCTOR SUMMARY REPORT             
    +------------------------------+--------------------+
    | Feature                      | fnlwgt             |
    +------------------------------+--------------------+
    | Statistic                    | Value              |
    +------------------------------+--------------------+
    | Min                          |        12,285.0000 |
    | Max                          |     1,484,705.0000 |
    | Mean                         |       189,181.3719 |
    | Median                       |       177,955.0000 |
    | Std Dev                      |       105,417.5713 |
    +------------------------------+--------------------+
    | Quartile                     | Value              |
    +------------------------------+--------------------+
    | Q1 (25%)                     |       117,292.0000 |
    | Q2 (50% = Median)            |       177,955.0000 |
    | Q3 (75%)                     |       236,769.0000 |
    | IQR                          |       119,477.0000 |
    +------------------------------+--------------------+
    | Outlier Bound                | Value              |
    +------------------------------+--------------------+
    | Lower Bound                  |       -61,923.5000 |
    | Upper Bound                  |       415,984.5000 |
    +------------------------------+--------------------+

.. raw:: html

   <div class="no-click">

.. image:: ../assets/fnlwgt_None_boxplot_hist.svg
   :alt: Outlier Detection W/ Data Doctor
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html

   <div style="height: 20px;"></div>

In this visualization, the boxplot and histogram display outliers prominently, 
showing you exactly where the extreme values lie. This setup serves as a baseline 
view of the raw data, making it useful for assessing the initial distribution 
before any scaling or transformation is applied.


Treated Outliers With Cutoffs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this scenario, we address the extreme values observed in the ``fnlwgt`` feature 
by applying a visual cutoff based on the distribution seen in the previous example. 
Here, we set an approximate upper cutoff of **400,000** to limit the impact of outliers 
without any additional scaling or transformation. By using ``apply_cutoff=True`` along 
with ``upper_cutoff=400000``, we effectively cap the extreme values.

This example also demonstrates how you can further customize the visualization by 
specifying additional histogram keyword arguments with ``hist_kws``. Here, we use 
``bins=20`` to adjust the bin size, creating a smoother view of the feature's 
distribution within the cutoff limits.

In the resulting visualization, you will see that the boxplot and histogram have a 
controlled range due to the applied upper cutoff, limiting the influence of extreme 
outliers on the visual representation. This treatment provides a clearer view of the 
primary distribution, allowing for a more focused analysis on the bulk of the data 
without outliers distorting the scale.

The following code demonstrates this configuration:

.. code-block:: python

    from eda_toolkit import data_doctor

    data_doctor(
        df=df,
        feature_name="fnlwgt",
        data_fraction=0.6,
        apply_as_new_col_to_df=True,
        apply_cutoff=True,
        upper_cutoff=400000,
        plot_type=["box_violin", "hist"],
        hist_kws={"color": "gray", "bins": 20},
        figsize=(8, 4),
        image_path_svg=image_path_svg,
        save_plot=True,
        random_state=111,
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/fnlwgt_None_boxplot_hist_cutoff.svg
   :alt: Outlier Detection W/ Data Doctor
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html

   <div style="height: 20px;"></div>

.. raw:: html

    <style type="text/css">
    .tg-wrap {
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    .tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:monospace, sans-serif !important;font-size:11px !important;
      overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:monospace, sans-serif !important;font-size:11px !important;
      font-weight:normal;overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg .tg-zv4m{border-color:#ffffff;text-align:left;vertical-align:top}
    .tg .tg-8jgo{border-color:#ffffff;text-align:center;vertical-align:top}
    .tg .tg-aw21{border-color:#ffffff;font-weight:bold;text-align:center;vertical-align:top}
    .tg .tg-lightpink{background-color:#FFCCCC; border-width: 0px;} /* Remove borders and apply solid pink color */
    </style>
    <div class="tg-wrap">
    <table class="tg">
      <thead>
        <tr>
          <th class="tg-zv4m"></th>
          <th class="tg-aw21">age</th>
          <th class="tg-aw21">workclass</th>
          <th class="tg-aw21">fnlwgt</th>
          <th class="tg-aw21">education</th>
          <th class="tg-aw21">marital-status</th>
          <th class="tg-aw21">occupation</th>
          <th class="tg-aw21">relationship</th>
          <th class="tg-aw21 tg-lightpink">fnlwgt_w_cutoff</th> 
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="tg-aw21">census_id</td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo"></td>
          <td class="tg-8jgo tg-lightpink"></td>
        </tr>
        <tr>
          <td class="tg-zv4m">582248222</td>
          <td class="tg-8jgo">39</td>
          <td class="tg-8jgo">State-gov	</td>
          <td class="tg-8jgo">77516</td>
          <td class="tg-8jgo">Bachelors</td>
          <td class="tg-8jgo">Never-married</td>
          <td class="tg-8jgo">Adm-clerical</td>
          <td class="tg-8jgo">Not-in-family</td>
          <td class="tg-8jgo tg-lightpink">132222</td> <!-- New cell with data -->
        </tr>
        <tr>
          <td class="tg-zv4m">561810758</td>
          <td class="tg-8jgo">50</td>
          <td class="tg-8jgo">Self-emp-not-inc</td>
          <td class="tg-8jgo">83311</td>
          <td class="tg-8jgo">Bachelors</td>
           <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Exec-managerial</td>
          <td class="tg-8jgo">Husband</td>
           <td class="tg-8jgo tg-lightpink">68624</td> <!-- New cell with data -->
        </tr>
        <tr>
          <td class="tg-zv4m">598098459</td>
          <td class="tg-8jgo">38</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">215646</td>
          <td class="tg-8jgo">HS-grad</td>
          <td class="tg-8jgo">Divorced</td>
          <td class="tg-8jgo">Handlers-cleaners</td>
          <td class="tg-8jgo">Not-in-family</td>
          <td class="tg-8jgo tg-lightpink">161880</td> 
        </tr>
        <tr>
          <td class="tg-zv4m">776705221</td>
          <td class="tg-8jgo">53</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">234721</td>
          <td class="tg-8jgo">11th</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Handlers-cleaners</td>
          <td class="tg-8jgo">Husband</td>
          <td class="tg-8jgo tg-lightpink">73402</td> 
        </tr>
        <tr>
          <td class="tg-zv4m">479262902</td>
          <td class="tg-8jgo">28</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">338409</td>
          <td class="tg-8jgo">Bachelors</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Prof-specialty</td>
          <td class="tg-8jgo">Wife</td>
          <td class="tg-8jgo tg-lightpink">97261</td> <!-- New cell with data -->
        </tr>
        <tr>
        </tr>
      </tbody>
    </table>
    </div>

\

RobustScaler Outliers Examples
--------------------------------

In this example from the US Census dataset [1]_, we apply the :ref:`RobustScaler 
transformation <Robust_Scaler>` to the age column in a DataFrame to address potential outliers. 
The ``data_doctor`` function enables users to apply transformations with specific 
configurations via the ``scale_conversion_kws`` parameter, making it ideal for 
refining how outliers affect scaling.

For this example, we set the following custom keyword arguments:

- Disable centering: By setting ``with_centering=False``, the transformation scales based only on the range, without shifting the median to zero.
- Adjust quantile range: We specify a narrower ``quantile_range`` of (10.0, 90.0) to reduce the influence of extreme values on scaling.

The following code demonstrates this transformation:

.. code-block:: python

    from eda_toolkit import data_doctor

    data_doctor(
        df=df,
        feature_name='age',
        data_fraction=0.6,
        scale_conversion="robust",
        apply_as_new_col_to_df=True,
        scale_conversion_kws={
            "with_centering": False,  # Disable centering
            "quantile_range": (10.0, 90.0)  # Use a custom quantile range
        },
        random_state=111,
    )

.. code-block:: text

                 DATA DOCTOR SUMMARY REPORT             
    +------------------------------+--------------------+
    | Feature                      | age                |
    +------------------------------+--------------------+
    | Statistic                    | Value              |
    +------------------------------+--------------------+
    | Min                          | 0.4722             |
    | Max                          | 2.5000             |
    | Mean                         | 1.0724             |
    | Median                       | 1.0278             |
    | Std Dev                      | 0.3809             |
    +------------------------------+--------------------+
    | Quartile                     | Value              |
    +------------------------------+--------------------+
    | Q1 (25%)                     | 0.7778             |
    | Q2 (Median)                  | 1.0278             |
    | IQR                          | 0.5556             |
    | Q3 (75%)                     | 1.3333             |
    | Q4 (Max)                     | 2.5000             |
    +------------------------------+--------------------+
    | Outlier Bound                | Value              |
    +------------------------------+--------------------+
    | Lower Bound                  | -0.0556            |
    | Upper Bound                  | 2.1667             |
    +------------------------------+--------------------+

    New Column Name: age_robust


.. raw:: html

   <div class="no-click">

.. image:: ../assets/age_robust_boxplot_hist.svg
   :alt: Box-Cox Transformation W/ Data Doctor
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Stacked Crosstab Plots
=======================

**Generate stacked or regular bar plots and crosstabs for specified columns in a DataFrame.**

The ``stacked_crosstab_plot`` function is a powerful tool for visualizing categorical data relationships through stacked bar plots and contingency tables (crosstabs). It supports extensive customization options, including plot appearance, color schemes, and saving output in multiple formats. Users can choose between regular or normalized plots and control whether the function returns the generated crosstabs as a dictionary.

.. function:: stacked_crosstab_plot(df, col, func_col, legend_labels_list, title, kind="bar", width=0.9, rot=0, custom_order=None, image_path_png=None, image_path_svg=None, save_formats=None, color=None, output="both", return_dict=False, x=None, y=None, p=None, file_prefix=None, logscale=False, plot_type="both", show_legend=True, label_fontsize=12, tick_fontsize=10, text_wrap=50, remove_stacks=False, xlim=None, ylim=None)

    :param df: The DataFrame containing the data to plot.
    :type df: pandas.DataFrame
    :param col: The name of the column in the DataFrame to be analyzed.
    :type col: str
    :param func_col: List of columns in the DataFrame to generate the crosstabs and stack the bars in the plot.
    :type func_col: list of str
    :param legend_labels_list: List of legend labels corresponding to each column in ``func_col``.
    :type legend_labels_list: list of list of str
    :param title: List of titles for each plot generated.
    :type title: list of str
    :param kind: Type of plot to generate (``"bar"`` or ``"barh"`` for horizontal bars). Default is ``"bar"``.
    :type kind: str, optional
    :param width: Width of the bars in the bar plot. Default is ``0.9``.
    :type width: float, optional
    :param rot: Rotation angle of the x-axis labels. Default is ``0``.
    :type rot: int, optional
    :param custom_order: Custom order for the categories in ``col``.
    :type custom_order: list, optional
    :param image_path_png: Directory path to save PNG plot images.
    :type image_path_png: str, optional
    :param image_path_svg: Directory path to save SVG plot images.
    :type image_path_svg: str, optional
    :param save_formats: List of file formats to save the plots (e.g., ``["png", "svg"]``). Default is ``None``.
    :type save_formats: list of str, optional
    :param color: List of colors to use for the plots. Default is the seaborn color palette.
    :type color: list of str, optional
    :param output: Specify the output type: ``"plots_only"``, ``"crosstabs_only"``, or ``"both"``. Default is ``"both"``.
    :type output: str, optional
    :param return_dict: Return the crosstabs as a dictionary. Default is ``False``.
    :type return_dict: bool, optional
    :param x: Width of the figure in inches.
    :type x: int, optional
    :param y: Height of the figure in inches.
    :type y: int, optional
    :param p: Padding between subplots.
    :type p: int, optional
    :param file_prefix: Prefix for filenames when saving plots.
    :type file_prefix: str, optional
    :param logscale: Apply a logarithmic scale to the y-axis. Default is ``False``.
    :type logscale: bool, optional
    :param plot_type: Type of plot to generate: ``"both"``, ``"regular"``, or ``"normalized"``. Default is ``"both"``.
    :type plot_type: str, optional
    :param show_legend: Show the legend on the plot. Default is ``True``.
    :type show_legend: bool, optional
    :param label_fontsize: Font size for axis labels. Default is ``12``.
    :type label_fontsize: int, optional
    :param tick_fontsize: Font size for tick labels. Default is ``10``.
    :type tick_fontsize: int, optional
    :param text_wrap: Maximum width of the title text before wrapping. Default is ``50``.
    :type text_wrap: int, optional
    :param remove_stacks: Remove stacks and create a regular bar plot. Only works when ``plot_type`` is ``"regular"``. Default is ``False``.
    :type remove_stacks: bool, optional
    :param xlim: Tuple or list specifying limits of the x-axis (e.g., ``(min, max)``).
    :type xlim: tuple or list, optional
    :param ylim: Tuple or list specifying limits of the y-axis (e.g., ``(min, max)``).
    :type ylim: tuple or list, optional

    :raises ValueError:
        - If ``remove_stacks`` is ``True`` and ``plot_type`` is not ``"regular"``.
        - If ``output`` is not one of ``"both"``, ``"plots_only"``, or ``"crosstabs_only"``.
        - If ``plot_type`` is not one of ``"both"``, ``"regular"``, or ``"normalized"``.
        - If lengths of ``title``, ``func_col``, and ``legend_labels_list`` are unequal.
    :raises KeyError: If any column in ``col`` or ``func_col`` is missing from the DataFrame.

    :returns: Dictionary of crosstabs DataFrames if ``return_dict`` is ``True``. Otherwise, returns ``None``.
    :rtype: dict or None

.. admonition:: Notes

    - To save images, specify the paths in ``image_path_png`` or ``image_path_svg`` along with a valid ``file_prefix``.
    - The ``save_formats`` parameter determines the file types for saved images.
    - This function is ideal for analyzing and visualizing categorical data distributions.


Stacked Bar Plots With Crosstabs Example
-----------------------------------------

The provided code snippet demonstrates how to use the ``stacked_crosstab_plot`` 
function to generate stacked bar plots and corresponding crosstabs for different 
columns in a DataFrame. Here's a detailed breakdown of the code using the census
dataset as an example [1]_.

First, the ``func_col`` list is defined, specifying the columns ``["sex", "income"]`` 
to be analyzed. These columns will be used in the loop to generate separate plots. 
The ``legend_labels_list`` is then defined, with each entry corresponding to a 
column in ``func_col``. In this case, the labels for the ``sex`` column are 
``["Male", "Female"]``, and for the ``income`` column, they are ``["<=50K", ">50K"]``. 
These labels will be used to annotate the legends of the plots.

Next, the ``title`` list is defined, providing titles for each plot corresponding 
to the columns in ``func_col``. The titles are set to ``["Sex", "Income"]``, 
which will be displayed on top of each respective plot.

.. note::

    The ``legend_labels_list`` parameter should be a list of lists, where each 
    inner list corresponds to the ground truth labels for the respective item in 
    the ``func_col`` list. Each element in the ``func_col`` list represents a 
    column in your DataFrame that you wish to analyze, and the corresponding 
    inner list in ``legend_labels_list`` should contain the labels that will be 
    used in the legend of your plots.


For example:

.. code-block:: python

    # Define the func_col to use in the loop in order of usage
    func_col = ["sex", "income"]

    # Define the legend_labels to use in the loop
    legend_labels_list = [
        ["Male", "Female"],  # Corresponds to "sex"
        ["<=50K", ">50K"],   # Corresponds to "income"
    ]

    # Define titles for the plots
    title = [
        "Sex",
        "Income",
    ]

.. important::
    
    Ensure that ``func_col``, ``legend_labels_list``, and ``title`` have the 
    same number of elements. Each item in ``func_col`` must correspond to a list 
    of labels in ``legend_labels_list`` and a title in ``title`` to ensure the 
    function generates plots with the correct labels and titles.

    Additionally, in this example, remove trailing periods from the ``income`` 
    column to correctly split its contents into two categories.


In this example:

- ``func_col`` contains two elements: ``"sex"`` and ``"income"``. Each corresponds to a specific column in your DataFrame.  
- ``legend_labels_list`` is a nested list containing two inner lists: 

    - The first inner list, ``["Male", "Female"]``, corresponds to the ``"sex"`` column in ``func_col``.
    - The second inner list, ``["<=50K", ">50K"]``, corresponds to the ``"income"`` column in ``func_col``.

- ``title`` contains two elements: ``"Sex"`` and ``"Income"``, which will be used as the titles for the respective plots.


.. note:: 

    Before proceeding with any further examples in this documentation, ensure that the ``age`` variable is binned into a new variable, ``age_group``.  
    Detailed instructions for this process can be found under :ref:`Binning Numerical Columns <Binning_Numerical_Columns>`.


.. code-block:: python

    from eda_toolkit import stacked_crosstab_plot

    stacked_crosstabs = stacked_crosstab_plot(
        df=df,
        col="age_group",
        func_col=func_col,
        legend_labels_list=legend_labels_list,
        title=title,
        kind="bar",
        width=0.8, 
        rot=0, 
        custom_order=None,
        color=["#00BFC4", "#F8766D"], 
        output="both",
        return_dict=True,
        x=14,
        y=8,
        p=10,
        logscale=False,
        plot_type="both",
        show_legend=True,
        label_fontsize=14,
        tick_fontsize=12,
    )

The above example generates stacked bar plots for ``"sex"`` and ``"income"`` 
grouped by ``"education"``. The plots are executed with legends, labels, and 
tick sizes customized for clarity. The function returns a dictionary of 
crosstabs for further analysis or export.

.. important:: 
    
    **Importance of Correctly Aligning Labels**

    It is crucial to properly align the elements in the ``legend_labels_list``, 
    ``title``, and ``func_col`` parameters when using the ``stacked_crosstab_plot`` 
    function. Each of these lists must be ordered consistently because the function 
    relies on their alignment to correctly assign labels and titles to the 
    corresponding plots and legends. 

    **For instance, in the example above:** 

    - The first element in ``func_col`` is ``"sex"``, and it is aligned with the first set of labels ``["Male", "Female"]`` in ``legend_labels_list`` and the first title ``"Sex"`` in the ``title`` list.
    - Similarly, the second element in ``func_col``, ``"income"``, aligns with the labels ``["<=50K", ">50K"]`` and the title ``"Income"``.

    **Misalignment between these lists would result in incorrect labels or titles being 
    applied to the plots, potentially leading to confusion or misinterpretation of the data. 
    Therefore, it's important to ensure that each list is ordered appropriately and 
    consistently to accurately reflect the data being visualized.**

    **Proper Setup of Lists**

    When setting up the ``legend_labels_list``, ``title``, and ``func_col``, ensure 
    that each element in the lists corresponds to the correct variable in the DataFrame. 
    This involves:

    - **Ordering**: Maintaining the same order across all three lists to ensure that labels and titles correspond correctly to the data being plotted.
    - **Consistency**: Double-checking that each label in ``legend_labels_list`` matches the categories present in the corresponding ``func_col``, and that the ``title`` accurately describes the plot.

    By adhering to these guidelines, you can ensure that the ``stacked_crosstab_plot`` 
    function produces accurate and meaningful visualizations that are easy to interpret and analyze.

**Output**

.. raw:: html

   <div class="no-click">

.. image:: ../assets/Stacked_Bar_Age_sex.svg
   :alt: KDE Distributions
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>

.. raw:: html

   <div class="no-click">

.. image:: ../assets/Stacked_Bar_Age_income.svg
   :alt: Stacked Bar Plot Age vs. Income
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


.. note::

    When you set ``return_dict=True``, you are able to see the crosstabs printed out 
    as shown below. 

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    font-weight:normal;overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg .tg-mwxe{text-align:right;vertical-align:middle}
    .tg .tg-p3ql{background-color:rgba(130, 130, 130, 0.08);text-align:right;vertical-align:middle}
    .tg .tg-yla0{font-weight:bold;text-align:left;vertical-align:middle}
    .tg .tg-7zrl{text-align:left;vertical-align:bottom}
    .tg .tg-zt7h{font-weight:bold;text-align:right;vertical-align:middle}
    .tg .tg-k750{background-color:rgba(130, 130, 130, 0.08);font-weight:bold;text-align:right;vertical-align:middle}
    @media screen and (max-width: 767px) {.tg {width: auto !important;}.tg col {width: auto !important;}.tg-wrap {overflow-x: auto;-webkit-overflow-scrolling: touch;margin: auto 0px;}}
    </style>
    <div class="tg-wrap"><table class="tg"><thead>
    <tr>
        <th class="tg-yla0" colspan="6">Crosstab for sex</th>
    </tr>
    <tr style="height: 10px;"><!-- Added empty row for spacing -->
        <td colspan="6" style="border: none;"></td>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td class="tg-zt7h">sex</td>
        <td class="tg-zt7h">Female</td>
        <td class="tg-zt7h">Male</td>
        <td class="tg-zt7h">Total</td>
        <td class="tg-zt7h">Female_%</td>
        <td class="tg-zt7h">Male_%</td>
    </tr>
    <tr>
        <td class="tg-k750">age_group</td>
        <td class="tg-k750"> </td>
        <td class="tg-k750"> </td>
        <td class="tg-k750"> </td>
        <td class="tg-k750"> </td>
        <td class="tg-k750"> </td>
    </tr>
    <tr>
        <td class="tg-mwxe">&lt; 18</td>
        <td class="tg-mwxe">295</td>
        <td class="tg-mwxe">300</td>
        <td class="tg-mwxe">595</td>
        <td class="tg-mwxe">49.58</td>
        <td class="tg-mwxe">50.42</td>
    </tr>
    <tr>
        <td class="tg-p3ql">18-29</td>
        <td class="tg-p3ql">5707</td>
        <td class="tg-p3ql">8213</td>
        <td class="tg-p3ql">13920</td>
        <td class="tg-p3ql">41</td>
        <td class="tg-p3ql">59</td>
    </tr>
    <tr>
        <td class="tg-mwxe">30-39</td>
        <td class="tg-mwxe">3853</td>
        <td class="tg-mwxe">9076</td>
        <td class="tg-mwxe">12929</td>
        <td class="tg-mwxe">29.8</td>
        <td class="tg-mwxe">70.2</td>
    </tr>
    <tr>
        <td class="tg-p3ql">40-49</td>
        <td class="tg-p3ql">3188</td>
        <td class="tg-p3ql">7536</td>
        <td class="tg-p3ql">10724</td>
        <td class="tg-p3ql">29.73</td>
        <td class="tg-p3ql">70.27</td>
    </tr>
    <tr>
        <td class="tg-mwxe">50-59</td>
        <td class="tg-mwxe">1873</td>
        <td class="tg-mwxe">4746</td>
        <td class="tg-mwxe">6619</td>
        <td class="tg-mwxe">28.3</td>
        <td class="tg-mwxe">71.7</td>
    </tr>
    <tr>
        <td class="tg-p3ql">60-69</td>
        <td class="tg-p3ql">939</td>
        <td class="tg-p3ql">2115</td>
        <td class="tg-p3ql">3054</td>
        <td class="tg-p3ql">30.75</td>
        <td class="tg-p3ql">69.25</td>
    </tr>
    <tr>
        <td class="tg-mwxe">70-79</td>
        <td class="tg-mwxe">280</td>
        <td class="tg-mwxe">535</td>
        <td class="tg-mwxe">815</td>
        <td class="tg-mwxe">34.36</td>
        <td class="tg-mwxe">65.64</td>
    </tr>
    <tr>
        <td class="tg-p3ql">80-89</td>
        <td class="tg-p3ql">40</td>
        <td class="tg-p3ql">91</td>
        <td class="tg-p3ql">131</td>
        <td class="tg-p3ql">30.53</td>
        <td class="tg-p3ql">69.47</td>
    </tr>
    <tr>
        <td class="tg-mwxe">90-99</td>
        <td class="tg-mwxe">17</td>
        <td class="tg-mwxe">38</td>
        <td class="tg-mwxe">55</td>
        <td class="tg-mwxe">30.91</td>
        <td class="tg-mwxe">69.09</td>
    </tr>
    <tr>
        <td class="tg-p3ql">Total</td>
        <td class="tg-p3ql">16192</td>
        <td class="tg-p3ql">32650</td>
        <td class="tg-p3ql">48842</td>
        <td class="tg-p3ql">33.15</td>
        <td class="tg-p3ql">66.85</td>
    </tr>
    <tr style="height: 10px;"><!-- Added empty row for spacing -->
        <td colspan="6" style="border: none;"></td>
    </tr>
    <tr>
        <th class="tg-yla0" colspan="6">Crosstab for income</th>
    </tr>
    <tr style="height: 10px;"><!-- Added empty row for spacing -->
        <td colspan="6" style="border: none;"></td>
    </tr>
    <tr>
        <td class="tg-zt7h">income</td>
        <td class="tg-zt7h">&lt;=50K</td>
        <td class="tg-zt7h">&gt;50K</td>
        <td class="tg-zt7h">Total</td>
        <td class="tg-zt7h">&lt;=50K_%</td>
        <td class="tg-zt7h">&gt;50K_%</td>
    </tr>
    <tr>
        <td class="tg-k750">age_group</td>
        <td class="tg-k750"> </td>
        <td class="tg-k750"> </td>
        <td class="tg-k750"> </td>
        <td class="tg-k750"> </td>
        <td class="tg-k750"> </td>
    </tr>
    <tr>
        <td class="tg-mwxe">&lt; 18</td>
        <td class="tg-mwxe">595</td>
        <td class="tg-mwxe">0</td>
        <td class="tg-mwxe">595</td>
        <td class="tg-mwxe">100</td>
        <td class="tg-mwxe">0</td>
    </tr>
    <tr>
        <td class="tg-p3ql">18-29</td>
        <td class="tg-p3ql">13174</td>
        <td class="tg-p3ql">746</td>
        <td class="tg-p3ql">13920</td>
        <td class="tg-p3ql">94.64</td>
        <td class="tg-p3ql">5.36</td>
    </tr>
    <tr>
        <td class="tg-mwxe">30-39</td>
        <td class="tg-mwxe">9468</td>
        <td class="tg-mwxe">3461</td>
        <td class="tg-mwxe">12929</td>
        <td class="tg-mwxe">73.23</td>
        <td class="tg-mwxe">26.77</td>
    </tr>
    <tr>
        <td class="tg-p3ql">40-49</td>
        <td class="tg-p3ql">6738</td>
        <td class="tg-p3ql">3986</td>
        <td class="tg-p3ql">10724</td>
        <td class="tg-p3ql">62.83</td>
        <td class="tg-p3ql">37.17</td>
    </tr>
    <tr>
        <td class="tg-mwxe">50-59</td>
        <td class="tg-mwxe">4110</td>
        <td class="tg-mwxe">2509</td>
        <td class="tg-mwxe">6619</td>
        <td class="tg-mwxe">62.09</td>
        <td class="tg-mwxe">37.91</td>
    </tr>
    <tr>
        <td class="tg-p3ql">60-69</td>
        <td class="tg-p3ql">2245</td>
        <td class="tg-p3ql">809</td>
        <td class="tg-p3ql">3054</td>
        <td class="tg-p3ql">73.51</td>
        <td class="tg-p3ql">26.49</td>
    </tr>
    <tr>
        <td class="tg-mwxe">70-79</td>
        <td class="tg-mwxe">668</td>
        <td class="tg-mwxe">147</td>
        <td class="tg-mwxe">815</td>
        <td class="tg-mwxe">81.96</td>
        <td class="tg-mwxe">18.04</td>
    </tr>
    <tr>
        <td class="tg-p3ql">80-89</td>
        <td class="tg-p3ql">115</td>
        <td class="tg-p3ql">16</td>
        <td class="tg-p3ql">131</td>
        <td class="tg-p3ql">87.79</td>
        <td class="tg-p3ql">12.21</td>
    </tr>
    <tr>
        <td class="tg-mwxe">90-99</td>
        <td class="tg-mwxe">42</td>
        <td class="tg-mwxe">13</td>
        <td class="tg-mwxe">55</td>
        <td class="tg-mwxe">76.36</td>
        <td class="tg-mwxe">23.64</td>
    </tr>
    <tr>
        <td class="tg-p3ql">Total</td>
        <td class="tg-p3ql">37155</td>
        <td class="tg-p3ql">11687</td>
        <td class="tg-p3ql">48842</td>
        <td class="tg-p3ql">76.07</td>
        <td class="tg-p3ql">23.93</td>
    </tr>
    </tbody></table></div>

\

When you set ``return_dict=True``, you can access these crosstabs as 
DataFrames by assigning them to their own vriables. For example: 

.. code-block:: python 

    crosstab_age_sex = stacked_crosstabs["sex"]
    crosstab_age_income = stacked_crosstabs["income"]


Pivoted Stacked Bar Plots Example
-----------------------------------

Using the census dataset [1]_, to create horizontal stacked bar plots, set the ``kind`` parameter to 
``"barh"`` in the ``stacked_crosstab_plot function``. This option pivots the 
standard vertical stacked bar plot into a horizontal orientation, making it easier 
to compare categories when there are many labels on the ``y-axis``.

.. raw:: html

   <div class="no-click">

.. image:: ../assets/Stacked_Bar_Age_income_pivoted.svg
   :alt: Stacked Bar Plot Age vs. Income (Pivoted)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Non-Normalized Stacked Bar Plots Example
----------------------------------------------------

In the census data [1]_, to create stacked bar plots without the normalized versions, 
set the ``plot_type`` parameter to ``"regular"`` in the ``stacked_crosstab_plot`` 
function. This option removes the display of normalized plots beneath the regular 
versions. Alternatively, setting the ``plot_type`` to ``"normalized"`` will display 
only the normalized plots. The example below demonstrates regular stacked bar plots 
for income by age.

.. raw:: html

   <div class="no-click">

.. image:: ../assets/Stacked_Bar_Age_income_regular.svg
   :alt: Stacked Bar Plot Age vs. Income (Regular)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Regular Non-Stacked Bar Plots Example
----------------------------------------------------

In the census data [1]_, to generate regular (non-stacked) bar plots without 
displaying their normalized versions, set the ``plot_type`` parameter to ``"regular"`` 
in the ``stacked_crosstab_plot`` function and enable ``remove_stacks`` by setting 
it to ``True``. This configuration removes any stacked elements and prevents the 
display of normalized plots beneath the regular versions. Alternatively, setting 
``plot_type`` to ``"normalized"`` will display only the normalized plots.

When unstacking bar plots in this fashion, the distribution is aligned in descending 
order, making it easier to visualize the most prevalent categories.

In the example below, the color of the bars has been set to a dark grey (``#333333``), 
and the legend has been removed by setting ``show_legend=False``. This illustrates 
regular bar plots for income by age, without stacking.


.. raw:: html

   <div class="no-click">

.. image:: ../assets/Bar_Age_regular_income.svg
   :alt: Bar Plot Age vs. Income (Regular)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Box and Violin Plots
=====================

**Create and save individual boxplots or violin plots, an entire grid of plots, or both for specified metrics and comparisons.**

The ``box_violin_plot`` function generates individual and/or grid-based plots of boxplots or violin plots for specified metrics against comparison categories in a DataFrame. It offers extensive customization options, including control over plot type, display mode, axis label rotation, figure size, and saving preferences, making it suitable for a wide range of data visualization needs.

This function supports:
- Rotating plots (swapping x and y axes).
- Adjusting font sizes for axis labels and tick labels.
- Wrapping plot titles for better readability.
- Saving plots in PNG and/or SVG format with customizable file paths.
- Visualizing the distribution of metrics across categories, either individually, as a grid, or both.

.. function:: box_violin_plot(df, metrics_list, metrics_comp, n_rows=None, n_cols=None, image_path_png=None, image_path_svg=None, save_plots=False, show_legend=True, plot_type="boxplot", xlabel_rot=0, show_plot="both", rotate_plot=False, individual_figsize=(6, 4), grid_figsize=None, label_fontsize=12, tick_fontsize=10, text_wrap=50, xlim=None, ylim=None, label_names=None, **kwargs)

    :param df: The DataFrame containing the data to plot.
    :type df: pandas.DataFrame
    :param metrics_list: List of column names representing the metrics to plot.
    :type metrics_list: list of str
    :param metrics_comp: List of column names representing the comparison categories.
    :type metrics_comp: list of str
    :param n_rows: Number of rows in the subplot grid. Automatically calculated if not provided.
    :type n_rows: int, optional
    :param n_cols: Number of columns in the subplot grid. Automatically calculated if not provided.
    :type n_cols: int, optional
    :param image_path_png: Directory path to save plots in PNG format.
    :type image_path_png: str, optional
    :param image_path_svg: Directory path to save plots in SVG format.
    :type image_path_svg: str, optional
    :param save_plots: Boolean indicating whether to save plots. Defaults to ``False``.
    :type save_plots: bool, optional
    :param show_legend: Whether to display the legend in the plots. Defaults to ``True``.
    :type show_legend: bool, optional
    :param plot_type: Type of plot to generate, either ``"boxplot"`` or ``"violinplot"``. Defaults to ``"boxplot"``.
    :type plot_type: str, optional
    :param xlabel_rot: Rotation angle for x-axis labels. Defaults to ``0``.
    :type xlabel_rot: int, optional
    :param show_plot: Specify the plot display mode: ``"individual"``, ``"grid"``, or ``"both"``. Defaults to ``"both"``.
    :type show_plot: str, optional
    :param rotate_plot: Whether to rotate the plots by swapping the x and y axes. Defaults to ``False``.
    :type rotate_plot: bool, optional
    :param individual_figsize: Dimensions (width, height) for individual plots. Defaults to ``(6, 4)``.
    :type individual_figsize: tuple, optional
    :param grid_figsize: Dimensions (width, height) for the grid plot.
    :type grid_figsize: tuple, optional
    :param label_fontsize: Font size for axis labels. Defaults to ``12``.
    :type label_fontsize: int, optional
    :param tick_fontsize: Font size for tick labels. Defaults to ``10``.
    :type tick_fontsize: int, optional
    :param text_wrap: Maximum width of plot titles before wrapping. Defaults to ``50``.
    :type text_wrap: int, optional
    :param xlim: Limits for the x-axis as a tuple or list (``min``, ``max``).
    :type xlim: tuple or list, optional
    :param ylim: Limits for the y-axis as a tuple or list (``min``, ``max``).
    :type ylim: tuple or list, optional
    :param label_names: Dictionary mapping original column names to custom labels for display purposes.
    :type label_names: dict, optional
    :param kwargs: Additional keyword arguments passed to the Seaborn plotting function.
    :type kwargs: additional keyword arguments

    :raises ValueError:
        - If ``show_plot`` is not one of ``"individual"``, ``"grid"``, or ``"both"``.
        - If ``save_plots`` is ``True`` but neither ``image_path_png`` nor ``image_path_svg`` is specified.
        - If ``rotate_plot`` is not a boolean value.
        - If ``individual_figsize`` is not a tuple or list of two numbers.
        - If ``grid_figsize`` is provided and is not a tuple or list of two numbers.

    :returns: None

.. admonition:: Notes

    - Automatically calculates grid dimensions if ``n_rows`` and ``n_cols`` are not specified.
    - Rotating plots swaps the roles of the x and y axes.
    - Saving plots requires specifying valid file paths for PNG and/or SVG formats.
    - Supports customization of plot labels, title wrapping, and font sizes for publication-quality visuals.



This function provides the ability to create and save boxplots or violin plots for specified metrics and comparison categories. It supports the generation of individual plots, a grid of plots, or both. Users can customize the appearance, save the plots to specified directories, and control the display of legends and labels.

Box Plots Grid Example
-----------------------

In this example with the US census data [1]_, the box_violin_plot function is employed to create a grid of 
boxplots, comparing different metrics against the ``"age_group"`` column in the 
DataFrame. The ``metrics_comp`` parameter is set to [``"age_group"``], meaning 
that the comparison will be based on different age groups. The ``metrics_list`` is 
provided as ``age_boxplot_list``, which contains the specific metrics to be visualized. 
The function is configured to arrange the plots in a grid formatThe ``image_path_png`` and 
``image_path_svg`` parameters are specified to save the plots in both PNG and 
SVG formats, and the save_plots option is set to ``"all"``, ensuring that both 
individual and grid plots are saved.

The plots are displayed in a grid format, as indicated by the ``show_plot="grid"`` 
parameter. The ``plot_type`` is set to ``"boxplot"``, so the function will generate 
boxplots for each metric in the list. Additionally, the ```x-axis``` labels are rotated 
by 90 degrees (``xlabel_rot=90``) to ensure that the labels are legible. The legend is 
hidden by setting ``show_legend=False``, keeping the plots clean and focused on the data. 
This configuration provides a comprehensive visual comparison of the specified 
metrics across different age groups, with all plots saved for future reference or publication.


.. code-block:: python

    age_boxplot_list = df[
        [
            "education-num",
            "hours-per-week",
        ]
    ].columns.to_list()


.. code-block:: python

    from eda_toolkit import box_violin_plot

    metrics_comp = ["age_group"]

    box_violin_plot(
        df=df,
        metrics_list=age_boxplot_list,
        metrics_comp=metrics_comp,
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        save_plots="all",
        show_plot="both",
        show_legend=False,
        plot_type="boxplot",
        xlabel_rot=90,
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/all_plots_comparisons_boxplot.png
   :alt: Box Plot Comparisons
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>

Violin Plots Grid Example
--------------------------

In this example with the US census data [1]_, we keep everything the same as the prior example, but change the 
``plot_type`` to ``violinplot``. This adjustment will generate violin plots instead 
of boxplots while maintaining all other settings.


.. code-block:: python

    from eda_toolkit import box_violin_plot

    metrics_comp = ["age_group"]

    box_violin_plot(
        df=df,
        metrics_list=age_boxplot_list,
        metrics_comp=metrics_comp,
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        save_plots="all",
        show_plot="both",
        show_legend=False,
        plot_type="violinplot",
        xlabel_rot=90,
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/all_plots_comparisons_violinplot.png
   :alt: Violin Plot Comparisons
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Pivoted Violin Plots Grid Example
------------------------------------

In this example with the US census data [1]_, we set ``xlabel_rot=0`` and ``rotate_plot=True`` 
to pivot the plot, changing the orientation of the axes while keeping the ``x-axis`` labels upright. 
This adjustment flips the axes, providing a different perspective on the data distribution.

.. code-block:: python

    from eda_toolkit import box_violin_plot

    metrics_comp = ["age_group"]

    box_violin_plot(
        df=df,
        metrics_list=age_boxplot_list,
        metrics_comp=metrics_comp,
        show_plot="both",
        rotate_plot=True,
        show_legend=False,
        plot_type="violinplot",
        xlabel_rot=0,
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/all_plots_comparisons_violinplot_pivoted.png
   :alt: Violin Plot Comparisons (Pivoted)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Scatter Plots and Best Fit Lines
==================================

Scatter Fit Plot
------------------

**Create and Save Scatter Plots or a Grid of Scatter Plots**

This function, ``scatter_fit_plot``, is designed to generate scatter plots for 
one or more pairs of variables (``x_vars`` and ``y_vars``) from a given DataFrame. 
The function can produce either individual scatter plots or organize multiple 
scatter plots into a grid layout, making it easy to visualize relationships between 
different pairs of variables in one cohesive view.

**Optional Best Fit Line**

An optional feature of this function is the ability to add a best fit line to the 
scatter plots. This line, often called a regression line, is calculated using a 
linear regression model and represents the trend in the data. By adding this line, 
you can visually assess the linear relationship between the variables, and the 
function can also display the equation of this line in the plot’s legend.s

**Customizable Plot Aesthetics**

The function offers a wide range of customization options to tailor the appearance 
of the scatter plots:

- **Point Color**: You can specify a default color for the scatter points or use a ``hue`` parameter to color the points based on a categorical variable. This allows for easy comparison across different groups within the data.

- **Point Size**: The size of the scatter points can be controlled and scaled based on another variable, which can help highlight differences or patterns related to that variable.

- **Markers**: The shape or style of the scatter points can also be customized. Whether you prefer circles, squares, or other marker types, the function allows you to choose the best representation for your data.

**Axis and Label Configuration**

The function also provides flexibility in setting axis labels, tick marks, and grid sizes. You can rotate axis labels for better readability, adjust font sizes, and even specify limits for the x and y axes to focus on particular data ranges.

**Plot Display and Saving Options**

The function allows you to display plots individually, as a grid, or both. Additionally, you can save the generated plots as PNG or SVG files, making it easy to include them in reports or presentations.

**Correlation Coefficient Display**

For users interested in understanding the strength of the relationship between variables, the function can also display the Pearson correlation coefficient directly in the plot title. This numeric value provides a quick reference to the linear correlation between the variables, offering further insight into their relationship.

.. function:: scatter_fit_plot(df, x_vars=None, y_vars=None, n_rows=None, n_cols=None, max_cols=4, image_path_png=None, image_path_svg=None, save_plots=None, show_legend=True, xlabel_rot=0, show_plot="both", rotate_plot=False, individual_figsize=(6, 4), grid_figsize=None, label_fontsize=12, tick_fontsize=10, text_wrap=50, add_best_fit_line=False, scatter_color="C0", best_fit_linecolor="red", best_fit_linestyle="-", hue=None, hue_palette=None, size=None, sizes=None, marker="o", show_correlation=True, xlim=None, ylim=None, all_vars=None, label_names=None, **kwargs)

    Generate scatter plots or a grid of scatter plots for the given ``x_vars`` and ``y_vars``, 
    with optional best fit lines, correlation coefficients, and customizable aesthetics.

    :param df: The DataFrame containing the data for the plots.
    :type df: pandas.DataFrame

    :param x_vars: List of variable names to plot on the ``x-axis``. If a single string is provided, it will be converted into a list with one element.
    :type x_vars: list of str or str, optional

    :param y_vars: List of variable names to plot on the ``y-axis``. If a single string is provided, it will be converted into a list with one element.
    :type y_vars: list of str or str, optional

    :param n_rows: Number of rows in the subplot grid. Calculated based on the number of plots and ``n_cols`` if not specified.
    :type n_rows: int, optional

    :param n_cols: Number of columns in the subplot grid. Calculated based on the number of plots and ``max_cols`` if not specified.
    :type n_cols: int, optional

    :param max_cols: Maximum number of columns in the subplot grid. Default is ``4``.
    :type max_cols: int, optional

    :param image_path_png: Directory path to save PNG images of the scatter plots.
    :type image_path_png: str, optional

    :param image_path_svg: Directory path to save SVG images of the scatter plots.
    :type image_path_svg: str, optional

    :param save_plots: Controls which plots to save: ``"all"``, ``"individual"``, or ``"grid"``. If ``None``, plots will not be saved.
    :type save_plots: str, optional

    :param show_legend: Whether to display the legend on the plots. Default is ``True``.
    :type show_legend: bool, optional

    :param xlabel_rot: Rotation angle for ``x-axis`` labels. Default is ``0``.
    :type xlabel_rot: int, optional

    :param show_plot: Controls plot display: ``"individual"``, ``"grid"``, or ``"both"``. Default is ``"both"``.
    :type show_plot: str, optional

    :param rotate_plot: Whether to rotate (pivot) the plots, swapping x and y axes. Default is ``False``.
    :type rotate_plot: bool, optional

    :param individual_figsize: Dimensions (width, height) of the figure for individual plots. Default is ``(6, 4)``.
    :type individual_figsize: tuple or list, optional

    :param grid_figsize: Dimensions (width, height) of the figure for grid plots. Calculated automatically if not specified.
    :type grid_figsize: tuple or list, optional

    :param label_fontsize: Font size for axis labels. Default is ``12``.
    :type label_fontsize: int, optional

    :param tick_fontsize: Font size for tick labels. Default is ``10``.
    :type tick_fontsize: int, optional

    :param text_wrap: The maximum width of the title text before wrapping. Default is ``50``.
    :type text_wrap: int, optional

    :param add_best_fit_line: Whether to add a best fit line to the scatter plots. Default is ``False``.
    :type add_best_fit_line: bool, optional

    :param scatter_color: Color code for the scatter points. Default is ``"C0"``.
    :type scatter_color: str, optional

    :param best_fit_linecolor: Color code for the best fit line. Default is ``"red"``.
    :type best_fit_linecolor: str, optional

    :param best_fit_linestyle: Linestyle for the best fit line. Default is ``"-"``.
    :type best_fit_linestyle: str, optional

    :param hue: Column name for the grouping variable that produces points with different colors.
    :type hue: str, optional

    :param hue_palette: Specifies colors for each hue level. Accepts a dictionary mapping hue levels to colors, a list of colors, or a seaborn color palette name. This requires the ``hue`` parameter to be set.
    :type hue_palette: dict, list, or str, optional

    :param size: Column name for the grouping variable that produces points with different sizes.
    :type size: str, optional

    :param sizes: Dictionary mapping sizes (smallest and largest) to min and max values for scatter points.
    :type sizes: dict, optional

    :param marker: Marker style for scatter points. Default is ``"o"``.
    :type marker: str, optional

    :param show_correlation: Whether to display the Pearson correlation coefficient in the plot title. Default is ``True``.
    :type show_correlation: bool, optional

    :param xlim: Limits for the ``x-axis`` as a tuple or list of (``min``, ``max``).
    :type xlim: tuple or list, optional

    :param ylim: Limits for the ``y-axis`` as a tuple or list of (``min``, ``max``).
    :type ylim: tuple or list, optional

    :param all_vars: If provided, generates scatter plots for all combinations of variables in this list, overriding ``x_vars`` and ``y_vars``.
    :type all_vars: list of str, optional

    :param label_names: Dictionary mapping original column names to custom labels for plot titles and axis labels.
    :type label_names: dict, optional

    :param kwargs: Additional keyword arguments to pass to the ``sns.scatterplot`` function.
    :type kwargs: dict, optional

    :raises ValueError: 
        - If ``all_vars`` is provided alongside ``x_vars`` or ``y_vars``.
        - If neither ``all_vars`` nor both ``x_vars`` and ``y_vars`` are provided.
        - If ``hue_palette`` is specified without ``hue``.
        - If ``show_plot`` is not one of ``"individual"``, ``"grid"``, or ``"both"``.
        - If ``save_plots`` is not one of ``None``, ``"all"``, ``"individual"``, or ``"grid"``.
        - If ``save_plots`` is set but no ``image_path_png`` or ``image_path_svg`` is specified.
        - If ``rotate_plot`` is not a boolean value.
        - If ``individual_figsize`` or ``grid_figsize`` is not a tuple or list of two numeric values.

    :returns: 
        ``None``. This function does not return any value but generates and optionally saves scatter plots for the specified ``x_vars`` and ``y_vars``, or for all combinations in ``all_vars`` if provided.


Regression-Centric Scatter Plots Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this US census data [1]_ example, the ``scatter_fit_plot`` function is 
configured to display the Pearson correlation coefficient and a best fit line 
on each scatter plot. The correlation coefficient is shown in the plot title, 
controlled by the ``show_correlation=True`` parameter, which provides a measure 
of the strength and direction of the linear relationship between the variables. 
Additionally, the ``add_best_fit_line=True`` parameter adds a best fit line to 
each plot, with the equation for the line displayed in the legend. This equation, 
along with the best fit line, helps to visually assess the relationship between 
the variables, making it easier to identify trends and patterns in the data. The 
combination of the correlation coefficient and the best fit line offers both 
a quantitative and visual representation of the relationships, enhancing the 
interpretability of the scatter plots.

.. code-block:: python

    from eda_toolkit import scatter_fit_plot

    scatter_fit_plot(
        df=df,
        x_vars=["age", "education-num"],
        y_vars=["hours-per-week"],
        show_legend=True,
        show_plot="grid",
        grid_figsize=None,
        label_fontsize=14,
        tick_fontsize=12,
        add_best_fit_line=True,
        scatter_color="#808080",
        show_correlation=True,
    )


.. raw:: html

   <div class="no-click">

.. image:: ../assets/scatter_plots_grid.png
   :alt: Scatter Plot Comparisons (with Best Fit Lines)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>

Scatter Plots Grouped by Category Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, the ``scatter_fit_plot`` function is used to generate a grid of 
scatter plots that examine the relationships between ``age`` and ``hours-per-week`` 
as well as ``education-num`` and ``hours-per-week``. Compared to the previous 
example, a few key inputs have been changed to adjust the appearance and functionality 
of the plots:

1. **Hue and Hue Palette**: The ``hue`` parameter is set to ``"income"``, meaning that the 
   data points in the scatter plots are colored according to the values in the ``income`` 
   column. A custom color mapping is provided via the ``hue_palette`` parameter, where the 
   income categories ``"<=50K"`` and ``">50K"`` are assigned the colors ``"brown"`` and 
   ``"green"``, respectively. This change visually distinguishes the data points based on 
   income levels.

2. **Scatter Color**: The ``scatter_color`` parameter is set to ``"#808080"``, which applies 
   a grey color to the scatter points when no ``hue`` is provided. However, since a ``hue`` 
   is specified in this example, the ``hue_palette`` takes precedence and overrides this color setting.

3. **Best Fit Line**: The ``add_best_fit_line`` parameter is set to ``False``, meaning that 
   no best fit line is added to the scatter plots. This differs from the previous example where 
   a best fit line was included.

4. **Correlation Coefficient**: The ``show_correlation`` parameter is set to ``False``, so the 
   Pearson correlation coefficient will not be displayed in the plot titles. This is another 
   change from the previous example where the correlation coefficient was included.

5. **Hue Legend**: The ``show_legend`` parameter remains set to ``True``, ensuring that the 
   legend displaying the hue categories (``"<=50K"`` and ``">50K"``) appears on the plots, 
   helping to interpret the color coding of the data points.

These changes allow for the creation of scatter plots that highlight the income levels 
of individuals, with custom color coding and without additional elements like a best 
fit line or correlation coefficient. The resulting grid of plots is then saved as 
images in the specified paths.


.. code-block:: python

    from eda_toolkit import scatter_fit_plot

    hue_dict = {"<=50K": "brown", ">50K": "green"}

    scatter_fit_plot(
        df=df,
        x_vars=["age", "education-num"],
        y_vars=["hours-per-week"],
        show_legend=True,
        show_plot="grid",
        label_fontsize=14,
        tick_fontsize=12,
        add_best_fit_line=False,
        scatter_color="#808080",
        hue="income",
        hue_palette=hue_dict,
        show_correlation=False,
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/scatter_plots_grid_grouped.png
   :alt: Scatter Plot Comparisons (Grouped)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Scatter Plots (All Combinations Example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, the ``scatter_fit_plot`` function is used to generate a grid of scatter plots that explore the relationships between all numeric variables in the ``df`` DataFrame. The function automatically identifies and plots all possible combinations of these variables. Below are key aspects of this example:

1. **All Variables Combination**: The ``all_vars`` parameter is used to automatically generate scatter plots for all possible combinations of numerical variables in the DataFrame. This means you don't need to manually specify ``x_vars`` and ``y_vars``, as the function will iterate through each possible pair.

2. **Grid Display**: The ``show_plot`` parameter is set to ``"grid"``, so the scatter plots are displayed in a grid format. This is useful for comparing multiple relationships simultaneously.

3. **Font Sizes**: The ``label_fontsize`` and ``tick_fontsize`` parameters are set to ``14`` and ``12``, respectively. This increases the readability of axis labels and tick marks, making the plots more visually accessible.

4. **Best Fit Line**: The ``add_best_fit_line`` parameter is set to ``True``, meaning that a best fit line is added to each scatter plot. This helps in visualizing the linear relationship between variables.

5. **Scatter Color**: The ``scatter_color`` parameter is set to ``"#808080"``, applying a grey color to the scatter points. This provides a neutral color that does not distract from the data itself.

6. **Correlation Coefficient**: The ``show_correlation`` parameter is set to ``True``, so the Pearson correlation coefficient will be displayed in the plot titles. This helps to quantify the strength of the relationship between the variables.

These settings allow for the creation of scatter plots that comprehensively explore the relationships between all numeric variables in the DataFrame. The plots are saved in a grid format, with added best fit lines and correlation coefficients for deeper analysis. The resulting images can be stored in the specified directory for future reference.

.. code-block:: python

    from eda_toolkit import scatter_fit_plot

    scatter_fit_plot(
        df=df,
        all_vars=df.select_dtypes(np.number).columns.to_list(),
        show_legend=True,
        show_plot="grid",
        label_fontsize=14,
        tick_fontsize=12,
        add_best_fit_line=True,
        scatter_color="#808080",
        show_correlation=True,
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/scatter_plots_all_grid.png
   :alt: Scatter Plot Comparisons (Grouped2)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>



Correlation Matrices
=====================

**Generate and Save Customizable Correlation Heatmaps**

The ``flex_corr_matrix`` function is designed to create highly customizable correlation heatmaps for visualizing the relationships between variables in a DataFrame. This function allows users to generate either a full or triangular correlation matrix, with options for annotation, color mapping, and saving the plot in multiple formats.

**Customizable Plot Appearance**

The function provides extensive customization options for the heatmap's appearance:

- **Colormap Selection**: Choose from a variety of colormaps to represent the strength of correlations. The default is ``"coolwarm"``, but this can be adjusted to fit the needs of the analysis.

- **Annotation**: Optionally annotate the heatmap with correlation coefficients, making it easier to interpret the strength of relationships at a glance.

- **Figure Size and Layout**: Customize the dimensions of the heatmap to ensure it fits well within reports, presentations, or dashboards.

**Triangular vs. Full Correlation Matrix**


A key feature of the ``flex_corr_matrix`` function is the ability to generate either a full correlation matrix or only the upper triangle. This option is particularly useful when the matrix is large, as it reduces visual clutter and focuses attention on the unique correlations.

**Label and Axis Configuration**


The function offers flexibility in configuring axis labels and titles:

- **Label Rotation**: Rotate x-axis and y-axis labels for better readability, especially when working with long variable names.
- **Font Sizes**: Adjust the font sizes of labels and tick marks to ensure the plot is clear and readable.
- **Title Wrapping**: Control the wrapping of long titles to fit within the plot without overlapping other elements.

**Plot Display and Saving Options**


The ``flex_corr_matrix`` function allows you to display the heatmap directly or save it as PNG or SVG files for use in reports or presentations. If saving is enabled, you can specify file paths and names for the images.

.. function:: flex_corr_matrix(df, cols=None, annot=True, cmap="coolwarm", save_plots=False, image_path_png=None, image_path_svg=None, figsize=(10, 10), title=None, label_fontsize=12, tick_fontsize=10, xlabel_rot=45, ylabel_rot=0, xlabel_alignment="right", ylabel_alignment="center_baseline", text_wrap=50, vmin=-1, vmax=1, cbar_label="Correlation Index", triangular=True, **kwargs)

    Create a customizable correlation heatmap with options for annotation, color mapping, figure size, and saving the plot.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame

    :param cols: List of column names to include in the correlation matrix. If None, all columns are included.
    :type cols: list of str, optional

    :param annot: Whether to annotate the heatmap with correlation coefficients. Default is ``True``.
    :type annot: bool, optional

    :param cmap: The colormap to use for the heatmap. Default is ``"coolwarm"``.
    :type cmap: str, optional

    :param save_plots: Controls whether to save the plots. Default is ``False``.
    :type save_plots: bool, optional

    :param image_path_png: Directory path to save PNG images of the heatmap.
    :type image_path_png: str, optional

    :param image_path_svg: Directory path to save SVG images of the heatmap.
    :type image_path_svg: str, optional

    :param figsize: Width and height of the figure for the heatmap. Default is ``(10, 10)``.
    :type figsize: tuple, optional

    :param title: Title of the heatmap. Default is ``None``.
    :type title: str, optional

    :param label_fontsize: Font size for tick labels and colorbar label. Default is ``12``.
    :type label_fontsize: int, optional

    :param tick_fontsize: Font size for axis tick labels. Default is ``10``.
    :type tick_fontsize: int, optional

    :param xlabel_rot: Rotation angle for x-axis labels. Default is ``45``.
    :type xlabel_rot: int, optional

    :param ylabel_rot: Rotation angle for y-axis labels. Default is ``0``.
    :type ylabel_rot: int, optional

    :param xlabel_alignment: Horizontal alignment for x-axis labels. Default is ``"right"``.
    :type xlabel_alignment: str, optional

    :param ylabel_alignment: Vertical alignment for y-axis labels. Default is ``"center_baseline"``.
    :type ylabel_alignment: str, optional

    :param text_wrap: The maximum width of the title text before wrapping. Default is ``50``.
    :type text_wrap: int, optional

    :param vmin: Minimum value for the heatmap color scale. Default is ``-1``.
    :type vmin: float, optional

    :param vmax: Maximum value for the heatmap color scale. Default is ``1``.
    :type vmax: float, optional

    :param cbar_label: Label for the colorbar. Default is ``"Correlation Index"``.
    :type cbar_label: str, optional

    :param triangular: Whether to show only the upper triangle of the correlation matrix. Default is ``True``.
    :type triangular: bool, optional

    :param kwargs: Additional keyword arguments to pass to ``seaborn.heatmap()``.
    :type kwargs: dict, optional

    :raises ValueError: 

        - If ``annot`` is not a boolean.
        - If ``cols`` is not a list.
        - If ``save_plots`` is not a boolean.
        - If ``triangular`` is not a boolean.
        - If ``save_plots`` is True but no image paths are provided.

    :returns: ``None``
        This function does not return any value but generates and optionally saves a correlation heatmap.

.. note::

    To save images, you must specify the paths for ``image_path_png`` or ``image_path_svg``. 
    Saving plots is triggered by providing a valid ``save_formats`` string.


Triangular Correlation Matrix Example
--------------------------------------

The provided code filters the census [1]_ DataFrame ``df`` to include only numeric columns using 
``select_dtypes(np.number)``. It then utilizes the ``flex_corr_matrix()`` function 
to generate a right triangular correlation matrix, which only displays the 
upper half of the correlation matrix. The heatmap is customized with specific 
colormap settings, title, label sizes, axis label rotations, and other formatting 
options. 

.. note:: 
    
    This triangular matrix format is particularly useful for avoiding 
    redundancy in correlation matrices, as it excludes the lower half, 
    making it easier to focus on unique pairwise correlations. 
    
    The function also includes a labeled color bar, helping users quickly interpret 
    the strength and direction of the correlations.

.. code-block:: python

    # Select only numeric data to pass into the function
    df_num = df.select_dtypes(np.number)

.. code-block:: python

    from eda_toolkit import flex_corr_matrix

    flex_corr_matrix(
        df=df,
        cols=df_num.columns.to_list(),
        annot=True,
        cmap="coolwarm",
        figsize=(10, 8),
        title="US Census Correlation Matrix",
        xlabel_alignment="right",
        label_fontsize=14,
        tick_fontsize=12,
        xlabel_rot=45,
        ylabel_rot=0,
        text_wrap=50,
        vmin=-1,
        vmax=1,
        cbar_label="Correlation Index",
        triangular=True,
    )


.. raw:: html

   <div class="no-click">

.. image:: ../assets/us_census_correlation_matrix.svg
   :alt: Scatter Plot Comparisons (Grouped)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>

Full Correlation Matrix Example
----------------------------------

In this modified census [1]_ example, the key changes are the use of the viridis 
colormap and the decision to plot the full correlation matrix instead of just the 
upper triangle. By setting ``cmap="viridis"``, the heatmap will use a different 
color scheme, which can provide better visual contrast or align with specific 
aesthetic preferences. Additionally, by setting ``triangular=False``, the full 
correlation matrix is displayed, allowing users to view all pairwise correlations, 
including both upper and lower halves of the matrix. This approach is beneficial 
when you want a comprehensive view of all correlations in the dataset.

.. code-block:: python

    from eda_toolkit import flex_corr_matrix

    flex_corr_matrix(
        df=df,
        cols=df_num.columns.to_list(),
        annot=True,
        cmap="viridis",
        figsize=(10, 8),
        title="US Census Correlation Matrix",
        xlabel_alignment="right",
        label_fontsize=14,
        tick_fontsize=12,
        xlabel_rot=45,
        ylabel_rot=0,
        text_wrap=50,
        vmin=-1,
        vmax=1,
        cbar_label="Correlation Index",
        triangular=False,
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/us_census_correlation_matrix_full.svg
   :alt: Scatter Plot Comparisons (Grouped)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Partial Dependence Plots
=========================

**Partial Dependence Plots (PDPs)** are a powerful tool in machine learning 
interpretability, providing insights into how features influence the predicted 
outcome of a model. PDPs can be generated in both 2D and 3D, depending on 
whether you want to analyze the effect of one feature or the interaction between 
two features on the model's predictions.

.. _2D_Partial_Dependence_Plots:

2D Partial Dependence Plots
-----------------------------

The ``plot_2d_pdp`` function generates 2D partial dependence plots (PDPs) for specified features or pairs of features. These plots help analyze the marginal effect of individual or paired features on the predicted outcome.

**Key Features**:

- **Flexible Plot Layouts**: Generate all 2D PDPs in a grid layout, as separate individual plots, or both for maximum versatility.
- **Customization Options**: Adjust figure size, font sizes for labels and ticks, and wrap long titles to ensure clear and visually appealing plots.
- **Save Plots**: Save generated plots in PNG or SVG formats with options to save all plots, only individual plots, or just the grid plot.

.. function:: plot_2d_pdp(model, X_train, feature_names, features, title="Partial dependence plot", grid_resolution=50, plot_type="grid", grid_figsize=(12, 8), individual_figsize=(6, 4), label_fontsize=12, tick_fontsize=10, text_wrap=50, image_path_png=None, image_path_svg=None, save_plots=None, file_prefix="partial_dependence")

    Generate and save 2D partial dependence plots for specified features using a trained machine learning model. The function supports grid and individual layouts and provides options for customization and saving plots in various formats.

    :param model: The trained machine learning model used to generate partial dependence plots.
    :type model: estimator object

    :param X_train: The training data used to compute partial dependence. Should correspond to the features used to train the model.
    :type X_train: pandas.DataFrame or numpy.ndarray

    :param feature_names: A list of feature names corresponding to the columns in ``X_train``.
    :type feature_names: list of str

    :param features: A list of feature indices or tuples of feature indices for which to generate partial dependence plots.
    :type features: list of int or tuple of int

    :param title: The title for the entire plot. Default is ``"Partial dependence plot"``.
    :type title: str, optional

    :param grid_resolution: The resolution of the grid used to compute the partial dependence. Higher values provide smoother curves but may increase computation time. Default is ``50``.
    :type grid_resolution: int, optional

    :param plot_type: The type of plot to generate. Choose ``"grid"`` for a grid layout, ``"individual"`` for separate plots, or ``"both"`` to generate both layouts. Default is ``"grid"``.
    :type plot_type: str, optional

    :param grid_figsize: Tuple specifying the width and height of the figure for the grid layout. Default is ``(12, 8)``.
    :type grid_figsize: tuple, optional

    :param individual_figsize: Tuple specifying the width and height of the figure for individual plots. Default is ``(6, 4)``.
    :type individual_figsize: tuple, optional

    :param label_fontsize: Font size for the axis labels and titles. Default is ``12``.
    :type label_fontsize: int, optional

    :param tick_fontsize: Font size for the axis tick labels. Default is ``10``.
    :type tick_fontsize: int, optional

    :param text_wrap: The maximum width of the title text before wrapping. Useful for managing long titles. Default is ``50``.
    :type text_wrap: int, optional

    :param image_path_png: The directory path where PNG images of the plots will be saved, if saving is enabled.
    :type image_path_png: str, optional

    :param image_path_svg: The directory path where SVG images of the plots will be saved, if saving is enabled.
    :type image_path_svg: str, optional

    :param save_plots: Controls whether to save the plots. Options include ``"all"``, ``"individual"``, ``"grid"``, or ``None`` (default). If saving is enabled, ensure ``image_path_png`` or ``image_path_svg`` are provided.
    :type save_plots: str, optional

    :param file_prefix: Prefix for the filenames of the saved grid plots. Default is ``"partial_dependence"``.
    :type file_prefix: str, optional

    :raises ValueError:
        - If ``plot_type`` is not one of ``"grid"``, ``"individual"``, or ``"both"``.
        - If ``save_plots`` is enabled but neither ``image_path_png`` nor ``image_path_svg`` is provided.

    :returns: ``None``
        This function generates partial dependence plots and displays them. It does not return any values.


2D Plots - CA Housing Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a scenario where you have a machine learning model predicting median 
house values in California. [4]_ Suppose you want to understand how non-location 
features like the average number of occupants per household (``AveOccup``) and the 
age of the house (``HouseAge``) jointly influence house values. A 2D partial 
dependence plot allows you to visualize this relationship in two ways: either as 
individual plots for each feature or as a combined plot showing the interaction 
between two features.

For instance, the 2D partial dependence plot can help you analyze how the age of 
the house impacts house values while holding the number of occupants constant, or 
vice versa. This is particularly useful for identifying the most influential 
features and understanding how changes in these features might affect the 
predicted house value.

If you extend this to two interacting features, such as ``AveOccup`` and ``HouseAge``, 
you can explore their combined effect on house prices. The plot can reveal how 
different combinations of occupancy levels and house age influence the value, 
potentially uncovering non-linear relationships or interactions that might not be
immediately obvious from a simple 1D analysis.

Here’s how you can generate and visualize these 2D partial dependence plots using 
the California housing dataset:

**Fetch The CA Housing Dataset and Prepare The DataFrame**

.. code-block:: python

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    import pandas as pd

    # Load the dataset
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)

**Split The Data Into Training and Testing Sets**

.. code-block:: python

    X_train, X_test, y_train, y_test = train_test_split(
        df, data.target, test_size=0.2, random_state=42
    )

**Train a GradientBoostingRegressor Model**

.. code-block:: python

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        loss="huber",
        random_state=42,
    )
    model.fit(X_train, y_train)


**Create 2D Partial Dependence Plot Grid**

.. code-block:: python

    from eda_toolkit import plot_2d_pdp

    # Feature names
    names = data.feature_names

    # Generate 2D partial dependence plots
    plot_2d_pdp(
        model=model,
        X_train=X_train,
        feature_names=names,
        features=[
            "MedInc",
            "AveOccup",
            "HouseAge",
            "AveRooms",
            "Population",
            ("AveOccup", "HouseAge"),
        ],
        title="PDP of house value on CA non-location features",
        grid_figsize=(14, 10),
        individual_figsize=(12, 4),
        label_fontsize=14,
        tick_fontsize=12,
        text_wrap=120,
        plot_type="grid",
        image_path_png="path/to/save/png",  
        save_plots="all",
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/2d_pdp_grid.svg
   :alt: Scatter Plot Comparisons (Grouped)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>

.. _3D_Partial_Dependence_Plots:

3D Partial Dependence Plots
-----------------------------

The ``plot_3d_pdp`` function extends the concept of partial dependence to three dimensions, allowing you to visualize the interaction between two features and their combined effect on the model’s predictions.

- **Interactive and Static 3D Plots**: Generate static 3D plots using Matplotlib or interactive 3D plots using Plotly. The function also allows for generating both types simultaneously.
- **Colormap and Layout Customization**: Customize the colormaps for both Matplotlib and Plotly plots. Adjust figure size, camera angles, and zoom levels to create plots that fit perfectly within your presentation or report.
- **Axis and Title Configuration**: Customize axis labels for both Matplotlib and Plotly plots. Adjust font sizes and control the wrapping of long titles to maintain readability.

.. function:: plot_3d_pdp(model, dataframe, feature_names_list, x_label=None, y_label=None, z_label=None, title, html_file_path=None, html_file_name=None, image_filename=None, plot_type="both", matplotlib_colormap=None, plotly_colormap="Viridis", zoom_out_factor=None, wireframe_color=None, view_angle=(22, 70), figsize=(7, 4.5), text_wrap=50, horizontal=-1.25, depth=1.25, vertical=1.25, cbar_x=1.05, cbar_thickness=25, title_x=0.5, title_y=0.95, top_margin=100, image_path_png=None, image_path_svg=None, show_cbar=True, grid_resolution=20, left_margin=20, right_margin=65, label_fontsize=8, tick_fontsize=6, enable_zoom=True, show_modebar=True)

    Generate 3D partial dependence plots for two features of a machine learning model.

    This function supports both static (Matplotlib) and interactive (Plotly) visualizations, allowing for flexible and comprehensive analysis of the relationship between two features and the target variable in a model.

    :param model: The trained machine learning model used to generate partial dependence plots.
    :type model: estimator object

    :param dataframe: The dataset on which the model was trained or a representative sample. If a DataFrame is provided, ``feature_names_list`` should correspond to the column names. If a NumPy array is provided, ``feature_names_list`` should correspond to the indices of the columns.
    :type dataframe: pandas.DataFrame or numpy.ndarray

    :param feature_names_list: A list of two feature names or indices corresponding to the features for which partial dependence plots are generated.
    :type feature_names_list: list of str

    :param x_label: Label for the x-axis in the plots. Default is ``None``.
    :type x_label: str, optional

    :param y_label: Label for the y-axis in the plots. Default is ``None``.
    :type y_label: str, optional

    :param z_label: Label for the z-axis in the plots. Default is ``None``.
    :type z_label: str, optional

    :param title: The title for the plots.
    :type title: str

    :param html_file_path: Path to save the interactive Plotly HTML file. Required if ``plot_type`` is ``"interactive"`` or ``"both"``. Default is ``None``.
    :type html_file_path: str, optional

    :param html_file_name: Name of the HTML file to save the interactive Plotly plot. Required if ``plot_type`` is ``"interactive"`` or ``"both"``. Default is ``None``.
    :type html_file_name: str, optional

    :param image_filename: Base filename for saving static Matplotlib plots as PNG and/or SVG. Default is ``None``.
    :type image_filename: str, optional

    :param plot_type: The type of plots to generate. Options are:
                      - ``"static"``: Generate only static Matplotlib plots.
                      - ``"interactive"``: Generate only interactive Plotly plots.
                      - ``"both"``: Generate both static and interactive plots. Default is ``"both"``.
    :type plot_type: str, optional

    :param matplotlib_colormap: Custom colormap for the Matplotlib plot. If not provided, a default colormap is used.
    :type matplotlib_colormap: matplotlib.colors.Colormap, optional

    :param plotly_colormap: Colormap for the Plotly plot. Default is ``"Viridis"``.
    :type plotly_colormap: str, optional

    :param zoom_out_factor: Factor to adjust the zoom level of the Plotly plot. Default is ``None``.
    :type zoom_out_factor: float, optional

    :param wireframe_color: Color for the wireframe in the Matplotlib plot. If ``None``, no wireframe is plotted. Default is ``None``.
    :type wireframe_color: str, optional

    :param view_angle: Elevation and azimuthal angles for the Matplotlib plot view. Default is ``(22, 70)``.
    :type view_angle: tuple, optional

    :param figsize: Figure size for the Matplotlib plot. Default is ``(7, 4.5)``.
    :type figsize: tuple, optional

    :param text_wrap: Maximum width of the title text before wrapping. Useful for managing long titles. Default is ``50``.
    :type text_wrap: int, optional

    :param horizontal: Horizontal camera position for the Plotly plot. Default is ``-1.25``.
    :type horizontal: float, optional

    :param depth: Depth camera position for the Plotly plot. Default is ``1.25``.
    :type depth: float, optional

    :param vertical: Vertical camera position for the Plotly plot. Default is ``1.25``.
    :type vertical: float, optional

    :param cbar_x: Position of the color bar along the x-axis in the Plotly plot. Default is ``1.05``.
    :type cbar_x: float, optional

    :param cbar_thickness: Thickness of the color bar in the Plotly plot. Default is ``25``.
    :type cbar_thickness: int, optional

    :param title_x: Horizontal position of the title in the Plotly plot. Default is ``0.5``.
    :type title_x: float, optional

    :param title_y: Vertical position of the title in the Plotly plot. Default is ``0.95``.
    :type title_y: float, optional

    :param top_margin: Top margin for the Plotly plot layout. Default is ``100``.
    :type top_margin: int, optional

    :param image_path_png: Directory path to save the PNG file of the Matplotlib plot. Default is None.
    :type image_path_png: str, optional

    :param image_path_svg: Directory path to save the SVG file of the Matplotlib plot. Default is None.
    :type image_path_svg: str, optional

    :param show_cbar: Whether to display the color bar in the Matplotlib plot. Default is ``True``.
    :type show_cbar: bool, optional

    :param grid_resolution: The resolution of the grid for computing partial dependence. Default is ``20``.
    :type grid_resolution: int, optional

    :param left_margin: Left margin for the Plotly plot layout. Default is ``20``.
    :type left_margin: int, optional

    :param right_margin: Right margin for the Plotly plot layout. Default is ``65``.
    :type right_margin: int, optional

    :param label_fontsize: Font size for axis labels in the Matplotlib plot. Default is ``8``.
    :type label_fontsize: int, optional

    :param tick_fontsize: Font size for tick labels in the Matplotlib plot. Default is ``6``.
    :type tick_fontsize: int, optional

    :param enable_zoom: Whether to enable zooming in the Plotly plot. Default is ``True``.
    :type enable_zoom: bool, optional

    :param show_modebar: Whether to display the mode bar in the Plotly plot. Default is ``True``.
    :type show_modebar: bool, optional

    :raises ValueError: 
        - If ``plot_type`` is not one of ``"static"``, ``"interactive"``, or ``"both"``. 
        - If ``plot_type`` is ``"interactive"`` or ``"both"`` and ``html_file_path`` or ``html_file_name`` are not provided.

    :returns: ``None`` 
        This function generates 3D partial dependence plots and displays or saves them. It does not return any values.
    
    .. note::

        - This function handles warnings related to scikit-learn's ``partial_dependence`` function, specifically a ``FutureWarning`` related to non-tuple sequences for multidimensional indexing. This warning is suppressed as it stems from the internal workings of scikit-learn in Python versions like 3.7.4.
        - To maintain compatibility with different versions of scikit-learn, the function attempts to use ``"values"`` for grid extraction in newer versions and falls back to ``"grid_values"`` for older versions.

3D Plots - CA Housing Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a scenario where you have a machine learning model predicting median 
house values in California.[4]_ Suppose you want to understand how non-location 
features like the average number of occupants per household (``AveOccup``) and the 
age of the house (``HouseAge``) jointly influence house values. A 3D partial 
dependence plot allows you to visualize this relationship in a more comprehensive 
manner, providing a detailed view of how these two features interact to affect 
the predicted house value.

For instance, the 3D partial dependence plot can help you explore how different 
combinations of house age and occupancy levels influence house values. By 
visualizing the interaction between AveOccup and HouseAge in a 3D space, you can 
uncover complex, non-linear relationships that might not be immediately apparent 
in 2D plots.

This type of plot is particularly useful when you need to understand the joint 
effect of two features on the target variable, as it provides a more intuitive 
and detailed view of how changes in both features impact predictions simultaneously.

Here’s how you can generate and visualize these 3D partial dependence plots 
using the California housing dataset:

Static Plot
^^^^^^^^^^^^^^^^^

**Fetch The CA Housing Dataset and Prepare The DataFrame**

.. code-block:: python

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Load the dataset
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)

**Split The Data Into Training and Testing Sets**

.. code-block:: python

    X_train, X_test, y_train, y_test = train_test_split(
        df, data.target, test_size=0.2, random_state=42
    )

**Train a GradientBoostingRegressor Model**

.. code-block:: python

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        loss="huber",
        random_state=1,
    )
    model.fit(X_train, y_train)

**Create Static 3D Partial Dependence Plot**

.. code-block:: python

    from eda_toolkit import plot_3d_pdp

    plot_3d_pdp(
        model=model,
        dataframe=X_test,  
        feature_names_list=["HouseAge", "AveOccup"],
        x_label="House Age",
        y_label="Average Occupancy",
        z_label="Partial Dependence",
        title="3D Partial Dependence Plot of House Age vs. Average Occupancy",
        image_filename="3d_pdp",
        plot_type="static",
        figsize=[8, 5],
        text_wrap=40,
        wireframe_color="black",
        image_path_png=image_path_png,
        grid_resolution=30,
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/3d_pdp.svg
   :alt: Scatter Plot Comparisons (Grouped)
   :align: center
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>



Interactive Plot
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from eda_toolkit import plot_3d_pdp

    plot_3d_pdp(
        model=model,
        dataframe=X_test, 
        feature_names_list=["HouseAge", "AveOccup"],
        x_label="House Age",
        y_label="Average Occupancy",
        z_label="Partial Dependence",
        title="3D Partial Dependence Plot of House Age vs. Average Occupancy",
        html_file_path=image_path_png,
        image_filename="3d_pdp",
        html_file_name="3d_pdp.html",
        plot_type="interactive",
        text_wrap=80,
        zoom_out_factor=1.2,
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        grid_resolution=30,
        label_fontsize=8,
        tick_fontsize=6,
        title_x=0.38,
        top_margin=10,
        right_margin=50,
        left_margin=50,
        cbar_x=0.9,
        cbar_thickness=25,
        show_modebar=False,
        enable_zoom=True,
    )

.. warning::

   **Scrolling Notice:**

   While interacting with the interactive Plotly plot below, scrolling down the 
   page using the mouse wheel may be blocked when the mouse pointer is hovering 
   over the plot. To continue scrolling, either move the mouse pointer outside 
   the plot area or use the keyboard arrow keys to navigate down the page.


.. raw:: html

    <iframe src="3d_pdp.html" style="border:none; width:100%; height:650px; margin-left: 0; padding: 0; overflow: auto;" scrolling="no"></iframe>

    <div style="height: 50px;"></div>


This interactive plot was generated using Plotly, which allows for rich, 
interactive visualizations directly in the browser. The plot above is an example
of an interactive 3D Partial Dependence Plot. Here's how it differs from 
generating a static plot using Matplotlib.

**Key Differences**

**Plot Type**:

- The ``plot_type`` is set to ``"interactive"`` for the Plotly plot and ``"static"`` for the Matplotlib plot.

**Interactive-Specific Parameters**:

- **HTML File Path and Name**: The ``html_file_path`` and ``html_file_name`` parameters are required to save the interactive Plotly plot as an HTML file. These parameters are not needed for static plots.
  
- **Zoom and Positioning**: The interactive plot includes parameters like ``zoom_out_factor``, ``title_x``, ``cbar_x``, and ``cbar_thickness`` to control the zoom level, title position, and color bar position in the Plotly plot. These parameters do not affect the static plot.
  
- **Mode Bar and Zoom**: The ``show_modebar`` and ``enable_zoom`` parameters are specific to the interactive Plotly plot, allowing you to toggle the visibility of the mode bar and enable or disable zoom functionality.

**Static-Specific Parameters**:

- **Figure Size and Wireframe Color**: The static plot uses parameters like ``figsize`` to control the size of the Matplotlib plot and ``wireframe_color`` to define the color of the wireframe in the plot. These parameters are not applicable to the interactive Plotly plot.

By adjusting these parameters, you can customize the behavior and appearance of your 3D Partial Dependence Plots according to your needs, whether for static or interactive visualization.




.. [1] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.

.. [2] Waskom, M. (2021). *Seaborn: Statistical Data Visualization*. *Journal of Open Source Software*, 6(60), 3021. `https://doi.org/10.21105/joss.03021 <https://doi.org/10.21105/joss.03021>`_.

.. [3] Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. *Computing in Science & Engineering*, 9(3), 90-95. `https://doi.org/10.1109/MCSE.2007.55 <https://doi.org/10.1109/MCSE.2007.55>`_.

.. [4] Pace, R. K., & Barry, R. (1997). *Sparse Spatial Autoregressions*. *Statistics & Probability Letters*, 33(3), 291-297. `https://doi.org/10.1016/S0167-7152(96)00140-X <https://doi.org/10.1016/S0167-7152(96)00140-X>`_.

