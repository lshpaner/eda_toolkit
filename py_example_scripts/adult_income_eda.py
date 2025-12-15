################################################################################
## Import Requisite Libraries
################################################################################

import pandas as pd
import numpy as np
import os
import textwrap
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from eda_toolkit import (
    ensure_directory,
    kde_distributions,
    data_doctor,
    strip_trailing_period,
    stacked_crosstab_plot,
    box_violin_plot,
    scatter_fit_plot,
    flex_corr_matrix,
    conditional_histograms,
    distribution_gof_plots,
)

import pandas as pd


plt.ion()  # enables interactive mode
plt.rcParams["figure.max_open_warning"] = 50  # or some other threshold

# Get the width of the terminal
terminal_width = os.get_terminal_size().columns

## Ensure Directory

base_path = os.path.join(os.pardir)

# Go up one level from 'notebooks' to parent directory,
# then into the 'data' folder
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, "..", "data")
data_output = os.path.join(os.pardir, "data_output")

# create image paths
image_path_png = os.path.join(base_path, "images", "png_images")
image_path_svg = os.path.join(base_path, "images", "svg_images")

# Use the function to ensure'data' directory exists
ensure_directory(data_path)
ensure_directory(data_output)
ensure_directory(image_path_png)
ensure_directory(image_path_svg)

################################################################################
## UCI ML Repository
################################################################################

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# Combine X and y into entire df
df = X.join(y, how="inner")

print()
print("Adult Income Dataset")
print
print(df.head())
print("*" * terminal_width)

## Binning Numerical Columns

bin_ages = [
    0,
    18,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    float("inf"),
]

label_ages = [
    "< 18",
    "18-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80-89",
    "90-99",
    "100 +",
]


df["age_group"] = pd.cut(
    df["age"],
    bins=bin_ages,
    labels=label_ages,
    right=False,
)

################################################################################
## KDE and Histograms
################################################################################

vars_of_interest = [
    "age",
    "education-num",
    "hours-per-week",
]


kde_distributions(
    df=df,
    n_rows=1,
    n_cols=3,
    subplot_figsize=(14, 4),  # Size of the overall subplots figure
    fill=True,
    fill_alpha=0.60,
    text_wrap=50,
    bbox_inches="tight",
    vars_of_interest=vars_of_interest,
    y_axis_label="Density",
    bins=10,
    plot_type="both",  # Can also just plot KDE by itself by passing "kde"
    label_fontsize=16,  # Font size for axis labels
    tick_fontsize=14,  # Font size for tick labels
    image_filename="age_distribution_kde",
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
)

## Histogram Example (Density)

kde_distributions(
    df=df,
    n_rows=1,
    n_cols=3,
    subplot_figsize=(14, 4),  # Size of the overall subplots figure
    fill=True,
    text_wrap=50,
    bbox_inches="tight",
    vars_of_interest=vars_of_interest,
    y_axis_label="Density",
    bins=10,
    plot_type="hist",
    label_fontsize=16,  # Font size for axis labels
    tick_fontsize=14,  # Font size for tick labels
    image_filename="age_distribution_density",
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
)

### Histogram Example (Count)


kde_distributions(
    df=df,
    n_rows=1,
    n_cols=3,
    subplot_figsize=(14, 4),  # Size of the overall subplots figure
    text_wrap=50,
    hist_color="orange",
    bbox_inches="tight",
    vars_of_interest=vars_of_interest,
    y_axis_label="Count",
    bins=10,
    plot_type="hist",
    stat="Count",
    label_fontsize=16,  # Font size for axis labels
    tick_fontsize=14,  # Font size for tick labels
    image_filename="age_distribution_count",
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
)

## Histogram Example - (Mean and Median)


kde_distributions(
    df=df,
    n_rows=1,
    n_cols=3,
    subplot_figsize=(14, 4),  # Size of the overall subplots figure
    text_wrap=50,
    hist_color="brown",
    bbox_inches="tight",
    vars_of_interest=vars_of_interest,
    y_axis_label="Density",
    bins=10,
    fill_alpha=0.60,
    plot_type="hist",
    stat="Density",
    label_fontsize=16,  # Font size for axis labels
    tick_fontsize=14,  # Font size for tick labels
    plot_mean=True,
    plot_median=True,
    mean_color="blue",
    image_filename="age_distribution_mean_median",
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
)

## Histogram Example - (Mean, Median, and Std. Deviation)

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
    label_fontsize=16,  # Font size for axis labels
    tick_fontsize=14,  # Font size for tick labels
    plot_mean=True,
    plot_median=True,
    mean_color="blue",
    image_path_svg=image_path_svg,
    image_path_png=image_path_png,
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
    image_filename="age_distribution_mean_median_std",
)


## from eda_toolkit import plot_distributions

vars_of_interest = [
    "age",
    "education-num",
    "hours-per-week",
]

kde_distributions(
    df=df,
    vars_of_interest=vars_of_interest,
    # layout
    n_rows=1,
    n_cols=3,
    hue=None,
    hist_color="yellow",
    subplot_figsize=(14, 4),
    # plotting intent
    plot_type="both",  # histogram + density overlay
    stat="density",
    density_function=["kde", "norm", "lognorm"],
    density_color=["blue", "black", "red"],
    density_fit="MLE",
    # histogram controls
    bins=10,
    fill=True,
    # labels & formatting
    y_axis_label="Density",
    text_wrap=50,
    label_fontsize=16,
    tick_fontsize=14,
    # output
    bbox_inches="tight",
    image_filename="age_distribution_norm_fit",
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
)

################################################################################
## Feature Scaling and Outliers
################################################################################

#### Box-Cox Transformation Example 1


def data_doctor_1():
    """
    Box-Cox Transformation Example 1
    ---------------------------------

    In this example from the US Census dataset 1, we demonstrate the usage of
    the data_doctor function to apply a Box-Cox transformation to the age
    column in a DataFrame. The data_doctor function provides a flexible way
    to preprocess data by applying various scaling techniques. In this case,
    we apply the Box-Cox transformation without any tuning of the alpha or
    lambda parameters, allowing the function to handle the transformation in
    a barebones approach. You can also choose other scaling conversions from
    the list of available options (such as 'minmax', 'standard', 'robust',
    etc.), depending on your needs.

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
        figsize=(10, 3),
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        image_filename="data_doctor_age_box_cox_adult_income",
    )
    """


clean_docstring = textwrap.dedent(data_doctor_1.__doc__)
print()
print(clean_docstring)
print("*" * terminal_width)

################################################################################
print()

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
    figsize=(10, 3),
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    image_filename="data_doctor_age_box_cox_adult_income",
)

print("*" * terminal_width)
print()
print("Data Doctor Example 1")
print()
print(df.head())

################################################################################
print()


def data_doctor_2():
    """
    Box-Cox Transformation Example 2
    ---------------------------------

    In this second example from the US Census dataset, we apply the Box-Cox
    transformation to the age column in a DataFrame, but this time with custom
    keyword arguments passed through the scale_conversion_kws. Specifically, we
    provide an alpha value of 0.8, influencing the confidence interval for the
    transformation. Additionally, we customize the visual appearance of the plots
    by specifying keyword arguments for the violinplot, KDE, and histogram plots.
    These customizations allow for greater control over the visual output.

    data_doctor(
        df=df,
        feature_name="fnlwgt",
        data_fraction=0.6,
        plot_type=["box_violin", "hist"],
        hist_kws={"color": "gray"},
        figsize=(8, 4),
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        image_filename="data_doctor_age_box_cox_full_data_adult_income",
        random_state=111,
    )
    """


clean_docstring = textwrap.dedent(data_doctor_2.__doc__)
print(clean_docstring)


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
    figsize=(10, 3),
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    image_filename="data_doctor_age_box_cox_full_data_adult_income",
)

print()
print("*" * terminal_width)
print("Data Doctor Example 2")
print()
print(df.head())
print("*" * terminal_width)

##### Retaining a Sample for Analysis

print()
print("Retaining a sample for analysis:")
print()
sampled_df = df.sample(frac=0.6, random_state=111)
print(sampled_df.head())
print()
print("*" * terminal_width)

sampled_df.head()

print(
    f"The sampled dataframe has {sampled_df.shape[0]} rows and "
    f"{sampled_df.shape[1]} columns."
)

################################################################################
print()


def data_doctor_3():
    """
    Box-Cox Transformation Example 3
    ---------------------------------

    In this example, we examine the final weight (fnlwgt) feature from the US
    Census dataset, focusing on detecting outliers without applying any scaling
    transformations. The data_doctor function is used with minimal configuration
    to visualize where outliers are present in the raw data.

    By enabling apply_cutoff=True and selecting plot_type=["box_violin", "hist"],
    we can clearly identify outliers both visually and numerically. This basic
    setup highlights the outliers without altering the data distribution, making
    it easy to see extreme values that could affect further analysis.


    data_doctor(
        df=df,
        feature_name="fnlwgt",
        data_fraction=0.6,
        plot_type=["box_violin", "hist"],
        hist_kws={"color": "gray"},
        figsize=(8, 4),
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        image_filename="data_doctor_fnlwgt_adult_income",
        random_state=111,
    )

    """


clean_docstring = textwrap.dedent(data_doctor_3.__doc__)
print(clean_docstring)

data_doctor(
    df=df,
    feature_name="fnlwgt",
    data_fraction=0.6,
    plot_type=["box_violin", "hist"],
    hist_kws={"color": "gray"},
    figsize=(8, 4),
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    image_filename="data_doctor_fnlwgt_adult_income",
    random_state=111,
)

################################################################################
print()


def data_doctor_4():
    """
    Box-Cox Transformation Example 4
    ---------------------------------

    In this scenario, we address the extreme values observed in the fnlwgt
    feature by applying a visual cutoff based on the distribution seen in the
    previous example. Here, we set an approximate upper cutoff of 400,000 to
    limit the impact of outliers without any additional scaling or transformation.
    By using apply_cutoff=True along with upper_cutoff=400000, we effectively
    cap the extreme values.

    This example also demonstrates how you can further customize the visualization
    by specifying additional histogram keyword arguments with hist_kws. Here, we
    use bins=20 to adjust the bin size, creating a smoother view of the feature’s
    distribution within the cutoff limits.

    In the resulting visualization, you will see that the boxplot and histogram
    have a controlled range due to the applied upper cutoff, limiting the
    influence of extreme outliers on the visual representation. This treatment
    provides a clearer view of the primary distribution, allowing for a more
    focused analysis on the bulk of the data without outliers distorting the scale.


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
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        image_filename="data_doctor_fnlwgt_cutoff_adult_income",
        random_state=111,
    )

    """


clean_docstring = textwrap.dedent(data_doctor_4.__doc__)
print(clean_docstring)

# ### Treated Outliers With Cutoffs

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
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    image_filename="data_doctor_fnlwgt_cutoff_adult_income",
    random_state=111,
)

print("*" * terminal_width)
print()

################################################################################
#### RobustScaler Outliers Examples

print()


def data_doctor_5():
    """
    Box-Cox Transformation Example 5
    ---------------------------------

    In this example from the US Census dataset 1, we apply the RobustScaler
    transformation to the age column in a DataFrame to address potential outliers.
    The data_doctor function enables users to apply transformations with specific
    configurations via the scale_conversion_kws parameter, making it ideal for
    refining how outliers affect scaling.

    For this example, we set the following custom keyword arguments:

    Disable centering: By setting with_centering=False, the transformation scales
    based only on the range, without shifting the median to zero.

    Adjust quantile range: We specify a narrower quantile_range of (10.0, 90.0)
    to reduce the influence of extreme values on scaling.

    The following code demonstrates this transformation:


    data_doctor(
        df=df,
        feature_name="age",
        data_fraction=0.6,
        scale_conversion="robust",
        apply_as_new_col_to_df=True,
        scale_conversion_kws={
            "with_centering": False,  # Disable centering
            "quantile_range": (10.0, 90.0),  # Use a custom quantile range
        },
        random_state=111,
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        image_filename="data_doctor_age_robust_scaling_adult_income",
    )

    """


clean_docstring = textwrap.dedent(data_doctor_5.__doc__)
print(clean_docstring)

data_doctor(
    df=df,
    feature_name="age",
    data_fraction=0.6,
    scale_conversion="robust",
    apply_as_new_col_to_df=True,
    scale_conversion_kws={
        "with_centering": False,  # Disable centering
        "quantile_range": (10.0, 90.0),  # Use a custom quantile range
    },
    random_state=111,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    image_filename="data_doctor_age_robust_scaling_adult_income",
)

print("*" * terminal_width)
print()


def data_doctor_example_multi_plot():
    """
    Multi-Plot Diagnostic Example
    ------------------------------

    In this example using the US Census Adult Income dataset, we demonstrate how
    to generate multiple diagnostic plots for a single feature using the
    `data_doctor` function.

    Rather than generating all available plots, we explicitly request a custom
    subset of visualizations by passing a list to the `plot_type` argument.
    This allows fine-grained control over which diagnostics are displayed while
    keeping everything within a single, aligned figure.

    For this example, we apply the following configuration:

    Feature analyzed:
    - age

    Transformation:
    - Natural logarithm (`scale_conversion="log"`) to reduce right skew and
      compress large values.

    Plot types requested:
    - KDE: to visualize the smoothed density of the transformed feature
    - ECDF: to inspect the cumulative distribution and tail behavior
    - Box/Violin: to summarize central tendency and dispersion

    This combination is especially useful for quickly assessing distribution
    shape, skewness, and outliers after a transformation.

    The following code demonstrates this workflow:


    print("\\nRunning data_doctor with plot_type=['kde', 'ecdf', 'box_violin'] ...\\n")

    data_doctor(
        df=adult_df,
        feature_name="age",
        plot_type=["kde", "ecdf", "box_violin"],
        scale_conversion="log",
    )

    """


print("\\nRunning data_doctor with plot_type=['kde', 'ecdf', 'box_violin'] ...\\n")

data_doctor(
    df=df,
    feature_name="age",
    plot_type=["kde", "ecdf", "box_violin"],
    scale_conversion="log",
)

print("*" * terminal_width)
print()

################################################################################
## Stacked Crosstab Plots
################################################################################

## Stacked Bar Plots With Crosstabs Example

print("Stripping Trailing Period From `Income` Column...")
df = strip_trailing_period(df=df, column_name="income")
print()
print(df.head())
print("*" * terminal_width)
print()
################################################################################


# Define the func_col to use in the loop in order of usage
func_col = ["sex", "income"]

# Define the legend_labels to use in the loop
legend_labels_list = [
    ["Male", "Female"],  # Corresponds to "sex"
    ["<=50K", ">50K"],  # Corresponds to "income"
]

# Define titles for the plots
title = [
    "Sex",
    "Income",
]
################################################################################

# Call the stacked_crosstab_plot function
stacked_crosstabs = stacked_crosstab_plot(
    df=df,
    col="age_group",
    func_col=func_col,
    legend_labels_list=legend_labels_list,
    title=title,
    kind="bar",
    width=0.8,
    rot=0,  # axis rotation angle
    custom_order=None,
    color=["#00BFC4", "#F8766D"],  # default color schema
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
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    file_prefix="stacked_bar",
    save_formats=["png", "svg"],
)


print("*" * terminal_width)

## Non-Normalized Stacked Bar Plots Example

# Call the stacked_crosstab_plot function
stacked_crosstabs = stacked_crosstab_plot(
    df=df,
    col="age_group",
    func_col=func_col,
    legend_labels_list=legend_labels_list,
    title=title,
    kind="bar",
    width=0.8,
    rot=0,  # axis rotation angle
    custom_order=None,
    color=["#00BFC4", "#F8766D"],  # default color schema
    output="both",
    return_dict=True,
    x=14,
    y=8,
    p=10,
    logscale=False,
    plot_type="regular",
    show_legend=True,
    label_fontsize=14,
    tick_fontsize=12,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    file_prefix="stacked_bar_non_normalized",
    save_formats=["png", "svg"],
)

print("*" * terminal_width)

## Regular Non-Stacked Bar Plots Example

# Call the stacked_crosstab_plot function
stacked_crosstabs = stacked_crosstab_plot(
    df=df,
    col="age_group",
    func_col=func_col,
    legend_labels_list=legend_labels_list,
    title=title,
    kind="bar",
    width=0.8,
    rot=0,  # axis rotation angle
    custom_order=None,
    color="#333333",
    output="both",
    return_dict=True,
    x=14,
    y=8,
    p=10,
    logscale=False,
    plot_type="regular",
    show_legend=True,
    label_fontsize=14,
    tick_fontsize=12,
    remove_stacks=True,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    file_prefix="unstacked_bar",
    save_formats=["png", "svg"],
)

print("*" * terminal_width)

################################################################################
## Box and Violin Plots
## Box Plots Subplots Example
################################################################################

age_boxplot_list = df[
    [
        "education-num",
        "hours-per-week",
    ]
].columns.to_list()

metrics_comp = ["age_group"]

box_violin_plot(
    df=df,
    metrics_list=age_boxplot_list,
    metrics_comp=metrics_comp,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    save_plots=True,
    show_plot="both",
    show_legend=False,
    plot_type="boxplot",
    xlabel_rot=90,
)

## Violin Plots Subplots Example

metrics_comp = ["age_group"]

box_violin_plot(
    df=df,
    metrics_list=age_boxplot_list,
    metrics_comp=metrics_comp,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    save_plots=True,
    show_plot="both",
    show_legend=False,
    plot_type="violinplot",
    xlabel_rot=90,
)

## Pivoted Violin Plots Subplots Example

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

################################################################################
## Scatter Plots and Best Fit Lines
################################################################################

print()

## Regression-Centric Scatter Plots Example

scatter_fit_plot(
    df=df,
    x_vars=["age", "education-num"],
    y_vars=["hours-per-week"],
    show_legend=True,
    show_plot="subplots",
    subplot_figsize=None,
    label_fontsize=14,
    tick_fontsize=12,
    add_best_fit_line=True,
    scatter_color="#808080",
    show_correlation=True,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    save_plots="all",
)

## Scatter Plots Grouped by Category Example

hue_dict = {"<=50K": "brown", ">50K": "green"}

print()

scatter_fit_plot(
    df=df,
    x_vars=["age", "education-num"],
    y_vars=["hours-per-week"],
    show_legend=True,
    show_plot="subplots",
    label_fontsize=14,
    tick_fontsize=12,
    add_best_fit_line=False,
    scatter_color="#808080",
    hue="income",
    hue_palette=hue_dict,
    show_correlation=False,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    save_plots="all",
)

## Scatter Plots (All Combinations Example)

## Reload dataframe since a ton of `data_doctor` columns were added from above;
## this would cause too many additional columns to loop through; not necessary
## for this reproducible example, especially when it comes to saving so many files
df = pd.read_csv(os.path.join(data_path, "adult_income.csv"))
df = df.drop(columns=["Unnamed: 0"])

print()

scatter_fit_plot(
    df=df,
    all_vars=df.select_dtypes(np.number).columns.to_list(),
    show_legend=True,
    show_plot="subplots",
    label_fontsize=14,
    tick_fontsize=12,
    add_best_fit_line=True,
    scatter_color="#808080",
    show_correlation=True,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    save_plots="all",
)

################################################################################
## Correlation Matrices
################################################################################

## Triangular Correlation Matrix Example

# Select only numeric data to pass into the function
df_num = df.select_dtypes(np.number)

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
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    save_plots=True,
)

## Full Correlation Matrix Example

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
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    save_plots=True,
)


## Outcome CrossTab Plot Examples
### Clean up income column by removing trailing period

df["income"] = df["income"].astype(str)
## Clean target column by removing trailing period
df.loc[:, "income"] = df["income"].str.rstrip(".")

# Display class balance
print(f"\nBreakdown of y:\n{df['income'].value_counts()}\n")


bar_list = ["race", "sex"]


from eda_toolkit import outcome_crosstab_plot

outcome_crosstab_plot(
    df=df,
    list_name=bar_list,
    label_0="<=50k",
    label_1=">50K",
    figsize=(10, 6),
    normalize=False,
    image_path_svg=image_path_svg,
    image_path_png=image_path_png,
    string="outcome_by_feature",
    save_plots=True,
    outcome="income",
    show_value_counts=True,
    # color_schema=surg_tech_color,
)

################################################################################
## Conditional Histograms
################################################################################

features = [
    "age",
    "education-num",
    "hours-per-week",
    "capital-gain",
    "capital-loss",
]

conditional_histograms(
    df=df,
    features=features,
    by="income",
    bins=30,
    normalize="density",
    alpha=0.6,
    # figsize=(10, 6),
    plot_style="density",
    label_fontsize=10,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    image_filename="conditional_histograms_adult_income",
    # tick_fontsize=16,
    # text_wrap=10,
)

################################################################################
## Distribution Goodness-of-Fit Plots
################################################################################

from eda_toolkit import distribution_gof_plots

distribution_gof_plots(
    df,
    var="age",
    dist=["norm", "lognorm", "gamma"],
    plot_types="qq",
    qq_type="theoretical",
    show_reference=True,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    palette={
        "norm": "tab:blue",
        "lognorm": "tab:orange",
        "gamma": "tab:green",
    },
    image_filename="gof_qq_age_adult_income",
    xlim=(5, 100),
    ylim=(5, 100),
)
distribution_gof_plots(
    df,
    var="age",
    dist="norm",
    plot_types="qq",
    ylim=(0, 100),
    xlim=(0, 100),
)

### 2. Multiple distributions, theoretical QQ
# What this shows:

# - Multiple fitted distributions overlaid
# - Same sample, same axis
# - Color auto-assigned by Matplotlib

distribution_gof_plots(
    df,
    var="age",
    dist=["norm", "lognorm", "gamma"],
    plot_types="qq",
)
### 3. Theoretical QQ with explicit color palette
# What this shows:

# - Deterministic, user-controlled colors
# - One color per distribution
# - Preferred API over positional `color=`

distribution_gof_plots(
    df,
    var="age",
    dist=["norm", "lognorm", "gamma"],
    plot_types="qq",
    palette={
        "norm": "red",
        "lognorm": "blue",
        "gamma": "green",
    },
)

### 4. Empirical QQ plot (sample vs reference data)
# What this shows:

# - Sample quantiles vs reference empirical distribution
# - Distribution name still used for labeling
# - No theoretical distribution involved in the QQ geometry

reference_data = df.loc[df["sex"] == "Male", "age"].dropna().values

distribution_gof_plots(
    df,
    var="age",
    dist="norm",
    plot_types="qq",
    qq_type="empirical",
    reference_data=reference_data,
)
### 5. Empirical QQ with multiple fitted distributions (labeling only)
# What this shows:

# - Same empirical QQ geometry
# - Multiple overlays for visual comparison
# - Distribution names are semantic labels, not geometry drivers

reference_data = df.loc[df["sex"] == "Female", "age"].dropna().values

distribution_gof_plots(
    df,
    var="age",
    dist=["norm", "lognorm"],
    plot_types="qq",
    qq_type="empirical",
    reference_data=reference_data,
    palette={
        "norm": "purple",
        "lognorm": "orange",
    },
)

### 6. CDF plot (both tails)
# What this shows:

# - CDF and exceedance probability
# - Full distribution behavior
# - Linear scale

distribution_gof_plots(
    df,
    var="age",
    dist=["norm", "lognorm"],
    plot_types="cdf",
)
### 7. Lower-tail CDF only
# What this shows:

# - Focus on lower tail behavior
# - Useful for minimum-risk analysis

distribution_gof_plots(
    df,
    var="age",
    dist=["norm", "gamma"],
    plot_types="cdf",
    tail="lower",
)
### 8. Upper-tail exceedance probability only

# What this shows:

# - Exceedance probability
# - Heavy-tail diagnostics
# - Risk modeling use cases

distribution_gof_plots(
    df,
    var="age",
    dist=["norm", "gamma"],
    plot_types="cdf",
    tail="upper",
)
### 9. Combined QQ + CDF diagnostics
# What this shows:

# - Side-by-side diagnostics
# - QQ for shape
# - CDF for tail behavior

distribution_gof_plots(
    df,
    var="age",
    dist=["norm", "lognorm", "gamma"],
    plot_types=["qq", "cdf"],
    show_reference=False,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    image_filename="gof_qq_cdf_age_adult_income",
)
### 10. Log-scaled QQ plot
# What this shows:

# - Emphasizes upper-tail deviations
# - Useful for skewed distributions
# - Log scaling applied to quantile axis

distribution_gof_plots(
    df,
    var="age",
    dist=["lognorm", "gamma"],
    plot_types="qq",
    scale="log",
)
### 11. Method-of-Moments fitting
# What this shows:

# - Alternative fitting strategy
# - Shape sensitivity comparison
# - Educational contrast vs MLE

distribution_gof_plots(
    df,
    var="age",
    dist=["norm", "gamma"],
    fit_method="MM",
    plot_types=["qq", "cdf"],
)

input("Press ENTER to quit...")
