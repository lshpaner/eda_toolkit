# EDA Toolkit

Welcome to EDA Toolkit, a comprehensive collection of utility functions designed to streamline your exploratory data analysis (EDA) tasks. This repository provides tools for directory management, data preprocessing, reporting, visualization, and more, helping you efficiently manage various aspects of data manipulation and analysis.

---

## Table of Contents

1. [Installation](#installation)
2. [Overview](#overview)
3. [Functions](#functions)  
    a. [Path Directories](#path-directories)  
    b. [Generate Random IDs](#generate-random-ids)  
    c. [Trailing Periods](#trailing-periods)  
    d. [Standardized Dates](#standardized-dates)  
    e. [Data Types Reports](#data-types-reports)  
    f. [DataFrame Columns Analysis](#dataframe-columns-analysis)  
    g. [Summarize All Combinations](#summarize-all-combinations)  
    h. [Save DataFrames to Excel](#save-dataframes-to-excel)  
    i. [Contingency Table](#contingency-table)  
    j. [Highlight DataFrame Tables](#highlight-dataframe-tables)  
    k. [KDE Distribution Plots](#kde-distribution-plots)  
    l. [Stacked Bar Plots with Crosstab Options](#stacked-bar-plots-with-crosstab-options)  
    m. [Box and Violin Plots](#box-and-violin-plots)  
    n. [Multi-Purpose Scatter Plots](#multi-purpose-scatter-plots)  
    
4. [Usage Examples](#usage-examples)  
    - [KDE Distribution Plots](#kde-distribution-plots)  
    - [Stacked Bar Plots with Crosstab Options](#stacked-bar-plots-with-crosstab-options)  
    - [Box and Violin Plots](#box-and-violin-plots)  
    - [Multi-Purpose Scatter Plots](#multi-purpose-scatter-plots)  
5. [Overall Usage](#usage)    
    - [Use the functions as needed](#use-the-functions-as-needed-in-your-data-analysis-workflow)  
    - [Import the module and functions](#import-the-module-and-functions)  
6. [Contributors/Maintainers](#contributorsmaintainers)
7. [Contributing](#contributing)
8. [License](#license)
9. [References](#references)

---

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/lshpaner/eda_toolkit.git
cd eda_toolkit
pip install -r requirements.txt
```

## Overview

EDA Toolkit is designed to be a comprehensive toolkit for data analysts and data scientists alike. It offers a suite of functions to handle common EDA tasks, making your workflow more efficient and organized. The toolkit covers everything from directory management and ID generation to complex visualizations and data reporting.


## Functions
### Path Directories

- `ensure_directory(path)`: Ensures that the specified directory exists; if not, it creates it.

### Generate Random IDs
- `add_ids(df, id_colname="ID", num_digits=9, seed=None, set_as_index=False)`: Adds a column of unique, 9-digit IDs to a DataFrame.

### Trailing Periods
- `strip_trailing_period(df, column_name)`: Strips trailing periods from floats in a specified column of a DataFrame.

### Standardized Dates
- `parse_date_with_rule(date_str)`: Parses and standardizes date strings to the `ISO 8601` format (`YYYY-MM-DD`).

### Data Types Reports
- `data_types(df)`: Provides a report on every column in the DataFrame, showing column names, data types, number of nulls, and percentage of nulls.

### DataFrame Columns Analysis
`dataframe_columns(df)`: Analyzes DataFrame columns for `dtype`, `null` counts, `max` unique values, and their percentages.

### Summarize All Combinations
- `summarize_all_combinations(df, variables, data_path, data_name, min_length=2)`: Generates summary tables for all possible combinations of specified variables in the DataFrame and saves them to an Excel file.

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/summarize_combos.gif)


### Save DataFrames to Excel
- `save_dataframes_to_excel(file_path, df_dict, decimal_places=0)`: Saves multiple DataFrames to separate sheets in an Excel file with customized formatting.

### Contingency Table
- `contingency_table(df, cols=None, sort_by=0)`: Creates a contingency table from one or more columns in a DataFrame, with sorting options.

### Highlight DataFrame Tables
- `highlight_columns(df, columns, color="yellow")`: Highlights specific columns in a DataFrame with a specified background color.

## Usage Examples

The following examples utilize the Census Income Data (1994) from the UCI Machine Learning Repository [2]. This dataset provides a rich source of information for demonstrating the functionalities of the eda_toolkit.


### KDE Distribution Plots

```python
from eda_toolkit import kde_distributions

kde_distributions(
    df,
    vars_of_interest=None,
    grid_figsize=(10, 8),
    single_figsize=(6, 4),
    kde=True,
    hist_color="#0000FF",
    kde_color="#FF0000",
    hist_edgecolor="#000000",
    hue=None,
    fill=True,
    fill_alpha=1,
    n_rows=1,
    n_cols=1,
    w_pad=1.0,
    h_pad=1.0,
    text_wrap=50,
    image_path_png=None,
    image_path_svg=None,
    image_filename=None,
    bbox_inches=None,
    single_var_image_path_png=None,
    single_var_image_path_svg=None,
    single_var_image_filename=None,
    y_axis_label="Density",
    plot_type="both",
    log_scale_vars=None,
    bins="auto",
    binwidth=None,
    label_fontsize=10,
    tick_fontsize=10,
    disable_sci_notation=False,
    stat="density",
    xlim=None,
    ylim=None,
)

```

Generates KDE and/or histogram distribution plots for specified columns in a DataFrame.

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/kde_density_distributions.svg)

### Stacked Bar Plots with Crosstab Options

```python
from eda_toolkit import stacked_crosstab_plot

stacked_crosstab_plot(
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
    remove_stacks=False,
    xlim=None,
    ylim=None,
)
```

Generates stacked or regular bar plots and crosstabs for specified columns.

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/Stacked_Bar_Age_sex.svg)

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/Stacked_Bar_Age_income.svg)

**Crosstab for sex**

| sex         | Female | Male  | Total | Female_% | Male_% |
|-------------|--------|-------|-------|----------|--------|
| **age_group** |        |       |       |          |        |
| < 18        | 295    | 300   | 595   | 49.58    | 50.42  |
| 18-29       | 5707   | 8213  | 13920 | 41       | 59     |
| 30-39       | 3853   | 9076  | 12929 | 29.8     | 70.2   |
| 40-49       | 3188   | 7536  | 10724 | 29.73    | 70.27  |
| 50-59       | 1873   | 4746  | 6619  | 28.3     | 71.7   |
| 60-69       | 939    | 2115  | 3054  | 30.75    | 69.25  |
| 70-79       | 280    | 535   | 815   | 34.36    | 65.64  |
| 80-89       | 40     | 91    | 131   | 30.53    | 69.47  |
| 90-99       | 17     | 38    | 55    | 30.91    | 69.09  |
| **Total**   | 16192  | 32650 | 48842 | 33.15    | 66.85  |

**Crosstab for income**

| income      | <=50K  | >50K  | Total | <=50K_%  | >50K_% |
|-------------|--------|-------|-------|----------|--------|
| **age_group** |        |       |       |          |        |
| < 18        | 595    | 0     | 595   | 100      | 0      |
| 18-29       | 13174  | 746   | 13920 | 94.64    | 5.36   |
| 30-39       | 9468   | 3461  | 12929 | 73.23    | 26.77  |
| 40-49       | 6738   | 3986  | 10724 | 62.83    | 37.17  |
| 50-59       | 4110   | 2509  | 6619  | 62.09    | 37.91  |
| 60-69       | 2245   | 809   | 3054  | 73.51    | 26.49  |
| 70-79       | 668    | 147   | 815   | 81.96    | 18.04  |
| 80-89       | 115    | 16    | 131   | 87.79    | 12.21  |
| 90-99       | 42     | 13    | 55    | 76.36    | 23.64  |
| **Total**   | 37155  | 11687 | 48842 | 76.07    | 23.93  |



### Box and Violin Plots
```python
from eda_toolkit import box_violin_plot

box_violin_plot(
    df,
    metrics_list,
    metrics_boxplot_comp,
    n_rows,
    n_cols,
    image_path_png=None,
    image_path_svg=None,
    save_plots=None,
    show_legend=True,
    plot_type="boxplot",
    xlabel_rot=0,
    show_plot="grid",
    rotate_plot=False,
    individual_figsize=(6, 4),
    grid_figsize=None,
    label_fontsize=12,
    tick_fontsize=10,
    xlim=None,
    ylim=None,
)
```

Creates and saves individual boxplots or violin plots, or an entire grid of plots 
for given metrics and comparisons, with optional axis limits.

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/all_plots_comparisons_boxplot.svg)

```python
from eda_toolkit import box_violin_plot

box_violin_plot(
    df,
    metrics_list,
    metrics_boxplot_comp,
    n_rows,
    n_cols,
    image_path_png=None,
    image_path_svg=None,
    save_plots=None,
    show_legend=True,
    plot_type="violinplot",
    xlabel_rot=0,
    show_plot="grid",
    rotate_plot=False,
    individual_figsize=(6, 4),
    grid_figsize=None,
    label_fontsize=12,
    tick_fontsize=10,
    xlim=None,
    ylim=None,
)
```

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/all_plots_comparisons_violinplot.svg)

### Multi-Purpose Scatter Plots

```python
from eda_toolkit import scatter_fit_plot

scatter_fit_plot(
    df=df,
    x_vars=["age", "education-num"],
    y_vars=["hours-per-week"],
    n_rows=3,
    n_cols=4,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    save_plots="grid",
    show_legend=True,
    xlabel_rot=0,
    show_plot="grid",
    rotate_plot=False,
     grid_figsize=None,
    label_fontsize=14,
    tick_fontsize=12,
    add_best_fit_line=True,
    scatter_color="#808080",
    show_correlation=True,
)
```

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/scatter_plots_grid.svg)

Creates and saves scatter plots or a grid of scatter plots for given `x_vars` and `y_vars`, with an optional best fit line and customizable point `color`, `size`, and `markers`.

```python
from eda_toolkit import scatter_fit_plot

hue_dict = {"<=50K": "brown", ">50K": "green"}

scatter_fit_plot(
    df=df,
    x_vars=["age", "education-num"],
    y_vars=["hours-per-week"],
    n_rows=3,
    n_cols=4,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    save_plots="grid",
    show_legend=True,
    xlabel_rot=0,
    show_plot="grid",
    rotate_plot=False,
    grid_figsize=None,
    label_fontsize=14,
    tick_fontsize=12,
    add_best_fit_line=False,
    scatter_color="#808080",
    hue="income",
    hue_palette=hue_dict,
    show_correlation=False,
)
```
![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/scatter_plots_grid_grouped.svg)


## Overall Usage

### Import the Module and Functions

```python
import pandas as pd
import numpy as np
import random
from itertools import combinations
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import textwrap
import os
import sys
import warnings

# Import the utility functions from EDA Toolkit
from eda_toolkit import (
    ensure_directory,
    add_ids,
    strip_trailing_period,
    parse_date_with_rule,
    data_types,
    dataframe_columns,
    summarize_all_combinations,
    save_dataframes_to_excel,
    contingency_table,
    highlight_columns,
    kde_distributions,
    stacked_crosstab_plot,
    plot_filtered_dataframes,
    box_violin_plot,
    scatter_fit_plot,
)
```

### Use the Functions as Needed in Your Data Analysis Workflow

```python
# Example usage of ensure_directory function
directory_path = "path/to/save/directory"
ensure_directory(directory_path)

# Example usage of add_ids function
df = pd.DataFrame({"Name": ["Alice", "Bob", "Charlie"]})
df_with_ids = add_ids(df)

# Example usage of kde_distributions function
kde_distributions(
    df,
    vars_of_interest=["Age", "Income"],
    grid_figsize=(10, 8),
    single_figsize=(6, 4),
    kde=True,
    hist_color="#0000FF",
    kde_color="#FF0000",
    hist_edgecolor="#000000",
    fill=True,
    fill_alpha=0.6,
    n_rows=2,
    n_cols=1,
    y_axis_label="Density",
)
```

## Contributors/Maintainers

<img align="left" width="150" height="150" src="https://www.leonshpaner.com/author/leon-shpaner/avatar_hu48de79c369d5f7d4ff8056a297b2c4c5_1681850_270x270_fill_q90_lanczos_center.jpg">

[Leonid Shpaner](https://github.com/lshpaner) is a Data Scientist at UCLA Health. With over a decade experience in analytics and teaching, he has collaborated on a wide variety of projects within financial services, education, personal development, and healthcare. He serves as a course facilitator for Data Analytics and Applied Statistics at Cornell University and is a lecturer of Statistics in Python for the University of San Diego's M.S. Applied Artificial Intelligence program.  

<br>
<br>
<br>

<img align="left" width="150" height="150" src="https://oscargildata.com/portfolio_content/images/Oscar_LinkedIn_Pic.jpeg">

[Oscar Gil](https://github.com/Oscar-Gil-Data) is a Data Scientist at the University of California, Riverside, bringing over ten years of professional experience in the education data management industry. An effective data professional, he excels in Data Warehousing, Data Analytics, Data Wrangling, Machine Learning, SQL, Python, R, Data Automation, and Report Authoring. Oscar holds a Master of Science in Applied Data Science from the University of San Diego.

<br>
<br>

## Contributing
We welcome contributions! If you have suggestions or improvements, please submit an issue or pull request. Follow the standard GitHub flow for contributing.


## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/lshpaner/eda_toolkit/blob/readme/LICENSE.md) file for details.

For more detailed documentation, refer to the docstrings within each function.

## References

1. Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. *Computing in Science & Engineering*, 9(3), 90-95. [https://doi.org/10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)

2. Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5GP7S](https://doi.org/10.24432/C5GP7S).

3. Waskom, M. (2021). *Seaborn: Statistical Data Visualization*. *Journal of Open Source Software*, 6(60), 3021. [https://doi.org/10.21105/joss.03021](https://doi.org/10.21105/joss.03021).



