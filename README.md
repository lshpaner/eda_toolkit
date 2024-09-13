[![PyPI](https://img.shields.io/pypi/v/eda_toolkit.svg)](https://pypi.org/project/eda_toolkit/)
[![Downloads](https://pepy.tech/badge/eda_toolkit)](https://pepy.tech/project/eda_toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/lshpaner/eda_toolkit/blob/main/LICENSE.md)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.13162633.svg)](https://doi.org/10.5281/zenodo.13162633)

<br>

<img src="https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/eda_toolkit_logo.svg" width="300" style="border: none; outline: none; box-shadow: none;" oncontextmenu="return false;">

<br> 

# EDA Toolkit

Welcome to EDA Toolkit, a collection of utility functions designed to streamline your exploratory data analysis (EDA) tasks. This repository offers tools for directory management, some data preprocessing, reporting, visualizations, and more, helping you efficiently handle various aspects of data manipulation and analysis.

---

## Table of Contents

1. [Documentation](#documentation)
2. [Installation](#installation)
3. [Overview](#overview)
4. [Functions](#functions)  
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
    
5. [Usage Examples](#usage-examples)  
    - [KDE Distribution Plots](#kde-distribution-plots)  
    - [Stacked Bar Plots with Crosstab Options](#stacked-bar-plots-with-crosstab-options)  
    - [Box and Violin Plots](#box-and-violin-plots)  
    - [Multi-Purpose Scatter Plots](#multi-purpose-scatter-plots)  
6. [Overall Usage](#usage)    
    - [Use the functions as needed](#use-the-functions-as-needed-in-your-data-analysis-workflow)  
    - [Import the module and functions](#import-the-module-and-functions)  
7. [Contributors/Maintainers](#contributorsmaintainers)
8. [Contributing](#contributing)
9. [License](#license)
10. [Citing EDA Toolkit](#citing-eda_toolkit)
11. [References](#references)

---

## Documentation

https://lshpaner.github.io/eda_toolkit  

## Installation

Clone the repository and install the necessary dependencies:

```bash
pip install eda_toolkit
```

## Overview

EDA Toolkit is designed to be a comprehensive toolkit for data analysts and data scientists alike. It offers a suite of functions to handle common EDA tasks, making your workflow more efficient and organized. The toolkit covers everything from directory management and ID generation to complex visualizations and data reporting.


## Functions

Use the Functions as Needed in Your Data Analysis Workflow

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

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/kde_density_distributions.png)

### Stacked Bar Plots with Crosstab Options

Generates stacked or regular bar plots and crosstabs for specified columns.

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/Stacked_Bar_Age_sex.png)

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/Stacked_Bar_Age_income.png)

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

Creates and saves individual boxplots or violin plots, or an entire grid of plots 
for given metrics and comparisons, with optional axis limits.

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/all_plots_comparisons_boxplot.png)

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/all_plots_comparisons_violinplot.png)


### Multi-Purpose Scatter Plots

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/scatter_plots_grid.png)

Creates and saves scatter plots or a grid of scatter plots for given `x_vars` and `y_vars`, with an optional best fit line and customizable point `color`, `size`, and `markers`.

![](https://raw.githubusercontent.com/lshpaner/eda_toolkit/main/assets/scatter_plots_grid_grouped.png)


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

## Citing `eda_toolkit`

If you use `eda_toolkit` in your research or projects, please consider citing it.

```bibtex

@software{shpaner_2024_13162633,
  author       = {Shpaner, Leonid and
                  Gil, Oscar},
  title        = {EDA Toolkit},
  month        = aug,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.0.8d},
  doi          = {10.5281/zenodo.13162633},
  url          = {https://doi.org/10.5281/zenodo.13162633}
}

```

## References

1. Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. *Computing in Science & Engineering*, 9(3), 90-95. [https://doi.org/10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)

2. Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5GP7S](https://doi.org/10.24432/C5GP7S).

3. Pace, R. Kelley, & Barry, R. (1997). *Sparse Spatial Autoregressions*. *Statistics & Probability Letters*, 33(3), 291-297. [https://doi.org/10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X).

4. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). *Scikit-learn: Machine Learning in Python*. *Journal of Machine Learning Research*, 12, 2825-2830. [http://jmlr.org/papers/v12/pedregosa11a.html](http://jmlr.org/papers/v12/pedregosa11a.html).

5. Waskom, M. (2021). *Seaborn: Statistical Data Visualization*. *Journal of Open Source Software*, 6(60), 3021. [https://doi.org/10.21105/joss.03021](https://doi.org/10.21105/joss.03021).



