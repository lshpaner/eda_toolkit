################################################################################
############################### Library Imports ################################

import pandas as pd
import numpy as np
import math
import random
import itertools  # Import itertools for combinations
from itertools import combinations
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker  # Import for formatting
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
import textwrap
import os
import sys
import warnings
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

if sys.version_info >= (3, 7):
    from datetime import datetime
else:
    import datetime


################################################################################
############################# Path Directories #################################
################################################################################


def ensure_directory(path):
    """
    Ensure that the directory exists. If not, create it.
    """

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory exists: {path}")


################################################################################
############################ Generate Random IDs ###############################
################################################################################


def add_ids(
    df,
    id_colname="ID",
    num_digits=9,
    seed=None,
    set_as_index=False,
):
    """
    Add a column of unique IDs with specified number of digits to the dataframe.

    This function generates a unique ID with the specified number of digits for
    each row in the dataframe. The new IDs are added as a new column with the
    specified column name. Optionally, the new ID column can be set as the index
    or placed as the first column in the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to add IDs to.

    id_colname : str, optional (default="ID")
        The name of the new column for the IDs.

    num_digits : int, optional (default=9)
        The number of digits for the unique IDs.

    seed : int, optional
        The seed for the random number generator. Defaults to None.

    set_as_index : bool, optional (default=False)
        Whether to set the new ID column as the index. Defaults to False.

    Returns:
    --------
    pd.DataFrame
        The updated dataframe with the new ID column.

    Notes:
    ------
    - If the dataframe index is not unique, a warning is printed.
    - The function does not check if the number of rows exceeds the number of
      unique IDs that can be generated with the specified number of digits.
    - The first digit of the generated IDs is ensured to be non-zero.
    """

    # Check for unique indices
    if df.index.duplicated().any():
        print("Warning: DataFrame index is not unique.")
        print(
            "Duplicate index entries:",
            df.index[df.index.duplicated()].tolist(),
        )
    else:
        print("DataFrame index is unique.")

    random.seed(seed)

    # Ensure the first digit is non-zero
    def generate_id():
        first_digit = random.choice("123456789")
        other_digits = "".join(random.choices("0123456789", k=num_digits - 1))
        return first_digit + other_digits

    # Generate a set of unique IDs
    ids = set()
    while len(ids) < len(df):
        new_ids = {generate_id() for _ in range(len(df) - len(ids))}
        ids.update(new_ids)

    # Convert the set of unique IDs to a list
    ids = list(ids)

    # Create a new column in df for these IDs
    df[id_colname] = ids

    if set_as_index:
        # Optionally set the new ID column as the index
        df = df.set_index(id_colname)
    else:
        # Ensure the new ID column is the first column
        columns = [id_colname] + [col for col in df.columns if col != id_colname]
        df = df[columns]

    return df


################################################################################
################################# Trailing Periods #############################
################################################################################


def strip_trailing_period(
    df,
    column_name,
):
    """
    Strip the trailing period from floats in a specified column of a DataFrame,
    if present.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the column to be processed.

    column_name : str
        The name of the column containing floats with potential trailing periods.

    Returns:
    --------
    pd.DataFrame
        The updated DataFrame with the trailing periods removed from the
        specified column.
    """

    def fix_value(value):
        value_str = str(value)
        if value_str.endswith("."):
            value_str = value_str.rstrip(".")
        return float(value_str)

    df[column_name] = df[column_name].apply(fix_value)

    return df


################################################################################
########################### Standardized Dates #################################
################################################################################


# Function to parse and standardize date strings based on the new rule
def parse_date_with_rule(date_str):
    """
    Parse and standardize date strings based on the provided rule.

    This function takes a date string and standardizes it to the ISO 8601 format
    (YYYY-MM-DD). It assumes dates are provided in either day/month/year or
    month/day/year format. The function first checks if the first part of the
    date string (day or month) is greater than 12, which unambiguously indicates
    a day/month/year format. If the first part is 12 or less, the function
    attempts to parse the date as month/day/year, falling back to day/month/year
    if the former raises a ValueError due to an impossible date (e.g., month
    being greater than 12).

    Parameters:
        date_str (str): A date string to be standardized.

    Returns:
        str: A standardized date string in the format YYYY-MM-DD.

    Raises:
        ValueError: If date_str is in an unrecognized format or if the function
        cannot parse the date.
    """

    parts = date_str.split("/")
    # If the first part is greater than 12, it can only be a day, thus d/m/Y
    if int(parts[0]) > 12:
        return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
    # Otherwise, try both formats where ambiguity exists
    else:
        try:
            return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
        except ValueError:
            return datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y-%m-%d")


################################################################################
############################### DataFrame Columns ##############################
################################################################################


def dataframe_columns(
    df,
    background_color=None,
    return_df=False,
):
    """
    Analyze DataFrame columns to provide summary statistics such as data type,
    null counts, unique values, and most frequent values.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        background_color (str, optional): Hex color code or color name for
                                          background styling in the output
                                          DataFrame. Defaults to None.
        return_df (bool, optional): If True, returns the plain DataFrame with
                                    the summary statistics. If False, returns a
                                    styled DataFrame for visual presentation.
                                    Defaults to False.

    Raises:
        None.

    Returns:
        pandas.DataFrame: If `return_df` is True, returns the plain DataFrame
                          containing column summary statistics. If `return_df`
                          is False, returns a styled DataFrame with optional
                          background color for specific columns.

    Example:
        styled_df = dataframe_columns(df, background_color="#FFFF00")
        plain_df = dataframe_columns(df, return_df=True)
    """

    print("Shape: ", df.shape, "\n")
    start_time = (
        datetime.now() if sys.version_info >= (3, 7) else datetime.datetime.now()
    )

    # Convert dbdate dtype to datetime
    for col in df.columns:
        if df[col].dtype == "dbdate":
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Null pre-processing with Pandas NA
    df = df.fillna(pd.NA)
    # Replace empty strings with Pandas NA
    df = df.apply(
        lambda col: col.map(
            lambda x: pd.NA if isinstance(x, str) and x == "" else x,
        )
    )
    # Begin Process...
    columns_value_counts = []
    for col in df.columns:
        col_str = (
            df[col]
            .astype(str)
            .replace("<NA>", "null")
            .replace(
                "NaT",
                "null",
            )
        )
        value_counts = col_str.value_counts()
        max_unique_value = value_counts.index[0]
        max_unique_value_total = value_counts.iloc[0]
        columns_value_counts.append(
            {
                "column": col,
                "dtype": df[col].dtype,
                "null_total": df[col].isnull().sum(),
                "null_pct": round(df[col].isnull().sum() / df.shape[0] * 100, 2),
                "unique_values_total": df[col].nunique(),
                "max_unique_value": max_unique_value,
                "max_unique_value_total": max_unique_value_total,
                "max_unique_value_pct": round(
                    max_unique_value_total / df.shape[0] * 100, 2
                ),
            }
        )
    stop_time = (
        datetime.now() if sys.version_info >= (3, 7) else datetime.datetime.now()
    )
    print(
        "Total seconds of processing time:",
        (stop_time - start_time).total_seconds(),
    )

    result_df = pd.DataFrame(columns_value_counts)

    if return_df:
        # Return the plain DataFrame
        return result_df
    else:
        # Return the styled DataFrame

        # Output, try/except, accounting for the potential of Python version with
        # the styler as hide_index() is deprecated since Pandas 1.4, in such cases,
        # hide() is used instead
        try:
            return (
                pd.DataFrame(columns_value_counts)
                .style.hide()
                .format(precision=2)
                .set_properties(
                    subset=[
                        "unique_values_total",
                        "max_unique_value",
                        "max_unique_value_total",
                        "max_unique_value_pct",
                    ],
                    **{"background-color": background_color},
                )
            )
        except:
            return (
                pd.DataFrame(columns_value_counts)
                .style.hide_index()
                .format(precision=2)
                .set_properties(
                    subset=[
                        "unique_values_total",
                        "max_unique_value",
                        "max_unique_value_total",
                        "max_unique_value_pct",
                    ],
                    **{"background-color": background_color},
                )
            )


################################################################################
############################ Summarize All Combinations ########################
################################################################################


def summarize_all_combinations(
    df,
    variables,
    data_path,
    data_name,
    min_length=2,
):
    """
    Generates summary tables for all possible combinations of the specified
    variables in the DataFrame and saves them to an Excel file.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    variables : list of str
        List of column names from the DataFrame to generate combinations.
    data_path : str
        Path where the output Excel file will be saved.
    data_name : str
        Name of the output Excel file.
    min_length : int, optional (default=2)
        Minimum size of the combinations to generate.

    Returns:
    --------
    summary_tables : dict
        A dictionary where keys are tuples of column names (combinations) and
        values are the corresponding summary DataFrames.
    all_combinations : list of tuple
        A list of all generated combinations, where each combination is
        represented as a tuple of column names.

    Notes:
    ------
    - The function will create an Excel file with a sheet for each combination
      of the specified variables, as well as a "Table of Contents" sheet with
      hyperlinks to each summary table.
    - The sheet names are limited to 31 characters due to Excel's constraints.
    """

    summary_tables = {}
    grand_total = len(df)
    all_combinations = []

    df_copy = df.copy()

    for i in range(min_length, len(variables) + 1):
        for combination in combinations(variables, i):
            all_combinations.append(combination)
            for col in combination:
                df_copy[col] = df_copy[col].astype(str)

            count_df = (
                df_copy.groupby(list(combination)).size().reset_index(name="Count")
            )
            count_df["Proportion"] = (count_df["Count"] / grand_total * 100).fillna(0)

            summary_tables[tuple(combination)] = count_df

    sheet_names = [
        ("_".join(combination)[:31]) for combination in summary_tables.keys()
    ]
    descriptions = [
        "Summary for " + ", ".join(combination) for combination in summary_tables.keys()
    ]
    legend_df = pd.DataFrame(
        {"Sheet Name": sheet_names, "Description": descriptions},
    )

    file_path = f"{data_path}/{data_name}"
    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        # Write the Table of Contents (legend sheet)
        legend_df.to_excel(writer, sheet_name="Table of Contents", index=False)

        workbook = writer.book
        toc_worksheet = writer.sheets["Table of Contents"]

        # Add hyperlinks to the sheet names
        for i, sheet_name in enumerate(sheet_names, start=2):
            cell = f"A{i}"
            toc_worksheet.write_url(cell, f"#'{sheet_name}'!A1", string=sheet_name)

        # Set column widths and alignment for Table of Contents
        toc_worksheet.set_column("A:A", 50)  # Set width for column A (Sheet Name)
        toc_worksheet.set_column("B:B", 100)  # Set width for column B (Description)

        # Create a format for left-aligned text
        cell_format = workbook.add_format({"align": "left"})
        toc_worksheet.set_column("A:A", 50, cell_format)  # Column A
        toc_worksheet.set_column("B:B", 100, cell_format)  # Column B

        # Format the header row of Table of Contents
        header_format_toc = workbook.add_format(
            {"bold": True, "align": "left", "border": 0}
        )
        toc_worksheet.write_row("A1", legend_df.columns, header_format_toc)

        # Define a format with no borders for the header row in other sheets
        header_format_no_border = workbook.add_format(
            {"bold": True, "border": 0, "align": "left"}
        )

        # Define a format for left-aligned text in other sheets
        left_align_format = workbook.add_format({"align": "left"})

        # Format the summary tables
        for sheet_name, table in summary_tables.items():
            sheet_name_str = "_".join(sheet_name)[
                :31
            ]  # Ensure sheet name is <= 31 characters
            table.to_excel(writer, sheet_name=sheet_name_str, index=False)

            worksheet = writer.sheets[sheet_name_str]

            # Apply format to the header row (top row)
            for col_num, col_name in enumerate(table.columns):
                worksheet.write(0, col_num, col_name, header_format_no_border)

            # Apply left alignment to all columns
            for row_num in range(1, len(table) + 1):
                for col_num in range(len(table.columns)):
                    worksheet.write(
                        row_num,
                        col_num,
                        table.iloc[row_num - 1, col_num],
                        left_align_format,
                    )

            # Auto-fit all columns with added space
            for col_num, col_name in enumerate(table.columns):
                max_length = max(
                    table[col_name].astype(str).map(len).max(), len(col_name)
                )
                worksheet.set_column(
                    col_num, col_num, max_length + 2, left_align_format
                )  # Add extra space

    print(f"Data saved to {file_path}")

    return summary_tables, all_combinations


################################################################################
############################ Save DataFrames to Excel ##########################
################################################################################


def save_dataframes_to_excel(
    file_path,
    df_dict,
    decimal_places=0,
):
    """
    Save multiple DataFrames to separate sheets in an Excel file with customized
    formatting, including column autofit and numeric formatting.

    Parameters:
    -----------
    file_path : str
        Full path to the output Excel file.

    df_dict : dict
        Dictionary where keys are sheet names and values are DataFrames to save
        to those sheets.

    decimal_places : int, optional (default=0)
        Number of decimal places to round numeric columns. If set to 0, numeric
        columns will be saved as integers.

    Notes:
    ------
    - Columns will be autofitted to content and left-aligned.
    - Numeric columns will be formatted with the specified number of decimal
      places.
    - Headers will be bold, left-aligned, and have no borders.
    - This function requires the 'xlsxwriter' library for Excel writing.

    Example:
    --------
    df1 = pd.DataFrame({"A": [1, 2], "B": [3.14159, 2.71828]})
    df2 = pd.DataFrame({"X": ["foo", "bar"], "Y": ["baz", "qux"]})
    df_dict = {"Sheet1": df1, "Sheet2": df2}
    save_dataframes_to_excel("output.xlsx", df_dict, decimal_places=2)
    """

    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Customize header format (remove borders)
        header_format = workbook.add_format(
            {
                "bold": True,
                "text_wrap": True,
                "valign": "top",
                "border": 0,  # Remove borders
                "align": "left",  # Left align
            }
        )

        # Customize cell format (left align)
        cell_format_left = workbook.add_format({"align": "left"})  # Left align

        # Customize number format based on decimal_places
        if decimal_places == 0:
            number_format_str = "0"
            cell_format_number = workbook.add_format(
                {
                    "align": "left",
                    "num_format": number_format_str,
                }  # Left align  # Number format
            )
        else:
            number_format_str = f"0.{decimal_places * '0'}"
            cell_format_number = workbook.add_format(
                {
                    "align": "left",
                    "num_format": number_format_str,
                }  # Left align  # Number format
            )

        # Write each DataFrame to its respective sheet
        for sheet_name, df in df_dict.items():
            # Round numeric columns to the specified number of decimal places
            df = df.round(decimal_places)
            if decimal_places == 0:
                df = df.apply(
                    lambda x: x.astype(int) if pd.api.types.is_numeric_dtype(x) else x
                )
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]

            # Write header with custom format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Auto-fit all columns with added space
            for col_num, col_name in enumerate(df.columns):
                max_length = max(
                    df[col_name].astype(str).map(len).max(),
                    len(col_name),
                )
                # Determine if the column is numeric by dtype
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    worksheet.set_column(
                        col_num, col_num, max_length + 2, cell_format_number
                    )
                else:
                    worksheet.set_column(
                        col_num, col_num, max_length + 2, cell_format_left
                    )

    print(f"DataFrames saved to {file_path}")


################################################################################
############################## Contingency Table ###############################
################################################################################


def contingency_table(
    df,
    cols=None,
    sort_by=0,
):
    """
    Create a contingency table from one or more columns in a DataFrame, with
    options to sort the results.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to analyze.

    cols : str or list of str, optional
        The name of a single column (as a string) or a list of column names
        for multiple columns. At least one column must be provided.

    sort_by : int, optional (default=0)
        Sorting option for the results. Enter 0 to sort the results by the
        column group(s) specified in `cols`, or enter 1 to sort the results
        by the 'Total' column in descending order.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the contingency table with the specified columns,
        a 'Total' column representing the count of occurrences, and a
        'Percentage' column representing the percentage of the total count.

    Raises:
    -------
    ValueError
        If no columns are specified in `cols`, or if `sort_by` is not 0 or 1.

    """

    # Ensure at least one column is specified
    if not cols or (isinstance(cols, list) and not cols):
        raise ValueError("At least one DataFrame column must be specified.")

    # Ensure sort_by is either 0 or 1
    if sort_by not in [0, 1]:
        raise ValueError("`sort_by` must be 0 or 1.")

    # Convert single column to list
    if isinstance(cols, str):
        cols = [cols]

    # Convert categorical columns to string to avoid fillna issue
    for col in cols:
        if df[col].dtype.name == "category":
            df[col] = df[col].astype(str)

    # Convert all values in dataframe to object
    # then fill NA values in the dataframe with empty spaces
    df = df.astype(str).fillna("")

    # Create the contingency table with observed=True
    cont_df = (
        df.groupby(cols, observed=True)
        .size()
        .reset_index(
            name="Total",
        )
    )

    # Calculate the percentage
    cont_df["Percentage"] = 100 * cont_df["Total"] / len(df)

    # Sort values based on provided sort_by parameter
    if sort_by == 0:
        cont_df = cont_df.sort_values(by=cols)
    elif sort_by == 1:
        cont_df = cont_df.sort_values(by="Total", ascending=False)

    # Convert categorical columns to string to avoid fillna issue
    cont_df[cols] = cont_df[cols].astype(str)

    # Results for all groups
    all_groups = pd.DataFrame(
        [
            {
                **{col: "" for col in cols},
                "Total": cont_df["Total"].sum(),
                "Percentage": cont_df["Percentage"].sum(),
            }
        ]
    )

    # Combine results
    c_table = pd.concat(
        [cont_df.fillna(""), all_groups.fillna("")],
        ignore_index=True,
    )

    # Update GroupPct to reflect as a percentage rounded to 2 decimal places
    c_table["Percentage"] = c_table["Percentage"].round(2)

    return c_table


################################################################################
############################## Highlight DF Tables #############################
################################################################################


def highlight_columns(
    df,
    columns,
    color="yellow",
):
    """
    Highlight specific columns in a DataFrame with a specified background color.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be styled.
    columns : list of str
        List of column names to be highlighted.
    color : str, optional
        The background color to be applied for highlighting (default is "yellow").

    Returns:
    --------
    pandas.io.formats.style.Styler
        A Styler object with the specified columns highlighted.
    """

    def highlight(s):
        return [
            f"background-color: {color}" if col in columns else "" for col in s.index
        ]

    return df.style.apply(highlight, axis=1)


################################################################################
############################ KDE Distribution Plots ############################
################################################################################


def kde_distributions(
    df,
    vars_of_interest=None,
    figsize=(5, 5),  # Unified figsize parameter
    grid_figsize=None,  # Size of the overall grid
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

    grid_figsize : tuple of int, optional
        Size of the overall grid of plots when there are multiple variables
        being plotted in a single grid. This parameter is ignored when only one
        variable is plotted or when using `single_var_image_filename`.

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
        individual plots. The `grid_figsize` param. is ignored in this context.

    y_axis_label : str, optional (default='Density')
        The label to display on the y-axis.

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
        Maximum width of the title text before wrapping.

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
        If `grid_figsize` is provided when only one plot is being created.

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
        if grid_figsize is not None:
            raise ValueError(
                f"Cannot use `grid_figsize` when there is only one "
                f"plot. Use `figsize` instead."
            )
    else:
        # Calculate columns based on square root
        if n_rows is None or n_cols is None:
            n_cols = int(np.ceil(np.sqrt(num_vars)))
            n_rows = int(np.ceil(num_vars / n_cols))

        # Adjust figsize for grid if multiple plots
        if grid_figsize is None:
            figsize = (figsize[0] * n_cols, figsize[1] * n_rows)
        else:
            figsize = grid_figsize

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
            f"Not enough colors specified in `std_color`. "
            f"You have {len(std_color)} color(s) but {len(std_dev_levels)} "
            f"standard deviation level(s). "
            f"Please provide at least as many colors as standard deviation levels."
        )

    # Validate plot_type parameter
    valid_plot_types = ["hist", "kde", "both"]
    if plot_type.lower() not in valid_plot_types:
        raise ValueError(
            f"Invalid `plot_type` value. Expected one of {valid_plot_types}, "
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
            f"Invalid `stat` value. Expected one of {valid_stats}, "
            f"got '{stat}' instead."
        )

    # Check if all log_scale_vars are in the DataFrame
    if log_scale_vars:
        invalid_vars = [var for var in log_scale_vars if var not in df.columns]
        if invalid_vars:
            raise ValueError(f"Invalid `log_scale_vars`: {invalid_vars}")

    # Check if edgecolor is being set while fill is False
    if not fill and hist_edgecolor != "#000000":
        raise ValueError("Cannot change `edgecolor` when `fill` is set to False")

    # Check if fill_alpha is being set while fill is False
    if not fill and fill_alpha != 0.6:
        raise ValueError("Cannot set `fill_alpha` when `fill` is set to False")

    # Warn if both bins and binwidth are set
    if bins != "auto" and binwidth is not None:
        warnings.warn(
            "Specifying both `bins` and `binwidth` may affect performance.",
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

            # Add "(Log)" to the label if log_scale is applied
            xlabel = f"{get_label(col)} (Log)" if log_scale else get_label(col)

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
                        legend=False,  # Do not add legend automatically
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
                        legend=False,  # Do not add legend automatically
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
                        legend=False,  # Do not add legend automatically
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
                        legend=False,  # Do not add legend automatically
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

                # Conditionally add the legend
                if show_legend:
                    ax.legend(loc="best")

            except Exception as e:
                # Handle different Python versions or issues w/ legends & labels
                if "No artists with labels found to put in legend." in str(e):
                    print(f"Warning encountered while plotting '{col}': {str(e)}")
                    if show_legend:
                        ax.legend(loc="best")

            ax.set_xlabel(
                xlabel,
                fontsize=label_fontsize,
            )

            ax.set_ylabel(
                y_axis_label.capitalize(),
                fontsize=label_fontsize,
            )
            ax.set_title(
                "\n".join(textwrap.wrap(title, width=text_wrap)),
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
                            legend=False,  # Do not add legend automatically
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
                            legend=False,  # Do not add legend automatically
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
                            legend=False,  # Do not add legend automatically
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
                            legend=False,  # Do not add legend automatically
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

                    # Conditionally add the legend
                    if show_legend:
                        ax.legend(loc="best")

                except Exception as e:
                    # Handle different Python versions or issues w/ legends & labels
                    if "No artists with labels found to put in legend." in str(e):
                        print(f"Warning encountered while plotting '{var}': {str(e)}")
                        if show_legend:
                            ax.legend(loc="best")

                ax.set_xlabel(xlabel, fontsize=label_fontsize)

                ax.set_ylabel(
                    y_axis_label.capitalize(),
                    fontsize=label_fontsize,
                )
                ax.set_title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)),
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

    save_formats : list of str, optional
        List of file formats to save the plot images in. Valid formats are
        'png' and 'svg'.

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

    # Initialize the dictionary to store crosstabs
    crosstabs_dict = {}
    # Default color settings
    if color is None:
        color = ["#00BFC4", "#F8766D"]  # Default colors

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
            plt.close(fig)  # Ensure plot is closed after showing

    # Generate crosstabs if output is "both" or "crosstabs_only"
    if output in ["both", "crosstabs_only"]:
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
            # Process counter
            legend_counter += 1
            # Display results
            print("Crosstab for " + col_results)
            display(crosstab_df)
            # Store the crosstab in the dictionary
            # Use col_results as the key
            crosstabs_dict[col_results] = crosstab_df

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
    save_plots=None,  # Parameter to control saving plots
    show_legend=True,  # Parameter to toggle legend
    plot_type="boxplot",  # Parameter to specify plot type
    xlabel_rot=0,  # Parameter to rotate x-axis labels
    show_plot="both",  # Parameter to control plot display
    rotate_plot=False,  # Parameter to rotate (pivot) plots
    individual_figsize=(6, 4),
    grid_figsize=None,  # Parameter to specify figure size for grid plots
    label_fontsize=12,  # Parameter to control axis label fontsize
    tick_fontsize=10,  # Parameter to control tick label fontsize
    text_wrap=50,  # Add text_wrap parameter
    xlim=None,  # New parameter for setting x-axis limits
    ylim=None,  # New parameter for setting y-axis limits
    label_names=None,
    **kwargs,  # To allow passing additional parameters to Seaborn
):
    """
    Create and save individual boxplots or violin plots, an entire grid
    of plots, or both for given metrics and comparisons, with optional
    axis limits.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    metrics_list : list of str
        List of metric names (columns in df) to plot.

    metrics_comp : list of str
        List of comparison categories (columns in df).

    n_rows : int, optional
        Number of rows in the subplot grid. Calculated automatically
        if not provided.

    n_cols : int, optional
        Number of columns in the subplot grid. Calculated automatically
        if not provided.

    image_path_png : str, optional
        Directory path to save .png images.

    image_path_svg : str, optional
        Directory path to save .svg images.

    save_plots : str, optional
        String, "all", "individual", or "grid" to control saving plots.

    show_legend : bool, optional (default=True)
        True if showing the legend in the plots.

    plot_type : str, optional (default='boxplot')
        String, "boxplot" or "violinplot" to specify the type of plot.

    xlabel_rot : int, optional (default=0)
        Rotation angle for x-axis labels.

    show_plot : str, optional (default='both')
        String, "individual", "grid", or "both" to control plot display.

    rotate_plot : bool, optional (default=False)
        True if rotating (pivoting) the plots.

    individual_figsize : tuple or list, optional (default=(6, 4))
        Width and height of the figure for individual plots.

    grid_figsize : tuple or list, optional
        Width and height of the figure for grid plots.

    label_fontsize : int, optional (default=12)
        Font size for axis labels.

    tick_fontsize : int, optional (default=10)
        Font size for axis tick labels.

    text_wrap : int, optional (default=50)
        The maximum width of the title text before wrapping.

    xlim : tuple, optional
        Tuple specifying the limits of the x-axis.

    ylim : tuple, optional
        Tuple specifying the limits of the y-axis.

    label_names : dict, optional
        Dictionary mapping original column names to custom labels.

    **kwargs : additional keyword arguments
        Additional keyword arguments passed to the Seaborn plotting function.

    Returns:
    --------
    None
        This function does not return any value but generates and optionally
        saves boxplots or violin plots for the specified metrics and comparisons.

    Raises:
    -------
    ValueError
        If 'show_plot' is not one of "individual", "grid", or "both".

    ValueError
        If 'save_plots' is not one of None, "all", "individual", or "grid".

    ValueError
        If 'save_plots' is set without specifying 'image_path_png' or
        'image_path_svg'.

    ValueError
        If 'rotate_plot' is not a boolean value.

    ValueError
        If 'individual_figsize' is not a tuple or list of two numbers
        (width, height).

    ValueError
        If 'grid_figsize' is provided and is not a tuple or list of two numbers
        (width, height).

    Notes:
    ------
    - `n_rows` and `n_cols` are automatically calculated if not provided, based
      on the number of plots.
    - `label_names` allows you to map original column names to custom labels
      used in plot titles and labels.
    """

    # Check for valid show_plot values
    if show_plot not in ["individual", "grid", "both"]:
        raise ValueError(
            "Invalid `show_plot` value selected. Choose from 'individual', "
            "'grid', or 'both'."
        )

    # Check for valid save_plots values
    if save_plots not in [None, "all", "individual", "grid"]:
        raise ValueError(
            "Invalid `save_plots` value selected. Choose from 'all', "
            "'individual', 'grid', or None."
        )

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

    # Check for valid grid_figsize values if specified
    if grid_figsize is not None and not (
        isinstance(grid_figsize, (tuple, list))
        and len(grid_figsize) == 2
        and all(isinstance(x, (int, float)) for x in grid_figsize)
    ):
        raise ValueError(
            "Invalid `grid_figsize` value. It should be a tuple or list of two "
            "numbers (width, height)."
        )

    # Calculate n_rows and n_cols dynamically if not provided
    if n_rows is None or n_cols is None:
        total_plots = len(metrics_list) * len(metrics_comp)
        n_cols = int(np.ceil(np.sqrt(total_plots)))
        n_rows = int(np.ceil(total_plots / n_cols))

    # Set default grid figure size if not specified
    if grid_figsize is None:
        grid_figsize = (5 * n_cols, 5 * n_rows)

    # Determine saving options based on save_plots value
    save_individual = save_plots in ["all", "individual"]
    save_grid = save_plots in ["all", "grid"]

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
                    get_label(met_list) if rotate_plot else get_label(met_comp),
                    fontsize=label_fontsize,
                )
                plt.ylabel(
                    get_label(met_comp) if rotate_plot else get_label(met_list),
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

                if show_plot in ["individual", "both"]:
                    plt.show()  # Display the plot
                plt.close()

    # Save and/or show the entire grid if required
    if save_grid or show_plot in ["grid", "both"]:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=grid_figsize)
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
                    get_label(met_list) if rotate_plot else get_label(met_comp),
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    get_label(met_comp) if rotate_plot else get_label(met_list),
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
        if save_grid:
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

        if show_plot in ["grid", "both"]:
            plt.show()  # Display the plot
        plt.close(fig)


################################################################################
########################## Multi-Purpose Scatter Plots #########################
################################################################################


def scatter_fit_plot(
    df,
    x_vars=None,
    y_vars=None,
    n_rows=None,
    n_cols=None,
    max_cols=4,
    image_path_png=None,  # Make image paths optional
    image_path_svg=None,  # Make image paths optional
    save_plots=None,  # Parameter to control saving plots
    show_legend=True,  # Parameter to toggle legend
    xlabel_rot=0,  # Parameter to rotate x-axis labels
    show_plot="both",  # Parameter to control plot display
    rotate_plot=False,  # Parameter to rotate (pivot) plots
    individual_figsize=(6, 4),
    grid_figsize=None,  # Parameter to specify figure size for grid plots
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
    all_vars=None,
    label_names=None,  # New parameter for custom column renames
    **kwargs,  # Additional keyword arguments to pass to sns.scatterplot
):
    """
    Create and save scatter plots or a grid of scatter plots for given
    x_vars and y_vars, with an optional best fit line and customizable
    point color, size, and markers.

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
        Controls which plots to save: "all", "individual", or "grid".
        If None, plots will not be saved.

    show_legend : bool, optional (default=True)
        Whether to display the legend on the plots.

    xlabel_rot : int, optional (default=0)
        Rotation angle for x-axis labels.

    show_plot : str, optional (default="both")
        Controls plot display: "individual", "grid", or "both".

    rotate_plot : bool, optional (default=False)
        Whether to rotate (pivot) the plots.

    individual_figsize : tuple or list, optional (default=(6, 4))
        Width and height of the figure for individual plots.

    grid_figsize : tuple or list, optional
        Width and height of the figure for grid plots.
        If not specified, defaults to a calculated size based on the number of
        rows and columns.

    label_fontsize : int, optional (default=12)
        Font size for axis labels.

    tick_fontsize : int, optional (default=10)
        Font size for axis tick labels.

    text_wrap : int, optional (default=50)
        The maximum width of the title text before wrapping.

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

    ylim : tuple or list, optional
        Limits for the y-axis as a tuple or list of (min, max).

    all_vars : list of str, optional
        If provided, automatically generates scatter plots for all combinations
        of variables in this list, overriding x_vars and y_vars.

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
        If `show_plot` is not one of ["individual", "grid", "both"].

    ValueError
        If `save_plots` is not one of [None, "all", "individual", "grid"].

    ValueError
        If `save_plots` is set without specifying either `image_path_png` or
        `image_path_svg`.

    ValueError
        If `rotate_plot` is not a boolean value.

    ValueError
        If `individual_figsize` is not a tuple or list of two numbers
        (width, height).

    ValueError
        If `grid_figsize` is provided and is not a tuple or list of two numbers
        (width, height).

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

    # Calculate the number of plots
    num_plots = len(combinations)

    # Set a fixed number of columns and calculate rows
    if n_cols is None:
        n_cols = min(num_plots, max_cols)
    if n_rows is None:
        n_rows = math.ceil(num_plots / n_cols)

    # Set default grid figure size if not specified
    if grid_figsize is None:
        grid_figsize = (5 * n_cols, 5 * n_rows)

    # Validate the show_plot input
    if show_plot not in ["individual", "grid", "both"]:
        raise ValueError(
            f"Invalid `show_plot`. Choose 'individual', 'grid', " f"or 'both'."
        )

    # Validate the save_plots input
    if save_plots not in [None, "all", "individual", "grid"]:
        raise ValueError(
            "Invalid `save_plots` value. Choose from 'all', "
            "'individual', 'grid', or None."
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

    # Validate the grid_figsize input if specified
    if grid_figsize is not None and not (
        isinstance(grid_figsize, (tuple, list))
        and len(grid_figsize) == 2
        and all(isinstance(x, (int, float)) for x in grid_figsize)
    ):
        raise ValueError(
            "Invalid `grid_figsize` value. It should be a tuple or list of two "
            "numbers (width, height)."
        )

    # Determine saving options based on save_plots value
    save_individual = save_plots in ["all", "individual"]
    save_grid = save_plots in ["all", "grid"]

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
        ax.legend(loc="best")

    # Save and/or show individual plots if required
    if save_individual or show_plot in ["individual", "both"]:
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
                get_label(x_var) if not rotate_plot else get_label(y_var),
                fontsize=label_fontsize,
            )
            plt.ylabel(
                get_label(y_var) if not rotate_plot else get_label(x_var),
                fontsize=label_fontsize,
            )
            ax.tick_params(axis="x", rotation=xlabel_rot)
            ax.tick_params(axis="both", labelsize=tick_fontsize)

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            if not show_legend and ax.legend_:
                ax.legend().remove()

            if save_individual:
                safe_x_var = (
                    x_var.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_per_")
                )
                safe_y_var = (
                    y_var.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_per_")
                )
                if image_path_png:
                    filename_png = f"scatter_{safe_x_var}_vs_{safe_y_var}.png"
                    plt.savefig(
                        os.path.join(image_path_png, filename_png),
                        bbox_inches="tight",
                    )
                if image_path_svg:
                    filename_svg = f"scatter_{safe_x_var}_vs_{safe_y_var}.svg"
                    plt.savefig(
                        os.path.join(image_path_svg, filename_svg),
                        bbox_inches="tight",
                    )

            if show_plot in ["individual", "both"]:
                plt.show()
            plt.close()

    # Save and/or show the entire grid if required
    if save_grid or show_plot in ["grid", "both"]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=grid_figsize)

        # Flatten the axes array to simplify iteration
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < num_plots:
                x_var, y_var = combinations[i]
                sns.scatterplot(
                    x=x_var if not rotate_plot else y_var,
                    y=y_var if not rotate_plot else x_var,
                    data=df,
                    color=scatter_color,
                    hue=hue,
                    size=size,
                    sizes=sizes,
                    marker=marker,
                    ax=ax,
                    palette=hue_palette,
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
                    get_label(x_var) if not rotate_plot else get_label(y_var),
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    get_label(y_var) if not rotate_plot else get_label(x_var),
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)

                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

                if not show_legend and ax.legend_:
                    ax.legend().remove()
            else:
                ax.set_visible(False)

        plt.tight_layout()
        if save_grid:
            if image_path_png:
                fig.savefig(
                    os.path.join(image_path_png, "scatter_plots_grid.png"),
                    bbox_inches="tight",
                )
            if image_path_svg:
                fig.savefig(
                    os.path.join(image_path_svg, "scatter_plots_grid.svg"),
                    bbox_inches="tight",
                )

        if show_plot in ["grid", "both"]:
            plt.show()
        plt.close(fig)


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
    title="Cervical Cancer Data: Correlation Matrix",
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
    **kwargs,
):
    """
    Creates a correlation heatmap with enhanced customization and options to
    save the plots in specified formats.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    cols : list of str, optional
        List of column names to include in the correlation matrix.
        If None, all columns are included.

    annot : bool, optional (default=True)
        Whether to annotate the heatmap with correlation coefficients.

    cmap : str, optional (default='coolwarm')
        The colormap to use for the heatmap.

    save_plots : bool, optional (default=False)
        Controls whether to save the plots.

    image_path_png : str, optional
        Directory path to save PNG image of the heatmap.

    image_path_svg : str, optional
        Directory path to save SVG image of the heatmap.

    figsize : tuple, optional (default=(10, 10))
        Width and height of the figure for the heatmap.

    title : str, optional
        Title of the heatmap.

    label_fontsize : int, optional (default=12)
        Font size for the axis labels and title.

    tick_fontsize : int, optional (default=10)
        Font size for tick labels (variable names) and colorbar label.

    xlabel_rot : int, optional (default=45)
        Rotation angle for x-axis labels.

    ylabel_rot : int, optional (default=0)
        Rotation angle for y-axis labels.

    xlabel_alignment : str, optional (default="right")
        Horizontal alignment for x-axis labels (e.g., "center", "right").

    ylabel_alignment : str, optional (default="center_baseline")
        Vertical alignment for y-axis labels (e.g., "center", "top").

    text_wrap : int, optional (default=50)
        The maximum width of the title text before wrapping.

    vmin : float, optional
        Minimum value for the heatmap color scale.

    vmax : float, optional
        Maximum value for the heatmap color scale.

    cbar_label : str, optional (default='Correlation Index')
        Label for the colorbar.

    triangular : bool, optional (default=True)
        Whether to show only the upper triangle of the correlation matrix.

    label_names : dict, optional
        Dictionary to map original column names to custom labels.

    **kwargs : dict, optional
        Additional keyword arguments to pass to seaborn.heatmap().

    Returns:
    --------
    None
        This function does not return any value but generates and optionally
        saves a correlation heatmap.

    Raises:
    -------
    ValueError
        If `annot` is not a boolean value.

    ValueError
        If `cols` is provided and is not a list of column names.

    ValueError
        If `save_plots` is not a boolean value.

    ValueError
        If `triangular` is not a boolean value.

    ValueError
        If `save_plots` is set to True without specifying either
        `image_path_png` or `image_path_svg`.
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
    plt.figure(figsize=figsize)

    # Draw the heatmap with the mask and correct aspect ratio
    heatmap = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    # Set the font size for the colorbar label
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fontsize)  # Updated to use tick_fontsize
    cbar.set_label(cbar_label, fontsize=label_fontsize)

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
                label_names.get(label.get_text(), label.get_text())
                for label in heatmap.get_xticklabels()
            ],
            rotation=xlabel_rot,
            fontsize=tick_fontsize,
            ha=xlabel_alignment,
            rotation_mode="anchor",
        )
        heatmap.set_yticklabels(
            [
                label_names.get(label.get_text(), label.get_text())
                for label in heatmap.get_yticklabels()
            ],
            rotation=ylabel_rot,
            fontsize=tick_fontsize,
            va=ylabel_alignment,
        )
    else:
        # Rotate x-axis labels, adjust alignment, and apply padding
        plt.xticks(
            rotation=xlabel_rot,
            fontsize=tick_fontsize,  # Updated to use tick_fontsize
            ha=xlabel_alignment,
            rotation_mode="anchor",
        )

        # Rotate y-axis labels and adjust alignment
        plt.yticks(
            rotation=ylabel_rot,
            fontsize=tick_fontsize,  # Updated to use tick_fontsize
            va=ylabel_alignment,
        )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot if save_plots is True
    if save_plots:
        safe_title = title.replace(" ", "_").replace(":", "").lower()

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
