import pandas as pd
import numpy as np
import random
from itertools import combinations
import datetime
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import os
import warnings


################################################################################
############################# Path Directories #################################


def ensure_directory(path):
    """Ensure that the directory exists. If not, create it."""
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
    column_name="Patient_ID",
    seed=None,
):
    """
    Add a column of unique, 9-digit IDs to the dataframe.

    This function sets a random seed and then generates a 9-digit ID for
    each row in the dataframe. The new IDs are added as a new column with
    the specified column name, which is placed as the first column in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add IDs to.
        column_name (str): The name of the new column for the IDs.
        seed (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
        pd.DataFrame: The updated dataframe with the new ID column.
    """
    random.seed(seed)

    # Generate a list of unique IDs
    ids = ["".join(random.choices("0123456789", k=9)) for _ in range(len(df))]

    # Create a new column in df for these IDs
    df[column_name] = ids

    # Make the new ID column the first column and set it to index
    df = df.set_index(column_name)

    return df


################################################################################
################################# Trailing Periods #############################
################################################################################


def strip_trailing_period(
    df,
    column_name,
):
    """
    Strip the trailing period from floats in a specified column of a DataFrame, if present.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the column to be processed.

    column_name : str
        The name of the column containing floats with potential trailing periods.

    Returns:
    --------
    pd.DataFrame
        The updated DataFrame with the trailing periods removed from the specified column.
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
            return datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y-%m-%d")
        except ValueError:
            return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")


################################################################################
############################## Data Types Reports ##############################
################################################################################


def data_types(df):
    """
    This function provides a data types report on every column in the dataframe,
    showing column names, column data types, number of nulls, and percentage
    of nulls, respectively.
    Inputs:
        df: dataframe to run the datatypes report on
    Outputs:
        dat_type: report saved out to a dataframe showing column name,
                  data type, count of null values in the dataframe, and
                  percentage of null values in the dataframe
    """
    # Features' Data Types and Their Respective Null Counts
    dat_type = df.dtypes

    # create a new dataframe to inspect data types
    dat_type = pd.DataFrame(dat_type)

    # sum the number of nulls per column in df
    dat_type["Null_Values"] = df.isnull().sum()

    # reset index w/ inplace = True for more efficient memory usage
    dat_type.reset_index(inplace=True)

    # percentage of null values is produced and cast to new variable
    dat_type["perc_null"] = round(dat_type["Null_Values"] / len(df) * 100, 0)

    # columns are renamed for a cleaner appearance
    dat_type = dat_type.rename(
        columns={
            0: "Data Type",
            "index": "Column/Variable",
            "Null_Values": "# of Nulls",
            "perc_null": "Percent Null",
        }
    )

    return dat_type


################################################################################


def dataframe_columns(df):
    """
    Function to analyze dataframe columns, such as dtype, null,
    and max unique value and percentages.
    Args:
        df (dataframe): the dataframe to analyze
    Raises:
        No Raises
        Null and empty string pre-processing
    Returns:
        str:       Prints the shape of the dataframe at top
        dataframe: column_value_counts list in DataFrame format
    """
    print("Shape: ", df.shape, "\n")
    start_time = datetime.datetime.now()
    # Convert dbdate dtype to datetime
    for col in df.columns:
        if df[col].dtype == "dbdate":
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Null pre-processing with Pandas NA
    df = df.fillna(pd.NA)
    # Replace empty strings with Pandas NA
    df = df.apply(
        lambda col: col.map(lambda x: pd.NA if isinstance(x, str) and x == "" else x)
    )
    # Begin Process...
    columns_value_counts = []
    for col in df.columns:
        col_str = df[col].astype(str).replace("<NA>", "null").replace("NaT", "null")
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
    stop_time = datetime.datetime.now()
    print("Total seconds of processing time:", (stop_time - start_time).total_seconds())
    return pd.DataFrame(columns_value_counts)


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
    variables in the dataframe and saves them to an Excel file.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - variables (list): List of unique variables to generate combinations.
    - data_path (str): Path where the output Excel file will be saved.
    - data_name (str): Name of the output Excel file.
    - min_length (int): Minimum length of combinations to generate. Defaults to 2.

    Returns:
    - summary_tables (dict): Dictionary of summary tables.
    - all_combinations (list): List of all generated combinations.
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
    legend_df = pd.DataFrame({"Sheet Name": sheet_names, "Description": descriptions})

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


def save_dataframes_to_excel(file_path, df_dict, decimal_places=2):
    """
    Save multiple DataFrames to separate sheets in an Excel file with customized
    formatting.

    Parameters:
    ----------
    file_path : str
        Full path to the output Excel file.
    df_dict : dict
        Dictionary where keys are sheet names and values are DataFrames to save.
    decimal_places : int, optional
        Number of decimal places to round numeric columns. Default is 2.

    Notes:
    -----
    - The function will autofit columns and left-align text.
    - Numeric columns will be formatted with the specified number of decimal places.
    - Headers will be bold and left-aligned without borders.
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

        # Customize number format
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
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]

            # Write header with custom format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Auto-fit all columns with added space
            for col_num, col_name in enumerate(df.columns):
                max_length = max(df[col_name].astype(str).map(len).max(), len(col_name))
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


def contingency_table(df, col1, col2, SortBy):
    """
    Function to create contingency table from one or two columns in dataframe,
    with sorting options.

    Args:
        df (dataframe): the dataframe to analyze
        col1 (str): name of the first column in the dataframe to include
        col2 (str): name of the second column in the dataframe to include
                    if no second column, enter "None"
        SortBy (str): enter 'Group' to sort results by col1 + col2 group
                    any other value will sort by col1 + col2 group totals

    Raises:
        No Raises

    Returns:
        dataframe: dataframe with three columns; 'Groups', 'GroupTotal', and 'GroupPct'
    """
    if col2 != "None":
        group_cols = [col1, col2]
    else:
        group_cols = [col1]

    # Create the contingency table with observed=True
    cont_df = (
        df.groupby(group_cols, observed=True).size().reset_index(name="GroupTotal")
    )

    # Calculate the percentage
    cont_df["GroupPct"] = 100 * cont_df["GroupTotal"] / len(df)

    # Sort values based on provided SortBy parameter
    if SortBy == "Group":
        cont_df = cont_df.sort_values(by=group_cols)
    else:
        cont_df = cont_df.sort_values(by="GroupTotal", ascending=False)

    # Results for all groups
    all_groups = pd.DataFrame(
        [
            {
                **{col: "All" for col in group_cols},
                "GroupTotal": cont_df["GroupTotal"].sum(),
                "GroupPct": cont_df["GroupPct"].sum(),
            }
        ]
    )

    # Combine results
    c_table = pd.concat([cont_df, all_groups], ignore_index=True)

    # Update GroupPct to reflect as a percentage rounded to 2 decimal places
    c_table["GroupPct"] = c_table["GroupPct"].round(2)

    return c_table


################################################################################
############################## Highlight DF Tables #############################
################################################################################


def highlight_columns(df, columns, color="yellow"):
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
    dist_list,
    x,
    y,
    kde=True,
    n_rows=1,
    n_cols=1,
    w_pad=1.0,
    h_pad=1.0,
    text_wrap=50,
    image_path_png=None,
    image_path_svg=None,
    image_filename=None,
    bbox_inches=None,
    vars_of_interest=None,  # List of variables of interest
    single_var_image_path_png=None,
    single_var_image_path_svg=None,
    single_var_image_filename=None,
    y_axis="count",  # Parameter to control y-axis ('count' or 'density')
    plot_type="both",  # Parameter to control the plot type ('hist', 'kde', or 'both')
):
    """
    Generate KDE or histogram distribution plots for specified columns in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.

    dist_list : list of str
        List of column names for which to generate distribution plots.

    x : int or float
        Width of the overall figure.

    y : int or float
        Height of the overall figure.

    kde : bool, optional (default=True)
        Whether to include KDE plots on the histograms.

    n_rows : int, optional (default=1)
        Number of rows in the subplot grid.

    n_cols : int, optional (default=1)
        Number of columns in the subplot grid.

    w_pad : float, optional (default=1.0)
        Width padding between subplots.

    h_pad : float, optional (default=1.0)
        Height padding between subplots.

    text_wrap : int, optional (default=50)
        Maximum width of the title text before wrapping.

    image_path_png : str, optional
        Directory path to save the PNG image of the overall distribution plots.

    image_path_svg : str, optional
        Directory path to save the SVG image of the overall distribution plots.

    image_filename : str, optional
        Filename to use when saving the overall distribution plots.

    bbox_inches : str, optional
        Bounding box to use when saving the figure. For example, 'tight'.

    vars_of_interest : list of str, optional
        List of column names for which to generate separate distribution plots.

    single_var_image_path_png : str, optional
        Directory path to save the PNG images of the separate distribution plots.

    single_var_image_path_svg : str, optional
        Directory path to save the SVG images of the separate distribution plots.

    single_var_image_filename : str, optional
        Filename to use when saving the separate distribution plots.
        The variable name will be appended to this filename.

    y_axis : str, optional (default='count')
        The type of y-axis to display ('count' or 'density').

    plot_type : str, optional (default='both')
        The type of plot to generate ('hist', 'kde', or 'both').

    Returns:
    --------
    None
    """

    if not dist_list:
        print("Error: No distribution list provided.")
        return

    y_axis = y_axis.lower()
    if y_axis not in ["count", "density"]:
        raise ValueError('y_axis can either be "count" or "density"')

    plot_type = plot_type.lower()
    if plot_type not in ["hist", "kde", "both"]:
        raise ValueError('plot_type can either be "hist", "kde", or "both"')

    # Calculate the number of plots
    num_plots = len(dist_list)
    total_slots = n_rows * n_cols

    # Create subplots grid
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(x, y))

    # Flatten the axes array to simplify iteration
    axes = axes.flatten()

    # Iterate over the provided column list and corresponding axes
    for ax, col in zip(axes[:num_plots], dist_list):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Wrap the title if it's too long
            title = f"Distribution of {col}"

            if plot_type == "hist" or plot_type == "both":
                sns.histplot(
                    df[col],
                    kde=False if plot_type == "hist" else kde,
                    ax=ax,
                    stat="density" if y_axis == "density" else "count",
                )
            if plot_type == "kde":
                sns.kdeplot(df[col], ax=ax, fill=True)
            elif plot_type == "both":
                sns.kdeplot(df[col], ax=ax)

            ax.set_ylabel("Density" if y_axis == "density" else "Count")
            ax.set_title("\n".join(textwrap.wrap(title, width=text_wrap)))

    # Hide any remaining axes
    for ax in axes[num_plots:]:
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
            fig, ax = plt.subplots(figsize=(x, y))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                title = f"Distribution of {var}"

                if plot_type == "hist" or plot_type == "both":
                    sns.histplot(
                        df[var],
                        kde=False if plot_type == "hist" else kde,
                        ax=ax,
                        stat="density" if y_axis == "density" else "count",
                    )
                if plot_type == "kde":
                    sns.kdeplot(df[var], ax=ax, fill=True)
                elif plot_type == "both":
                    sns.kdeplot(df[var], ax=ax)

                ax.set_ylabel("Density" if y_axis == "density" else "Count")
                ax.set_title("\n".join(textwrap.wrap(title, width=text_wrap)))

            plt.tight_layout()

            # Save files for the variable of interest if paths are provided
            if single_var_image_path_png and single_var_image_filename:
                plt.savefig(
                    os.path.join(
                        single_var_image_path_png,
                        f"{single_var_image_filename}_{var}.png",
                    ),
                    bbox_inches=bbox_inches,
                )
            if single_var_image_path_svg and single_var_image_filename:
                plt.savefig(
                    os.path.join(
                        single_var_image_path_svg,
                        f"{single_var_image_filename}_{var}.svg",
                    ),
                    bbox_inches=bbox_inches,
                )
            plt.close(
                fig
            )  # Close the figure after saving to avoid displaying it multiple times


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
    plot_type="both",  # New parameter for plot type
    show_legend=True,
):
    """
    Generates stacked bar plots and crosstabs for specified columns.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - col (str): The name of the column in the DataFrame to be analyzed.
    - func_col (list): List of ground truth columns to be analyzed.
    - legend_labels_list (list): List of legend labels for each ground truth column.
    - title (list): List of titles for the plots.
    - kind (str, optional): The kind of plot to generate (e.g., 'bar', 'barh').
      Defaults to 'bar'.
    - width (float, optional): The width of the bars in the bar plot. Defaults to 0.9.
    - rot (int, optional): The rotation angle of the x-axis labels. Defaults to 0.
    - custom_order (list, optional): Specifies a custom order for the categories in the 'col'.
    - image_path_png (str, optional): Directory path where generated PNG plot images
      will be saved.
    - image_path_svg (str, optional): Directory path where generated SVG plot images
      will be saved.
    - save_formats (list, optional): List of file formats to save the plot images in.
    - color (list, optional): List of colors to use for the plots. If not provided,
      a default color scheme is used.
    - output (str, optional): Specify the output type: "plots_only", "crosstabs_only", or "both".
      Defaults to "both".
    - return_dict (bool, optional): Specify whether to return the crosstabs dictionary.
      Defaults to False.
    - x (int, optional): The width of the figure.
    - y (int, optional): The height of the figure.
    - p (int, optional): The padding between the subplots.
    - file_prefix (str, optional): Prefix for the filename when output includes plots.
    - logscale (bool, optional): Apply log scale to the y-axis. Defaults to False.
    - plot_type (str, optional): Specify the type of plot to generate: "both", "regular",
      "normalized". Defaults to "both".
    - show_legend (bool, optional): Specify whether to show the legend. Defaults to True.

    Returns:
    - crosstabs_dict (dict): Dictionary of crosstabs DataFrames if return_dict is True.
    - None: If return_dict is False.
    """
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

    # Check if the lengths of title, func_col, and legend_labels_list match
    if not (len(title) == len(func_col) == len(legend_labels_list)):
        raise ValueError(
            "Length mismatch: Ensure that the lengths of title, func_col, and legend_labels_list are equal. "
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
            raise ValueError("file_prefix must be provided when output includes plots")
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
            if image_path_png and image_path_svg:
                func_col_filename_png = os.path.join(
                    image_path_png, f"{file_prefix}_{truth}.png"
                )
                func_col_filename_svg = os.path.join(
                    image_path_svg, f"{file_prefix}_{truth}.svg"
                )
                image_path = {
                    "png": func_col_filename_png,
                    "svg": func_col_filename_svg,
                }
            else:
                image_path = {}

            # Verify the DataFrame state before creating plots
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(x, y))
            fig.tight_layout(w_pad=5, pad=p, h_pad=5)

            # Define crosstabdest to avoid UnboundLocalError
            crosstabdest = pd.crosstab(df_copy[col], df_copy[truth])
            try:
                crosstabdest.columns = legend  # Rename the columns to match the legend
            except ValueError:
                raise ValueError(
                    f"Length mismatch: Crosstab columns ({len(crosstabdest.columns)}) and legend ({len(legend)}). "
                    "Check the length of your legend_labels_list, func_col, and title to make sure you are not missing an item or comma or have an extra item."
                )

            if plot_type in ["both", "regular"]:
                # Plot the first graph (absolute counts)
                title1 = f"Prevalence of {tit} by {col.replace('_', ' ').title()}"
                xlabel1 = f"{col.replace('_', ' ')}"
                ylabel1 = "Count"
                crosstabdest.plot(
                    kind=kind,
                    stacked=True,
                    title=title1,
                    ax=axes[0] if plot_type == "both" else axes,
                    color=color,
                    width=width,
                    rot=rot,
                    fontsize=12,
                    logy=logscale,  # Apply log scale if logscale is True
                )
                ax0 = axes[0] if plot_type == "both" else axes
                ax0.set_title(title1, fontsize=12)
                ax0.set_xlabel(xlabel1, fontsize=12)
                ax0.set_ylabel(ylabel1, fontsize=12)
                if show_legend:
                    ax0.legend(legend, fontsize=12)
                else:
                    ax0.legend().remove()

            if plot_type in ["both", "normalized"]:
                # Plotting the second, normalized stacked bar graph
                title2 = f"Prevalence of {tit} by {col.replace('_', ' ').title()} (Normalized)"
                xlabel2 = f"{col.replace('_', ' ')}"
                ylabel2 = "Percentage"
                crosstabdestnorm = crosstabdest.div(crosstabdest.sum(1), axis=0)
                crosstabdestnorm.plot(
                    kind=kind,
                    stacked=True,
                    title=title2,
                    ylabel="Percentage",
                    ax=axes[1] if plot_type == "both" else axes,
                    color=color,
                    width=width,
                    rot=rot,
                    fontsize=12,
                    logy=logscale,  # Apply log scale if logscale is True
                )
                ax1 = axes[1] if plot_type == "both" else axes
                ax1.set_title(label=title2, fontsize=12)
                ax1.set_xlabel(xlabel2, fontsize=12)
                ax1.set_ylabel(ylabel2, fontsize=12)
                if show_legend:
                    ax1.legend(legend, fontsize=12)
                else:
                    ax1.legend().remove()

            fig.align_ylabels()
            if save_formats and isinstance(image_path, dict):
                for save_format in save_formats:
                    if save_format in image_path:
                        full_path = image_path[save_format]
                        plt.savefig(full_path, bbox_inches="tight")
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
            crosstabs_dict[col_results] = crosstab_df  # Use col_results as the key

    # Return the crosstabs_dict only if return_dict is True
    if return_dict:
        return crosstabs_dict


################################################################################


def plot_filtered_dataframes(
    df,
    col,
    func_col,
    legend_labels_list,
    title,
    file_prefix,
    condition_col=None,
    condition_val=1,
    x=12,
    y=8,
    p=10,
    kind="bar",
    width=0.9,
    rot=0,
    image_path_png=None,
    image_path_svg=None,
    save_formats=["png", "svg"],
    color=None,
    output="both",
    return_dict=True,
    logscale=True,
    plot_type="both",
    show_legend=True,
):
    """
    Filters the DataFrame based on a specified condition and generates plots and crosstabs.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - col (str): The name of the column in the DataFrame to be analyzed.
    - func_col (list): List of ground truth columns to be analyzed.
    - legend_labels_list (list): List of legend labels for each ground truth column.
    - title (list): List of titles for the plots.
    - file_prefix (str): Prefix for the filename when output includes plots.
    - condition_col (str, optional): Column to skip filtering.
    - condition_val (int, optional): Value to filter the DataFrame on. Defaults to 1.
    - x (int, optional): The width of the figure.
    - y (int, optional): The height of the figure.
    - p (int, optional): The padding between the subplots.
    - kind (str, optional): The kind of plot to generate (e.g., 'bar', 'barh').
      Defaults to 'bar'.
    - width (float, optional): The width of the bars in the bar plot. Defaults to 0.9.
    - rot (int, optional): The rotation angle of the x-axis labels. Defaults to 0.
    - image_path_png (str, optional): Directory path where generated PNG plot
      images will be saved.
    - image_path_svg (str, optional): Directory path where generated SVG plot
      images will be saved.
    - save_formats (list, optional): List of file formats to save the plot images in.
    - color (list, optional): List of colors to use for the plots. If not provided,
      a default color scheme is used.
    - output (str, optional): Specify the output type: "plots_only", "crosstabs_only",
      or "both". Defaults to "both".
    - return_dict (bool, optional): Specify whether to return the crosstabs dictionary.
      Defaults to True.
    - logscale (bool, optional): Apply log scale to the y-axis. Defaults to True.
    - plot_normalized (bool, optional): Specify whether to plot normalized stacked bar graph.
      Defaults to True.
    - show_legend (bool, optional): Specify whether to show the legend. Defaults to True.

    Returns:
    - results (list): List of crosstabs dictionaries if return_dict is True.
    - None: If return_dict is False.
    """
    # Initialize results list to store crosstabs
    results = []
    # Loop through each condition and create the filtered DataFrame
    for fc, ll, t in zip(func_col, legend_labels_list, title):
        # Filter the DataFrame based on the condition if condition_col is provided
        if condition_col and fc != condition_col:
            filtered_df = df[df[fc] == condition_val]
        else:
            filtered_df = df
        # Call the main plotting function
        result = stacked_crosstab_plot(
            filtered_df,
            col=col,
            func_col=[fc],
            legend_labels_list=[ll],
            title=[t],
            file_prefix=f"{file_prefix}_{fc}",
            x=x,
            y=y,
            p=p,
            kind=kind,
            width=width,
            rot=rot,
            image_path_png=image_path_png,
            image_path_svg=image_path_svg,
            save_formats=save_formats,
            color=color,
            output=output,
            return_dict=return_dict,
            logscale=logscale,
            plot_type=plot_type,
            show_legend=show_legend,
        )
        # Append results to the list if return_dict is True
        if return_dict:
            results.append(result)

    # Return the results list if return_dict is True
    if return_dict:
        return results


################################################################################


def metrics_box_violin(
    df,
    metrics_list,
    metrics_boxplot_comp,
    n_rows,
    n_cols,
    image_path_png,
    image_path_svg,
    save_individual=True,
    save_grid=True,
    save_both=False,
    show_legend=True,  # New parameter to toggle legend
    plot_type="boxplot",  # New parameter to specify plot type
):
    """
    Create and save individual boxplots or violin plots, an entire grid of plots, or both for
    given metrics and comparisons.

    Parameters:
    - df: DataFrame containing the data.
    - metrics_list: List of metric names (columns in df) to plot.
    - metrics_boxplot_comp: List of comparison categories (columns in df).
    - n_rows: Number of rows in the subplot grid.
    - n_cols: Number of columns in the subplot grid.
    - image_path_png: Directory path to save .png images.
    - image_path_svg: Directory path to save .svg images.
    - save_individual: Boolean, True if saving each subplot as an individual file.
    - save_grid: Boolean, True if saving the entire grid as one image.
    - save_both: Boolean, True if saving both individual and grid images.
    - show_legend: Boolean, True if showing the legend in the plots.
    - plot_type: String, "boxplot" or "violinplot" to specify the type of plot.
    """
    # Ensure the directories exist

    if save_both:
        save_individual = True
        save_grid = True

    def get_palette(n_colors):
        return sns.color_palette("tab10", n_colors=n_colors)

    # Map plot_type to the corresponding seaborn function
    plot_function = getattr(sns, plot_type)

    # Save individual plots if required
    if save_individual:
        for met_comp in metrics_boxplot_comp:
            unique_vals = df[met_comp].value_counts().count()
            palette = get_palette(unique_vals)
            for met_list in metrics_list:
                plt.figure(figsize=(6, 4))  # Adjust the size as needed
                ax = plot_function(
                    x=met_comp,
                    y=met_list,
                    data=df,
                    hue=met_comp,
                    palette=palette,
                    dodge=False,
                )
                plt.title(f"Distribution of {met_list} by {met_comp}")
                plt.xlabel(met_comp)
                plt.ylabel(met_list)

                # Toggle legend
                if not show_legend and ax.legend_:
                    ax.legend_.remove()

                safe_met_list = (
                    met_list.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_per_")
                )
                filename_png = f"{safe_met_list}_by_{met_comp}.png"
                filename_svg = f"{safe_met_list}_by_{met_comp}.svg"
                plt.savefig(
                    os.path.join(image_path_png, filename_png), bbox_inches="tight"
                )
                plt.savefig(
                    os.path.join(image_path_svg, filename_svg), bbox_inches="tight"
                )
                plt.close()

    # Save the entire grid if required
    if save_grid:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axs = axs.flatten()

        for i, ax in enumerate(axs):
            if i < len(metrics_list) * len(metrics_boxplot_comp):
                met_comp = metrics_boxplot_comp[i // len(metrics_list)]
                met_list = metrics_list[i % len(metrics_list)]
                unique_vals = df[met_comp].value_counts().count()
                palette = get_palette(unique_vals)
                plot_function(
                    x=met_comp,
                    y=met_list,
                    data=df,
                    hue=met_comp,
                    ax=ax,
                    palette=palette,
                    dodge=False,
                )
                ax.set_title(f"Distribution of {met_list} by {met_comp}")
                ax.set_xlabel(met_comp)
                ax.set_ylabel(met_list)

                # Toggle legend
                if not show_legend and ax.legend_:
                    ax.legend_.remove()
            else:
                ax.set_visible(False)

        plt.tight_layout()
        fig.savefig(
            os.path.join(image_path_png, "all_plots_comparisons.png"),
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(image_path_svg, "all_plots_comparisons.svg"),
            bbox_inches="tight",
        )
        plt.show()  # show the plot(s)
        plt.close(fig)
