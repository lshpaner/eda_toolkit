################################################################################
########################### Import Requisite Lbraries ##########################
################################################################################

import pandas as pd
import os

from eda_toolkit import (
    ensure_directory,
    read_csv_with_progress,
    strip_trailing_period,
    strip_trailing_period,
    add_ids,
    parse_date_with_rule,
    dataframe_profiler,
    summarize_all_combinations,
    save_dataframes_to_excel,
    contingency_table,
    generate_table1,
)

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")

################################################################################
################################ Ensure Directory ##############################
################################################################################
# Get the width of the terminal
try:
    terminal_width = os.get_terminal_size().columns
except OSError:
    terminal_width = 80

base_path = os.path.join(os.pardir)

# Go up one level from 'notebooks' to parent directory,
# then into the 'data' folder
data_path = os.path.join(os.pardir, "data")
data_output = os.path.join(os.pardir, "data_output")

# create image paths
image_path_png = os.path.join(base_path, "images", "png_images")
image_path_svg = os.path.join(base_path, "images", "svg_images")

print()
# Use the function to ensure'data' directory exists
ensure_directory(data_path)
ensure_directory(data_output)
ensure_directory(image_path_png)
ensure_directory(image_path_svg)

print()
print("*" * terminal_width)

################################################################################
############################### UCI ML Repository ##############################
################################################################################

from ucimlrepo import fetch_ucirepo

print("Adult Income Dataset From UCI Machine Learning Repository")
# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# Combine X and y into entire df
df = X.join(y, how="inner")

df.to_csv(os.path.join(data_path, "adult_income.csv"))

print(df.head())
print("*" * terminal_width)

################################################################################
### 1. read_csv_with_progress — basic
################################################################################

print("\n[1] read_csv_with_progress — basic read")
df = read_csv_with_progress(os.path.join(data_path, "adult_income.csv"))
df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
print(f"  Shape: {df.shape}")
print(df.head(3))
print("*" * terminal_width)

################################################################################
### 2. read_csv_with_progress — nrows
################################################################################

print("\n[2] read_csv_with_progress — nrows=500")
df_small = read_csv_with_progress(
    os.path.join(data_path, "adult_income.csv"),
    nrows=500,
)
df_small.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
assert len(df_small) == 500, f"Expected 500 rows, got {len(df_small)}"
print(f"  Shape: {df_small.shape}")
print("*" * terminal_width)

################################################################################
### 3. read_csv_with_progress — custom chunksize
################################################################################

print("\n[3] read_csv_with_progress — chunksize=5000")
df_chunked = read_csv_with_progress(
    os.path.join(data_path, "adult_income.csv"),
    chunksize=5000,
)
df_chunked.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
print(f"  Shape: {df_chunked.shape}")
print("*" * terminal_width)

################################################################################
### 4. read_csv_with_progress — usecols kwarg
################################################################################

print("\n[4] read_csv_with_progress — usecols kwarg")
df_cols = read_csv_with_progress(
    os.path.join(data_path, "adult_income.csv"),
    usecols=["age", "education", "income"],
)
print(f"  Columns: {list(df_cols.columns)}")
print(f"  Shape: {df_cols.shape}")
print("*" * terminal_width)

################################################################################
### 5. read_csv_with_progress — dtype kwarg
################################################################################

print("\n[5] read_csv_with_progress — dtype kwarg")
df_typed = read_csv_with_progress(
    os.path.join(data_path, "adult_income.csv"),
    dtype={"age": float, "hours-per-week": float},
)
df_typed.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
print(f"  age dtype: {df_typed['age'].dtype}")
print(f"  hours-per-week dtype: {df_typed['hours-per-week'].dtype}")
print("*" * terminal_width)

################################################################################
### 6. read_csv_with_progress — chunksize in kwargs raises ValueError
################################################################################

print("\n[6] read_csv_with_progress — chunksize in kwargs raises ValueError")
try:
    read_csv_with_progress(
        os.path.join(data_path, "adult_income.csv"),
        chunksize=10000,
        **{"chunksize": 5000},  # should raise
    )
    print("  ERROR: should have raised ValueError")
except TypeError:
    # Python raises TypeError before our guard for duplicate kwargs
    print("  Correctly raised error for duplicate chunksize kwarg")
except ValueError as e:
    print(f"  Correctly raised ValueError: {e}")
print("*" * terminal_width)


# Get DataFrame and Markdown string
df1 = generate_table1(
    df,
    value_counts=True,
    export_markdown=True,
    markdown_path="data/table1_summary.md",
)

print(f"\nTable 1\n{df1}\n")

# Save Markdown to file
generate_table1(
    df,
    value_counts=True,
    export_markdown=True,
    markdown_path="data/table1_summary.md",
)

df2 = generate_table1(
    df,
    value_counts=True,
    export_markdown=True,
    markdown_path="data/table1_summary.md",
    include_types="continuous",
)

print(f"\nTable 1 Filtered for Continuous ONLY\n{df2}\n")

################################################################################
##################################### Add Ids ##################################
################################################################################

# Add a column of unique IDs with 9 digits and call it "census_id"
df = add_ids(
    df=df,
    id_colname="census_id",
    num_digits=9,
    seed=111,
    set_as_index=True,
)

print()
print("Dataset after `add_ids` function implemented:")
print(df.head())

################################################################################
############################### Trailing Period Removal ########################
################################################################################

# Create a sample dataframe with trailing periods in some values
data = {
    "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
df_trail = pd.DataFrame(data)

# Remove trailing periods from the 'values' column
df_trail = strip_trailing_period(df=df_trail, column_name="values")
print("Testing the  `strip_trailing_period` function:")
print()
print(df_trail)
print("*" * terminal_width)

################################################################################
################################# Standardized Dates ###########################
################################################################################

# Sample date strings
date_strings = ["15/04/2021", "04/15/2021", "01/12/2020", "12/01/2020"]

# Standardize the date strings
standardized_dates = [parse_date_with_rule(date) for date in date_strings]

print("Standardized Dates")
print(standardized_dates)
print("*" * terminal_width)

# Creating the DataFrame
data = {
    "date_column": [
        "31/12/2021",
        "01/01/2022",
        "12/31/2021",
        "13/02/2022",
        "07/04/2022",
    ],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "amount": [100.0, 150.5, 200.75, 250.25, 300.0],
}

################################################################################
############################### Parse Date With Rule ###########################
################################################################################
df_fake = pd.DataFrame(data)

# Apply the function to the DataFrame column
df_fake["standardized_date"] = df_fake["date_column"].apply(parse_date_with_rule)

print()
print("Testing the `parse_date_with_rule` function:")
print()
print(df_fake)
print("*" * terminal_width)

################################################################################
################################ DataFrame Analysis ############################
################################################################################

print()
print("Testing `dataframe_profiler` function:")
dataframe_profiler_df = dataframe_profiler(df=df)
print(dataframe_profiler_df)
print()

################################################################################
############################ Binning Numerical Columns #########################
################################################################################

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
############## Generating Summary Tables for Variable Combinations #############
################################################################################

# Define unique variables for the analysis
unique_vars = [
    "age_group",
    "workclass",
    "education",
    "occupation",
    "race",
    "sex",
    "income",
]

# Generate summary tables for all combinations of the specified variables
summary_tables, all_combinations = summarize_all_combinations(
    df=df,
    data_path=data_output,
    variables=unique_vars,
    data_name="census_summary_tables.xlsx",
)

# Print all combinations of variables
print("*" * terminal_width)
print(all_combinations)
print("*" * terminal_width)

################################################################################
########### Saving DataFrames to Excel with Customized Formatting ##############
################################################################################

# Example usage
file_name = "df_census.xlsx"  # Name of the output Excel file
file_path = os.path.join(data_path, file_name)

# filter DataFrame to Ages 18-40
filtered_df = df[(df["age"] > 18) & (df["age"] < 40)]

df_dict = {
    "original_df": df,
    "ages_18_to_40": filtered_df,
}

save_dataframes_to_excel(
    file_path=file_path,
    df_dict=df_dict,
    decimal_places=0,
)

################################################################################
######################### Creating Contingency Tables ##########################
################################################################################

contingency_table = contingency_table(
    df=df.head(),
    cols=[
        "age_group",
        "workclass",
        "race",
        "sex",
    ],
    sort_by=1,
)
print("*" * terminal_width)
print("Contingency Table")
print(contingency_table)
print("*" * terminal_width)


################################################################################

################################################################################
########################### Normality Tests ###################################
################################################################################

from eda_toolkit import normality_tests

# All three tests, all numeric features
summary = normality_tests(df, decimal_places=4)


print("*" * terminal_width)
print("Normality Test Results: df")
print(summary)
print("*" * terminal_width)


# Specific features and tests
summary = normality_tests(
    df,
    features=["age", "capital-gain", "hours-per-week"],
    tests=["shapiro", "dagostino"],
    alpha=0.01,
    decimal_places=10
)

print("*" * terminal_width)
print("Normality Test Results: Feature Specific")
print(summary)
print("*" * terminal_width)
