################################################################################
########################### Import Requisite Libraries #########################
################################################################################

import pandas as pd
import numpy as np
import os

from eda_toolkit import (
    ensure_directory,
    read_csv_with_progress,
    detect_outliers,
)

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")

################################################################################
################################ Ensure Directory ##############################
################################################################################

try:
    terminal_width = os.get_terminal_size().columns
except OSError:
    terminal_width = 80

base_path = os.path.join(os.pardir)
data_path = os.path.join(os.pardir, "data")
data_output = os.path.join(os.pardir, "data_output")

print()
ensure_directory(data_path)
ensure_directory(data_output)

print()
print("*" * terminal_width)

################################################################################
############################### UCI ML Repository ##############################
################################################################################

try:
    df = read_csv_with_progress(os.path.join(data_path, "adult_income.csv"))
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    print("Loaded adult_income.csv from disk")
except FileNotFoundError:
    from ucimlrepo import fetch_ucirepo
    print("Fetching Adult Income Dataset from UCI ML Repository...")
    adult = fetch_ucirepo(id=2)
    df = adult.data.features.join(adult.data.targets, how="inner")
    df.to_csv(os.path.join(data_path, "adult_income.csv"))
    df = read_csv_with_progress(os.path.join(data_path, "adult_income.csv"))
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    print("Dataset saved to disk")

print(f"Shape: {df.shape}")
print("*" * terminal_width)

numeric_features = ["age", "fnlwgt", "education-num",
                    "capital-gain", "capital-loss", "hours-per-week"]

################################################################################
### 1. Basic IQR — all numeric features
################################################################################

print("\n[1] IQR — all numeric features (threshold=1.5)")
summary = detect_outliers(df, method="iqr", threshold=1.5)
print(summary.to_string(index=False))
print("*" * terminal_width)

################################################################################
### 2. IQR — strict threshold
################################################################################

print("\n[2] IQR — strict threshold (threshold=3.0)")
summary_strict = detect_outliers(
    df,
    features=numeric_features,
    method="iqr",
    threshold=3.0,
)
print(summary_strict.to_string(index=False))
print("*" * terminal_width)

################################################################################
### 3. Z-score
################################################################################

print("\n[3] Z-score — threshold=3.0")
summary_z = detect_outliers(
    df,
    features=numeric_features,
    method="zscore",
    threshold=3.0,
)
print(summary_z.to_string(index=False))
print("*" * terminal_width)

################################################################################
### 4. Isolation Forest
################################################################################

print("\n[4] Isolation Forest — contamination=0.05")
summary_iso = detect_outliers(
    df,
    features=numeric_features,
    method="isoforest",
    contamination=0.05,
)
print(summary_iso.to_string(index=False))
print("*" * terminal_width)

################################################################################
### 5. return_mask=True
################################################################################

print("\n[5] return_mask=True")
summary_mask, mask = detect_outliers(
    df,
    features=numeric_features,
    method="iqr",
    return_mask=True,
)
print(f"  Mask shape: {mask.shape}")
print(f"  Outlier rows per feature:\n{mask.sum()}")
print("*" * terminal_width)

################################################################################
### 6. flag_col
################################################################################

print("\n[6] flag_col='is_outlier'")
df_flagged = df.copy()
detect_outliers(
    df_flagged,
    features=numeric_features,
    method="iqr",
    flag_col="is_outlier",
)
n_flagged = df_flagged["is_outlier"].sum()
pct_flagged = n_flagged / len(df_flagged) * 100
print(f"  Total flagged rows: {n_flagged:,} ({pct_flagged:.1f}%)")
print(f"  Sample flagged rows:\n{df_flagged[df_flagged['is_outlier']].head(5)}")
print("*" * terminal_width)

################################################################################
### 7. groupby — within education groups
################################################################################

print("\n[7] groupby='education' — IQR within each education level")
summary_grp = detect_outliers(
    df,
    features=["age", "hours-per-week", "capital-gain"],
    method="iqr",
    groupby="education",
)
print(summary_grp.to_string(index=False))
print("*" * terminal_width)

################################################################################
### 8. groupby + return_mask + flag_col
################################################################################

print("\n[8] groupby + return_mask + flag_col combined")
df_grp = df.copy()
summary_grp2, mask_grp = detect_outliers(
    df_grp,
    features=["age", "hours-per-week"],
    method="iqr",
    groupby="education",
    return_mask=True,
    flag_col="is_outlier_grouped",
)
print(summary_grp2.to_string(index=False))
print(f"\n  Rows flagged (grouped): {df_grp['is_outlier_grouped'].sum():,}")
print("*" * terminal_width)

################################################################################
### 9. Feature-level filtering using mask
################################################################################

print("\n[9] Feature-level filtering using mask")
summary_m, mask_m = detect_outliers(
    df,
    features=numeric_features,
    method="iqr",
    return_mask=True,
)
# Rows where capital-gain is specifically an outlier
capital_gain_outliers = df[mask_m["capital-gain"]]
print(f"  capital-gain outlier rows: {len(capital_gain_outliers):,}")
print(capital_gain_outliers[["age", "capital-gain", "income"]].head(10).to_string())
print("*" * terminal_width)

################################################################################
### 10. Co-occurring outliers — rows flagged in multiple features
################################################################################

print("\n[10] Co-occurring outliers — rows flagged in multiple features")
summary_co, mask_co = detect_outliers(
    df,
    features=numeric_features,
    method="iqr",
    return_mask=True,
)
outlier_counts = mask_co.sum(axis=1)
print(outlier_counts.value_counts().sort_index().to_string())
print("*" * terminal_width)

################################################################################
### 11. verbose=True — ASCII summary report
################################################################################
 
print("\n[11] verbose=True — ASCII summary report")
detect_outliers(
    df,
    features=numeric_features,
    method="iqr",
    threshold=1.5,
    verbose=True,
)
print("*" * terminal_width)
 
################################################################################
### 12. return_bounds=True — pipe bounds into data_doctor
################################################################################
 
print("\n[12] return_bounds=True — bounds dict for downstream data_doctor use")
summary_b, bounds = detect_outliers(
    df,
    features=numeric_features,
    method="iqr",
    return_bounds=True,
)
print("  Bounds per feature:")
for feat, (lower, upper) in bounds.items():
    print(f"    {feat:<20} lower={lower:>10}  upper={upper:>10}")
 
print("\n  Example: pass capital-gain bounds directly to data_doctor:")
print(f"    lower_cutoff={bounds['capital-gain'][0]}")
print(f"    upper_cutoff={bounds['capital-gain'][1]}")
print("*" * terminal_width)
 
print("\nAll detect_outliers usage examples completed successfully.")