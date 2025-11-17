################################################################################
############################## Group_by Imputer ################################
################################################################################

import numpy as np
import os
from ucimlrepo import fetch_ucirepo
from eda_toolkit import groupby_imputer

# Get the width of the terminal
terminal_width = os.get_terminal_size().columns

################################################################################
## UCI ML Repository
################################################################################

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# Combine X and y into entire df
adult_df = X.join(y, how="inner")

print()
print("Adult Income Dataset")
print(adult_df.head())
print("*" * terminal_width)

################################################################################
## Introduce Missingness to 'age' column for demonstration
################################################################################

adult_df.loc[adult_df.sample(frac=0.3, random_state=42).index, "age"] = np.nan
print(f"\nProportion of missing values in 'age': {adult_df['age'].isna().mean()}\n")
print("*" * terminal_width)

################################################################################
## 1. Impute with fallback='global'
################################################################################

X_global = groupby_imputer(
    df=adult_df,
    impute_col="age",
    by=["workclass", "education"],
    stat="mean",
    fallback="global",
    as_new_col=True,
)

print("\n### Head with global fallback ###")
print(X_global[["age", "age_mean_imputed"]].head())
print("*" * terminal_width)

print("Means:")
print(f"Original adult_df['age'].mean(): {adult_df['age'].mean()}")
print(f"Imputed (global fallback) mean: {X_global['age_mean_imputed'].mean()}")
print("*" * terminal_width)

################################################################################
## 2. Impute with fallback equal to a fixed number instead of global
################################################################################

X_fixed = groupby_imputer(
    df=adult_df,
    impute_col="age",
    by=["workclass", "education"],
    stat="mean",
    fallback=50,  # <-- fixed fallback value
    as_new_col=True,
)

print("\n### Head with fixed fallback=50 ###")
print(X_fixed[["age", "age_mean_imputed"]].head())
print("*" * terminal_width)

print("Means:")
print(f"Original adult_df['age'].mean(): {adult_df['age'].mean()}")
print(f"Imputed (fixed fallback=50) mean: {X_fixed['age_mean_imputed'].mean()}")
print("*" * terminal_width)

################################################################################
## Wrap up comparison
################################################################################

print("\n### Comparison summary ###")
print(f"Missingness in original age: {adult_df['age'].isna().mean()}")
print(f"Global fallback mean: {X_global['age_mean_imputed'].mean()}")
print(f"Fixed fallback=50 mean: {X_fixed['age_mean_imputed'].mean()}")
print("*" * terminal_width)
