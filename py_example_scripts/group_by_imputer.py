################################################################################
############################## Group_by Imputer ################################
################################################################################

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
print
print(adult_df.head())
print("*" * terminal_width)

# Suppose X has: age (with nulls), workclass, education
X_imputed = groupby_imputer(
    df=adult_df,
    target="age",
    by=["workclass", "education"],
    stat="mean",
    fallback="global",
    as_new_col=True,  # create a new column instead of overwriting
)
