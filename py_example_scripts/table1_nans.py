import numpy as np
import pandas as pd
from eda_toolkit import generate_table1

print("NaN mixed with valid categories:")
# NaN mixed with valid categories
df1 = pd.DataFrame({
    "x":     ["a", "a", "a", "b", "b", "b", np.nan, np.nan, np.nan, np.nan],
    "group": ["A", "B", "A", "B", "A", "B", "A",    "B",    "A",    "B"],
})
print(generate_table1(df1, categorical_cols=["x"], continuous_cols=[],
                      groupby_col="group", value_counts=True))

print()
print("All-NaN column:")
# All-NaN column
df2 = pd.DataFrame({
    "x":     [np.nan]*6,
    "group": ["A", "B", "A", "B", "A", "B"],
})
print(generate_table1(df2, categorical_cols=["x"], continuous_cols=[],
                      groupby_col="group", value_counts=True))


