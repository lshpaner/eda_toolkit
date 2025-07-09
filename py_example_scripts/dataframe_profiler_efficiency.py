################################################################################
########################### Import Requisite Lbraries ##########################
################################################################################

# to create 2 million rows of fake data

import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta

from eda_toolkit import dataframe_profiler

# Set the number of rows and columns
n_rows = 2_000_000

# Generate random data
data = {
    "int_column": np.random.randint(
        1, 100, size=n_rows
    ),  # Random integers between 1 and 100
    "float_column": np.random.rand(n_rows),  # Random floats between 0 and 1
    "str_column": [
        "".join(random.choices(string.ascii_uppercase, k=5)) for _ in range(n_rows)
    ],  # Random 5-character strings
    "date_column": [
        datetime.now() - timedelta(days=random.randint(1, 365)) for _ in range(n_rows)
    ],  # Random dates from the last year
    "cat_column": np.random.choice(
        ["A", "B", "C", "D"], size=n_rows
    ),  # Random categorical values
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the first few rows to check
print(df.head())
