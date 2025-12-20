"""
Examples for del_inactive_dataframes

Each example recreates its own DataFrames so behavior is isolated and easy to follow.
These examples are intended to be read top-to-bottom or run individually.

Note:
Because del_inactive_dataframes lives in the eda_toolkit package, these examples pass
namespace=globals() so the function inspects variables defined in this examples script.
"""

import pandas as pd

from eda_toolkit.data_manager import del_inactive_dataframes


# ---------------------------------------------------------------------
# Example 1: List active DataFrames (no deletion)
# ---------------------------------------------------------------------
print("\n=== Example 1: List active DataFrames (no deletion) ===")

df_main = pd.DataFrame({"a": range(10)})
df_tmp = pd.DataFrame({"b": range(100)})

del_inactive_dataframes(["df_main"], del_dataframes=False, namespace=globals())

del df_main, df_tmp


# ---------------------------------------------------------------------
# Example 2: Delete everything except df_main
# ---------------------------------------------------------------------
print("\n=== Example 2: Delete everything except df_main ===")

df_main = pd.DataFrame({"a": range(10)})
df_tmp = pd.DataFrame({"b": range(100)})

del_inactive_dataframes(["df_main"], del_dataframes=True, namespace=globals())

del df_main


# ---------------------------------------------------------------------
# Example 3: Dry run (preview deletions)
# ---------------------------------------------------------------------
print("\n=== Example 3: Dry run (preview only) ===")

df_main = pd.DataFrame({"a": range(10)})
df_tmp = pd.DataFrame({"b": range(100)})

del_inactive_dataframes(
    ["df_main"],
    del_dataframes=True,
    dry_run=True,
    namespace=globals(),
)

del df_main, df_tmp


# ---------------------------------------------------------------------
# Example 4: Include IPython output-cache variables
# (Meaningful in notebooks; safe in scripts)
# ---------------------------------------------------------------------
print("\n=== Example 4: Include IPython cache variables ===")

df_main = pd.DataFrame({"a": range(10)})
df_tmp = pd.DataFrame({"b": range(100)})

# In a notebook, displaying df_tmp as the last expression can create an IPython cache var like _14.
# In a script, this line does nothing special.
_ = df_tmp  # avoids printing a huge repr while still creating a reference

del_inactive_dataframes(
    ["df_main"],
    del_dataframes=True,
    include_ipython_cache=True,
    namespace=globals(),
)

del df_main


# ---------------------------------------------------------------------
# Example 5: Track DataFrame memory usage (default, low confusion)
# ---------------------------------------------------------------------
print("\n=== Example 5: Track DataFrame memory usage ===")

df_main = pd.DataFrame({"a": range(10)})
df_tmp = pd.DataFrame({"b": range(100)})

del_inactive_dataframes(
    ["df_main"],
    del_dataframes=True,
    track_memory=True,
    namespace=globals(),
)

del df_main


# ---------------------------------------------------------------------
# Example 6: Track DataFrame memory + process RSS (power-user mode)
# ---------------------------------------------------------------------
print("\n=== Example 6: Track memory (DataFrames + RSS) ===")

df_main = pd.DataFrame({"a": range(10)})
df_tmp = pd.DataFrame({"b": range(100)})

del_inactive_dataframes(
    ["df_main"],
    del_dataframes=True,
    track_memory=True,
    memory_mode="all",
    namespace=globals(),
)

del df_main


# ---------------------------------------------------------------------
# Example 7: Programmatic use (no console output)
# ---------------------------------------------------------------------
print("\n=== Example 7: Programmatic use (verbose=False) ===")

df_main = pd.DataFrame({"a": range(10)})
df_tmp = pd.DataFrame({"b": range(100)})

summary = del_inactive_dataframes(
    ["df_main"],
    del_dataframes=True,
    verbose=False,
    track_memory=True,
    namespace=globals(),
)

print("Returned summary dict:")
print(summary)

del df_main
