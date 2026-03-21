# %%
################################################################################
##################### ECDF Plot Integration in data_doctor #####################
################################################################################

import os
from ucimlrepo import fetch_ucirepo
from eda_toolkit import data_doctor

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")

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


################################################################################
## data_doctor with ECDF
################################################################################

print("\nRunning data_doctor with plot_type=['kde', 'ecdf', 'box_violin'] ...\n")

data_doctor(
    df=adult_df,
    feature_name="age",
    plot_type=["kde", "ecdf", "box_violin"],
    scale_conversion="log",
)
