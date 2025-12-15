# %%
################################################################################
##################### ECDF Plot Integration in data_doctor #####################
################################################################################

from ucimlrepo import fetch_ucirepo
from eda_toolkit import data_doctor

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
