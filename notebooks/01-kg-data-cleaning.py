# %% [markdown]
#
# # Comprehensive Exam
#
# ## Coding Artifact
#
# Kalin Gibbons
#
# Nov 20, 2020

# ### Data Description
#
# This _training_ dataset is generated from simplified finite element models of a
# cruciate-sacrificing, post and cam driven knee implant performing a deep-knee-bend.
# The implant geometries and surgical alignments are parameterized by 13 predictor
# variables which were drawn using Latin hypercube sampling from a range of currently
# used manufacturer dimensions, and angles performed during successful surgeries. There
# were originally 15 predictors for this dataset, but two were fixed at average values
# for this particular batch of simulations. For the test dataset, the same predictors
# were uniformly drawn across the ranges of potential values.

# ### Data cleaning

import logging

# %%
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as spio
import seaborn as sns

from fepredict.datasets import select_by_regex

# !%load_ext autoreload
# !%autoreload 2


# !%matplotlib inline
# !%config InlineBackend.figure_format = 'retina'


# %%
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 14
mpl.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (9, 5.5)

sns.set_context("poster")
sns.set(rc={"figure.figsize": (16, 9.0)})
sns.set_style("whitegrid")

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
#
# ## Data Cleaning
#
# ---
#
# The data is contained within `MATLAB` binary files, which are easily imported into a
# `pandas` `DataFrame`. Some of the sensors implemented within the FEA model that
# generated the data do not consistently activate during a deep-knee-bend, so those
# columns need to be filtered out. Finally, if the uniform feature draw resulted in a
# particularly infeasible implant geometry, the simulation would fail, producing no
# results. These failed simulations will need to be removed.
#
# ### Controlling variable and function definitions


# %%
drop_regex = [
    # r'^time$',
    r"(femfe|patthick|patml|patsi)",  # features held constant
    r"^\w{3}_[xyz]_\w{3,4}$",
    r"^post\w+",
    r"^v[ilm][2-6]_(disp|force)$",
    r"^v[lm]1_(disp|force)$",
    r"^vert_(disp|force)$",
    r"^flex_(force|rot)$",
    r"^ap_force$",
    r"^(vv|ie)_torque",
    r"^(ml|pcm|pcl|pol)_force$",  # Always zero
    r"^(lclp|lcl|pmc|lcla|mcla)_force$",  # Often zero and bad predict
    r"^(pom|alc|mcl|mclp)_force$",  # Often zero and fairly bad predict.
]


def import_matlab_data(matfilepath):
    """Import MATLAB .mat binary into a panda DataFrame.

    Args:
        matfilepath (str|pathlike): The path to the .mat file

    Returns:
        pandas.DataFrame: A tabulation of features and response data, where
            each response entry is an array of data points for each step in
            the simulation.
    """
    data = spio.loadmat(matfilepath, squeeze_me=True)
    keys = list(data.keys())
    data = data[keys[-1]]
    columns = list(map(lambda x: x.lower(), data.dtype.names))
    old = ["femie", "femvv", "tibslope", "tibie", "tibvv", "xn", "ctf", "ctm"]
    new = [
        "fem_ie",
        "fem_vv",
        "tib_slope",
        "tib_ie",
        "tib_vv",
        "cop_",
        "force_",
        "torque_",
    ]
    for o, n in zip(old, new):
        columns = list(map(lambda x: x.replace(o, n), columns))

    data_df = pd.DataFrame(data)
    data_df.columns = columns
    return data_df


def drop_columns(data_df, regex_list):
    # """Remove columns using regular expressions."""
    return select_by_regex(data_df, regex_list, axis=1, negate=True)


def remove_failed(response_series, df_list):
    """Remove rows of DataFrames selected by empty entries in a series.

    Able to handle multiple DataFrames, allowing for removing empty rows after
    splitting into feature/response DataFrames.

    Args:
        response_series (pandas.Series): A series representing a sample column
            from the DataFrame to be split.
        df_list ([pandas.DataFrames]): A list of dataframes with shared indices
            that need empty rows removed.

    Returns:
        [pandas.DataFrames]: The list of dataframes (or single dataframe) with
            empty rows removed.
    """
    try:
        len(df_list)
    except TypeError:
        df_list = [df_list]

    failed_idx = response_series.apply(lambda x: x.size == 0)
    new_df_list = np.full(len(df_list), np.nan, dtype="object")
    for idx, df in enumerate(df_list):
        new_df_list[idx] = df.loc[~failed_idx]

    if len(new_df_list) == 1:
        return new_df_list[0]
    else:
        return new_df_list


# %% [markdown]
# ### Locating the data
#
# The `MATLAB` MAT files are stored in the `data/interim` folder because the raw data
# was stored in plaintext CSV files after being extracted from the FEA simulations.
# Once cleaned, we'll store the cleaned data in `data/preprocessed`.

# %%
# Source paths
# dirty_data_dir = Path.cwd().parent / "data" / "interim"
dirty_data_dir = Path.cwd().parent / "data" / "interim"
# print((dirty_data_dir)
dirty_test_path = dirty_data_dir / "test.mat"
dirty_train_path = dirty_data_dir / "doe.mat"

# Destination paths
cleaned_dir = dirty_data_dir.parent / "preprocessed"
cleaned_test_path = cleaned_dir / "test.parquet"
cleaned_train_path = cleaned_dir / "train.parquet"
print(dirty_test_path)
# %% [markdown]
# ### Data import and cleaning
#
# Reading the MAT file tables into memory, and outputting zpart of the dataframes to
# take a look at the data, then dropping the extraneous columns and taking a look at
# the results. We'll only look at the results from the testing set, but run identical
# operations on the training set, as well.

# %%
# Test
dirty_test = import_matlab_data(dirty_test_path)
dirty_test.shape

# %%
dirty_test.head()

# %%
dirty_test, dropped_cols = drop_columns(dirty_test, drop_regex)
print("Dropped", dropped_cols)

# %%
clean_test = remove_failed(dirty_test.iloc[:, -1], [dirty_test])
clean_test.shape

# %%
clean_test.head()

# %%
# Train
dirty_train = import_matlab_data(dirty_train_path)
dirty_train, _ = drop_columns(dirty_train, drop_regex)
clean_train = remove_failed(dirty_train.iloc[:, -1], [dirty_train])

# %% [markdown]
# ## Save the cleaned data

# Everything looked great, so we can save the cleaned data.

# %
# You'll likely have to modify the dataframes to use MultiIndexing, since to_parquet now
# dislikes the "object" datatype. MultiIndexing is pandas' way of dealing with more than
# 2D data. I added the old preprocessed data to the folders, but didn't check if loading
# them would work.

# clean_test.to_parquet(cleaned_test_path)
# clean_train.to_parquet(cleaned_train_path)

# %%
