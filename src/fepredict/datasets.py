# """Configures sample datasets and defines sample dataset loading functions."""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# from artifact.core_api import select_by_regex

# warnings.filterwarnings('ignore', category=UserWarning, module=pd)


# """A dictionary of regular expressions for selecting functional groups."""
tkr_group_lut = dict(
    contact_mechanics=r"^(?!pat).+(_area|_press|_cop_\d)$",
    joint_loads=r"^(?!pat).+(_force_\d|_torque_\d)$",
    kinematics=r"^(?!pat).+(_lat|_ant|_inf|_valgus|_external)$",
    ligaments=r"^(?!ml|pl).+(_force|_disp)$",
    patella=r"(pl|pat).*",
)


def select_by_regex(data_df, regex_list, axis=0, negate=False):
    """Index a dataframe using regex label matching.

    Args:
        data_df (pandas.DataFrame): the dataframe to be indexed.
        regex_list ([str]): A list of regular expressions used for pattern
        matching the index or columns of data_df.
        axis (int, optional): The axis to match against. Defaults to 0.
        negate (bool, optional): Flag to select the inverse of the regex
            match. Defaults to False.

    Returns:
        pandas.DataFrame: A copy of data_df containing only the matched index
            or columns (or with the negated match removed)
    """
    if axis == 0:
        labels = data_df.index
    elif axis == 1:
        labels = data_df.columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        is_match = np.any([labels.str.contains(x) for x in regex_list], axis=0)
    if negate:
        is_match = ~is_match

    selected_labels = labels[~is_match]
    return data_df.drop(selected_labels, axis=axis), selected_labels


def split_df(df, predictor_index):
    """Split dataframe into predictor and response columns.

    Uses the difference between two sets of indices to split, where the user
    passes the predictor columns as a function parameter.

    Args:
        df (DataFrame): The dataframe to be split.
        predictor_index ([list] int): A list of numeric indices for the
            predictor columns.

    Returns:
        [DataFrame]: The predictor and response dataframes.
    """
    every_index = np.arange(df.shape[1])
    response_index = np.setdiff1d(every_index, predictor_index)
    pred_df = df.iloc[:, predictor_index].drop(columns=["cam_rad"])  # constant
    resp_df = df.iloc[:, response_index]
    return pred_df.astype(np.float), resp_df


def drop_columns(data_df, regex_list, inplace=False):
    """Drop DataFrame columns by a list of regular expressions.

    Args:
        data_df (DataFrame): The dataframe with columns to be searched.
        regex_list ([list] str): A list of regular expressions for pattern
            matching.
        inplace (bool, Optional): Allow mutating of data_df. Defaults to False.

    Returns:
        DataFrame: A dataframe with any matching columns removed.
    """
    cols = data_df.columns
    needs_drop = np.any([cols.str.contains(x) for x in regex_list], axis=0)
    return data_df.drop(cols[needs_drop], axis="columns", inplace=inplace)


def load_tkr(functional_groups=None, subset=None):
    """Reader function for the 2018 total knee replacement dataset.

    Able to load only a subset of the data (train, test), as well as only a
    subset of the variables using functional group names as the selector:

        * contact_mechanics - Tibiofemoral contact areas and pressures
        * joint_loads - Tibiofemoral muscle forces and moments
        * kinematics - Joint coordinate system tibiofemoral kinematics
        * ligaments - Tibiofemoral ligament elongations and developed forces
        * patella - All of the above for the patellofemoral joint

    Args:
        functional_groups ([str], optional): A list of functional groups to load.
            Defaults to None.
        subset (str, optional): Either the train or test subset. Defaults to
            None.

    Returns:
        [DataFrame]: If a subset is selected, a pair of dataframes for the
            features or response variables. If no subset of passed, than a
            tuple of pairs of dataframes.
    """
    pred_idx = np.arange(0, 14)

    data_dir = Path.cwd().parent / "data" / "preprocessed"

    def select_group(df, functional_groups):
        if functional_groups is not None:
            functional_groups = pd.Series(functional_groups)
            patterns = df.iloc[0, pred_idx].index.to_list()
            patterns.append("time")
            for functional_group in functional_groups:
                if functional_group in tkr_group_lut.keys():
                    patterns.append(tkr_group_lut[functional_group])
                else:
                    warnings.warn(
                        f"{functional_group} not found. Choices: {tkr_group_lut.keys()}"
                    )
            df, _ = select_by_regex(df, patterns, axis=1)
        return df

    if (subset is None) or (subset.lower() == "test"):
        test_data = pd.read_parquet(data_dir / "test.parquet")
        test_data = select_group(test_data, functional_groups)
        test_feat, test_resp = split_df(test_data, pred_idx)

    if (subset is None) or (subset.lower() == "train"):
        train_data = pd.read_parquet(data_dir / "train.parquet")
        train_data = select_group(train_data, functional_groups)
        train_feat, train_resp = split_df(train_data, pred_idx)

    if subset is None:
        return (train_feat, train_resp), (test_feat, test_resp)
    if subset.lower() == "train":
        return train_feat, train_resp
    if subset.lower() == "test":
        return test_feat, test_resp
