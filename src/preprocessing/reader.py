"""Gather utilitary functions to read the data."""


from typing import Tuple

import pandas as pd
from pandas import DataFrame


def get_data(data_path: str = "data/", file_suffix: str = "final") -> Tuple[DataFrame, ...]:
    """Retrieve local data files and perform preprocessing oprations.

    Parameters
    ----------
    data_path : str, default="data/"
        Path to the data folder.

    file_suffix : str, default="final"
        Suffix to the training and test filenames.

    Returns
    -------
    train_df : DataFrame
        Training features.

    y_train : DataFrame
        Training labels.

    test_df : DataFrame
        Test features.

    y_test : DataFrame
        Test labels.
    """
    # Read files
    path_to_train = data_path + "train_" + file_suffix + ".csv"
    path_to_test = data_path + "test_" + file_suffix + ".csv"

    train_df = pd.read_csv(path_to_train, index_col=["Id"])
    test_df = pd.read_csv(path_to_test, index_col=["Id"])

    # Extract target
    label_var = ["Cover_Type"]
    y_train = train_df[label_var]
    train_df = train_df.drop(columns=label_var)
    y_test = test_df[label_var]
    test_df = test_df.drop(columns=label_var)

    return train_df, y_train, test_df, y_test
