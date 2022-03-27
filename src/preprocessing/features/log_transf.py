"""Compute log transformation of features."""


from typing import List, Optional

import numpy as np
from pandas import DataFrame


def create_log_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    log_feat: Optional[List[str]] = None
):
    """Auxiliary function to create log features.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    log_feat : optional list of str, default=None
        List of logarithm features. The name of a log feature is expected to be an existing
        feature.
    """
    if log_feat is None or len(log_feat) == 0:
        return

    print("\nComputing logarithm features...")

    for feat_name in log_feat:

        train_df[feat_name + "_Log"] = np.log(1 + train_df[feat_name])
        if test_df is not None:
            test_df[feat_name + "_Log"] = np.log(1 + test_df[feat_name])

    print(f"Number of logarithm features: {len(log_feat)}.")
