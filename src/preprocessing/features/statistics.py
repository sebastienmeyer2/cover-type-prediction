"""Compute statistics of features."""


from typing import List, Optional

from pandas import DataFrame


def create_stat_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    stat_feat: Optional[List[str]] = None
):
    """Auxiliary function to create stat features.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    stat_feat : optional list of str, default=None
        List of stat features. The name of a stat feature is expected to be either "Max", "Min",
        "Mean", "Median" or "Std".
    """
    if stat_feat is None or len(stat_feat) == 0:
        return

    print("\nComputing stat features...")

    dfs = [train_df]
    if test_df is not None:
        dfs.append(test_df)

    for df in dfs:

        for feat_name in stat_feat:

            if feat_name == "Max":

                df[feat_name] = df.max(axis=1)

            elif feat_name == "Min":

                df[feat_name] = df.min(axis=1)

            elif feat_name == "Mean":

                df[feat_name] = df.mean(axis=1)

            elif feat_name == "Median":

                df[feat_name] = df.median(axis=1)

            elif feat_name == "Std":

                df[feat_name] = df.std(axis=1)

            else:

                err_msg = f"Unknown statistic {feat_name}."
                raise ValueError(err_msg)

    print(f"Number of stat features: {len(stat_feat)}.")
