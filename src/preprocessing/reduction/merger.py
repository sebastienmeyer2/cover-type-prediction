"""Merge binary features."""


from typing import List, Optional

from pandas import DataFrame


def merge_binary_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    binary_feat: Optional[List[str]] = None
):
    """Auxiliary function to merge binary features.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    binary_feat : optional list of str, default=None
        List of binary features to merge. The name of a binary feature is expected to be of the
        form "Ax". A the base and final feature. x is the number of each related binary feature.
    """
    if binary_feat is None or len(binary_feat) == 0:
        return

    print("\nMerging binary features...")

    for feat_name in binary_feat:

        # Binary variables
        n = len(feat_name)

        cols = list(train_df.columns)
        binary_feat_ext = []

        for col in cols:

            if col.replace(feat_name, "").isdigit():

                binary_feat_ext.append(col)

        # Merge wilderness area features
        train_df[feat_name] = train_df[binary_feat_ext].idxmax(axis=1).str[n:].astype(int) - 1
        train_df = train_df.drop(columns=binary_feat_ext, inplace=True)

        test_df[feat_name] = test_df[binary_feat_ext].idxmax(axis=1).str[n:].astype(int) - 1
        test_df = test_df.drop(columns=binary_feat_ext, inplace=True)

    print("Merged binary features.")
