"""Compute additions and differences of features."""


from typing import List, Optional

from pandas import DataFrame


def create_pm_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    pm_feat: Optional[List[str]] = None
):
    """Auxiliary function to create additions and differences of features.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    pm_feat : optional list of str, default=None
        List of feature to sum and substract. The name of a pm feature is expected to be an
        existing features.
    """
    if pm_feat is None or len(pm_feat) == 0:
        return

    print("\nComputing plus/minus features...")

    dfs = [train_df]
    if test_df is not None:
        dfs.append(test_df)

    n = len(pm_feat)

    for df in dfs:

        for i in range(n):
            for j in range(i + 1, n):

                f1 = pm_feat[i]
                f2 = pm_feat[j]

                df[f1 + "_plus_" + f2] = df[f1] + df[f2]
                df[f1 + "_minus_" + f2] = df[f1] - df[f2]

    nb_new_feat = int(2 * (n * (n + 1) / 2))

    print(f"Number of plus/minus features: {nb_new_feat}.")
