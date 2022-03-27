"""Replace a specific value from a feature with predictions."""


from typing import List, Optional

from pandas import DataFrame

from sklearn.ensemble import ExtraTreesRegressor


def correct_feat_value(
    seed: int, train_df: DataFrame, test_df: Optional[DataFrame] = None,
    correct_feat: Optional[List[str]] = None, value_to_replace: float = 0.
):
    """Auxiliary function to predict missing values.

    Parameters
    ----------
    seed : int
        Seed to use everywhere for reproducibility.

    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    correct_feat : optional list of str, default=None
        List of features where a specific value has to be replaced.

    value_to_replace : float, default=0.0
        Value to replace by predictions.
    """
    if correct_feat is None or len(correct_feat) == 0:
        return

    print("\nCorrecting features values...")

    for feat_name in correct_feat:

        estimator = ExtraTreesRegressor(random_state=seed, n_jobs=-1)

        train_diff = train_df.index[train_df[feat_name] != value_to_replace].tolist()
        train_rep = train_df.index[train_df[feat_name] == value_to_replace].tolist()
        if test_df is not None:
            test_rep = test_df.index[test_df[feat_name] == value_to_replace].tolist()

        estimator.fit(
            train_df.drop(columns=[feat_name]).loc[train_diff, :],
            train_df.loc[train_diff, feat_name]
        )

        train_df.loc[train_rep, feat_name] = estimator.predict(
            train_df.drop(columns=[feat_name]).loc[train_rep, :]
        )
        if test_df is not None:
            test_df.loc[test_rep, feat_name] = estimator.predict(
                test_df.drop(columns=[feat_name]).loc[test_rep, :]
            )

    print("Corrected features values.")
