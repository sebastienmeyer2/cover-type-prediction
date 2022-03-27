"""Drop features from dataframes based on different criteria."""


from typing import List, Optional

from tqdm import tqdm

from pandas import DataFrame


def drop_corr_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None, corr_threshold: float = 1.
):
    """Iterate through features and return the most correlated ones.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    corr_threshold : float, default=1.0
        Correlation threshold to select features.
    """
    if corr_threshold >= 1.:
        return

    print("\nDropping correlated features...")

    # Compute correlation
    corr_df = train_df.corr()

    # Empty dictionary to hold correlated features
    above_threshold_cols = {}

    # For each column, record the features that are above the threshold
    for col in corr_df:
        above_threshold_cols[col] = list(corr_df.index[corr_df[col] > corr_threshold])

    # Track columns to remove and columns already examined
    cols_to_remove = set()
    cols_seen = set()
    cols_to_remove_pair = set()

    # Iterate through columns and correlated columns
    for col, corr_cols in tqdm(above_threshold_cols.items()):

        # Keep track of columns already examined
        cols_seen.add(col)

        for x in corr_cols:

            if x != col:  # a variable is totally correlated with itself
                # Only want to remove one in a pair
                if x not in cols_seen:
                    cols_to_remove.add(x)
                    cols_to_remove_pair.add(col)

    # Remove highly correlated features
    list_cols_to_remove = list(cols_to_remove)

    train_df.drop(columns=list_cols_to_remove, inplace=True)
    if test_df is not None:
        test_df.drop(columns=list_cols_to_remove, inplace=True)

    print(f"Number of features after decorrelation: {train_df.shape[1]}")


def drop_list_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    excl_feat: Optional[List[str]] = None, errors: str = "ignore"
) -> List[str]:
    """Drop a provided list of features from the dataframes.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    excl_feat : optional list of str, default=None
        List of features names to drop.

    errors : str, default="ignore"
        If "ignore", no error will be raised in case some features were already missing in the
        dataframes. If "raise", will prompt an error.
    """
    if excl_feat is None or len(excl_feat) == 0:
        return

    print("\nDropping specific features...")

    train_df.drop(columns=excl_feat, inplace=True, errors=errors)
    if test_df is not None:
        test_df.drop(columns=excl_feat, inplace=True, errors=errors)

    print(f"Number of features after dropping excluded ones: {train_df.shape[1]}")
