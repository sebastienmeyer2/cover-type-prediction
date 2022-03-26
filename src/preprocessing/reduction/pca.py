"""Perform PCA."""


from typing import Optional, Tuple

import pandas as pd
from pandas import DataFrame

from sklearn.decomposition import PCA


def perform_pca(
    seed: int, train_df: DataFrame, test_df: Optional[DataFrame] = None, pca_ratio: float = 1.
) -> Tuple[DataFrame, Optional[DataFrame]]:
    """Perform PCA on given dataframes.

    Parameters
    ----------
    seed : int
        Seed the use everywhere for reproducibility.

    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    pca_ratio : float, default=1.0
        Variance ratio parameter for the Principal Component Analysis.

    Returns
    -------
    train_df : DataFrame
        Training dataframe containing the PCA features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the PCA features.
    """
    if pca_ratio <= 0. or pca_ratio >= 1.:
        return train_df, test_df

    print("\nPerforming PCA...")

    # Fit the PCA
    pca = PCA(n_components=pca_ratio, random_state=seed)

    pca.fit(train_df.to_numpy())

    # Transform the dataframes
    pca_col = [f"PCA_{i}" for i in range(1, pca.n_components_+1)]

    train_df = pd.DataFrame(
        pca.transform(train_df.to_numpy()), columns=pca_col, index=train_df.index
    )
    if test_df is not None:
        test_df = pd.DataFrame(
            pca.transform(test_df.to_numpy()), columns=pca_col, index=test_df.index
        )

    print(f"Number of features after PCA: {train_df.shape[1]}")

    return train_df, test_df
