"""Resampling operations for training data set."""


from typing import Dict, Optional, Tuple

import numpy as np
from pandas import DataFrame

from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids


def training_resample(
    seed: int, train_df: DataFrame, y_train: DataFrame, test_df: Optional[DataFrame] = None,
    nb_sampling: int = 0, manual_resampling: Optional[Dict[int, int]] = None
) -> Tuple[DataFrame, ...]:
    """Manual or predicted resampling.

    Parameters
    ----------
    seed : int
        Seed to use everywhere for reproducibility.

    train_df : DataFrame
        Training dataframe containing the initial features.

    y_train : DataFrame
        Training labels.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    nb_sampling : int, default=0
        Number of predicted resampling operations.

    manual_resampling : optional dict of {int: int}, default=None
        Manual selection of number of samples for each class.

    Returns
    -------
    train_df : DataFrame
        Training dataframe after resampling.

    y_train : DataFrame
        Training labels after resampling.
    """
    if manual_resampling is None and (nb_sampling <= 0 or test_df is None):
        return train_df, y_train

    print("\nResampling...")

    train_df_sample = train_df.copy()
    y_train_sample = y_train.copy()

    if manual_resampling is not None:  # prioritize manual resampling

        clust_resampling = {i: min(2160, manual_resampling[i]) for i in range(1, 8)}

        print(f"Fixed repartition in training set after operations: {manual_resampling}.")

        # ClusteringCentroids
        kmeans = KMeans(random_state=seed)
        clust = ClusterCentroids(
            sampling_strategy=clust_resampling, random_state=seed, estimator=kmeans
        )

        train_df_sample, y_train_sample = clust.fit_resample(train_df, y_train)

        smote = SMOTE(sampling_strategy=manual_resampling, random_state=seed, n_jobs=-1)

        train_df_sample, y_train_sample = smote.fit_resample(train_df_sample, y_train_sample)

    elif nb_sampling > 0 and test_df is not None:

        for k in range(nb_sampling):

            # Fit an ExtraTreesClassifier as usual on whole training set
            etc = ExtraTreesClassifier(
                n_estimators=238, criterion="gini", max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features="auto", max_leaf_nodes=None,
                min_impurity_decrease=5.76e-8, bootstrap=False, ccp_alpha=3.64e-6,
                random_state=seed, n_jobs=-1, verbose=0
            )
            etc.fit(train_df_sample, y_train_sample.to_numpy().flatten())

            # Estimate repartition of the labels in the test set
            y_pred = etc.predict(test_df)

            _, pred_resampling = np.unique(y_pred, return_counts=True)

            pred_resampling = pred_resampling.astype(float)/test_df.shape[0]
            pred_resampling = {i: int(pred_resampling[i-1]*15120) for i in range(1, 8)}
            clust_resampling = {i: min(2160, pred_resampling[i]) for i in range(1, 8)}

            print(f"Training labels repartition after {k + 1} resampling: {pred_resampling}.")

            # ClusteringCentroids
            kmeans = KMeans(random_state=seed)
            clust = ClusterCentroids(
                sampling_strategy=clust_resampling, random_state=seed, estimator=kmeans
            )

            train_df_sample, y_train_sample = clust.fit_resample(train_df, y_train)

            smote = SMOTE(sampling_strategy=pred_resampling, random_state=seed, n_jobs=-1)

            train_df_sample, y_train_sample = smote.fit_resample(train_df_sample, y_train_sample)

    train_df = train_df_sample.copy()
    y_train = y_train_sample.copy()

    print("Resampling done.")

    return train_df, y_train
