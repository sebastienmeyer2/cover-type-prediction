"""Utilitary functions to build new features and save the datasets."""


import argparse

from typing import Dict, List, Optional, Tuple

import warnings

import pandas as pd
from pandas import DataFrame

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from preprocessing.features.groups import create_families_feat, create_soil_feat
from preprocessing.features.inference import correct_feat_value
from preprocessing.features.knowledge_domain import create_kd_feat
from preprocessing.features.log_transf import create_log_feat
from preprocessing.features.polynomial import create_poly_feat
from preprocessing.features.statistics import create_stat_feat
from preprocessing.features.sum_diff import create_pm_feat
from preprocessing.reduction.droping import drop_corr_feat, drop_list_feat
from preprocessing.reduction.merger import merge_binary_feat
from preprocessing.reduction.pca import perform_pca
from preprocessing.resampling import training_resample

from utils.args_fmt import float_zero_one


warnings.filterwarnings("ignore", message="After over-sampling, the number of samples ")


def create_features(
    seed: int = 42,
    correct_feat: Optional[List[str]] = None,
    value_to_replace: float = 0.,
    binary_feat: Optional[List[str]] = None,
    log_feat: Optional[List[str]] = None,
    stat_feat: Optional[List[str]] = None,
    pm_feat: Optional[List[str]] = None,
    kd_feat: Optional[List[str]] = None,
    families_feat: Optional[List[str]] = None,
    soil_feat: Optional[List[str]] = None,
    fixed_poly_feat: Optional[List[str]] = None,
    poly_feat: Optional[List[str]] = None,
    all_poly_feat: bool = False,
    poly_degree: int = 2,
    excl_feat: Optional[List[str]] = None,
    corr_threshold: float = 1.,
    rescale_data: bool = True,
    scaling_method: str = "standard",
    pca_ratio: float = 1.,
    nb_sampling: int = 0,
    manual_resampling: Optional[Dict[int, int]] = None,
    save_data: bool = True,
    file_suffix: str = "final"
) -> Tuple[DataFrame, ...]:
    """Retrieve local data files and perform preprocessing oprations.

    Parameters
    ----------
    seed : int, default=42
        Seed to use everywhere for reproducibility.

    correct_feat : optional list of str, default=None
        List of features where a specific value has to be replaced.

    value_to_replace : float, default=0.0
        Value to replace by predictions.

    binary_feat : optional list of str, default=None
        List of binary features to merge. The name of a binary feature is expected to be the base
        and final feature name, for example "Soil_Type".

    log_feat : optional list of str, default=None
        List of logarithm features. The name of a log feature is expected to be an existing
        feature.

    stat_feat : optional list of str, default=None
        List of stat features. The name of a stat feature is expected to be either "Max", "Min",
        "Mean", "Median" or "Std".

    pm_feat : optional list of str, default=None
        List of feature to sum and substract. The name of a pm feature is expected to be an
        existing features.

    kd_feat : optional list of str, default=None
        List of knowledge-domain features. The name of a knowledge-domain feature is expected to be
        either "Distance_To_Hydrology", "Mean_Distance_To_Points_Of_Interest",
        "Elevation_Shifted_Horizontal_Distance_To_Hydrology",
        "Elevation_Shifted_Vertical_Distance_To_Hydrology" or
        "Elevation_Shifted_Horizontal_Distance_To_Roadways".

    families_feat : optional list of str, default=None
        List of families features to compute. The name of a binary feature is expected to be either
        "Ratake", "Vanet", "Catamount", "Leighan", "Bullwark", "Como", "Moran" or "Other".

    soil_feat : optional list of str, default=None
        List of soil types features to compute. The name of a binary feature is expected to be
        either "Stony", "Rubly" or "Other".

    fixed_poly_feat : optional list of str, default=None
        List of specific polynomial features. A fixed polynomial feature must be of the form
        "A B C". A, B and C can be features or powers of features, and their product will be
        computed.

    poly_feat : optional list of str, default=None
        List of polynomial features to compute of which interaction terms will be computed.

    all_poly_feat : bool, default=False
        If True, will use all the computed features for polynomial interaction. Use with caution.

    poly_degree : int, default=2
        Define the degree until which products and powers of features are computed. If 1 or less,
        there will be no polynomial features.

    excl_feat : optional list of str, default=None
        List of features names to drop.

    corr_threshold : float, default=1.0
        Correlation threshold to select features.

    rescale_data : bool, default=True
        If True, will rescale all features with zero mean and unit variance.

    scaling_method : str, default="standard"
        If "standard", features are rescaled with zero mean and unit variance. If "positive",
        features are rescaled between zero and one.

    pca_ratio : float, default=1.0
        Variance ratio parameter for the Principal Component Analysis.

    nb_sampling : int, default=0
        Number of predicted resampling operations.

    manual_resampling : optional dict of {int: int}, default=None
        Manual selection of number of samples for each class.

    save_data : bool, default=True
        If True, will save the computed features in two csv files, one for training and one for
        testing.

    file_suffix : str, default="final"
        Suffix to append to the training and test files if **save_data** is True.

    Returns
    -------
    train_df : DataFrame
        Training dataframe containing the final features and training labels.

    test_df : DataFrame
        Test dataframe containing the final features and test labels.

    Raises
    ------
    ValueError
        If **scaling_method** is not supported.
    """
    # Read files
    test_df = pd.read_csv("data/covtype.csv", index_col=["Id"])

    training_ids = []

    with open("data/training_ids.txt", "r", encoding="utf-8") as f:

        training_ids = f.read().split(",")
        training_ids = [int(x) for x in training_ids]

    train_df = test_df.iloc[training_ids, :].copy()

    # Shuffle
    train_df = shuffle(train_df, random_state=seed)
    test_df = shuffle(test_df, random_state=seed)

    # Binary variables and target
    label_var = ["Cover_Type"]
    y_train = train_df[label_var]
    train_df = train_df.drop(columns=label_var)
    y_test = test_df[label_var]
    test_df = test_df.drop(columns=label_var)

    # Correct missing values
    correct_feat_value(
        seed, train_df, test_df=test_df, correct_feat=correct_feat,
        value_to_replace=value_to_replace
    )

    # Merge binary features
    merge_binary_feat(train_df, test_df=test_df, binary_feat=binary_feat)

    # Logarithm features
    create_log_feat(train_df, test_df=test_df, log_feat=log_feat)

    # Statistics
    create_stat_feat(train_df, test_df=test_df, stat_feat=stat_feat)

    # Make some additions and differences of features
    create_pm_feat(train_df, test_df=test_df, pm_feat=pm_feat)

    # Knowledge-domain features
    create_kd_feat(train_df, test_df=test_df, kd_feat=kd_feat)

    # Groups of families
    create_families_feat(
        train_df, test_df=test_df, families_feat=families_feat, excl_feat=excl_feat
    )

    # Groups of soil types
    create_soil_feat(train_df, test_df=test_df, soil_feat=soil_feat, excl_feat=excl_feat)

    # Compute polynomial and interaction features
    if all_poly_feat:
        fixed_poly_feat = []
        poly_feat = list(train_df.columns)

    create_poly_feat(
        train_df, test_df=test_df, fixed_poly_feat=fixed_poly_feat, poly_feat=poly_feat,
        poly_degree=poly_degree
    )

    # Drop excluded features
    drop_list_feat(train_df, test_df=test_df, excl_feat=excl_feat)

    # Drop correlated features
    drop_corr_feat(train_df, test_df=test_df, corr_threshold=corr_threshold)

    # Rescale the data
    if rescale_data:

        if scaling_method == "standard":
            sc = StandardScaler()
        elif scaling_method == "positive":
            sc = MinMaxScaler()
        else:
            err_msg = f"Unsupported scaling method {scaling_method}."
            raise ValueError(err_msg)

        train_df[train_df.columns] = sc.fit_transform(train_df[train_df.columns])
        test_df[test_df.columns] = sc.transform(test_df[test_df.columns])

    elif not rescale_data and pca_ratio < 1.:

        warn_msg = "Warning: Rescaling data is recommended when performing PCA."
        warn_msg += " Set rescale_data to True or pca_ratio to 1."
        print(warn_msg)

    # Reduce the dimensionality with PCA
    train_df, test_df = perform_pca(seed, train_df, test_df=test_df, pca_ratio=pca_ratio)

    # Predicted and manual resampling
    train_df, y_train = training_resample(
        seed, train_df, y_train, test_df=test_df, nb_sampling=nb_sampling,
        manual_resampling=manual_resampling
    )

    # Print some information
    print("Final training shape: ", train_df.shape)
    print("The features are: ", train_df.columns)

    # Re-create the label variable
    train_df = pd.merge(train_df, y_train, left_index=True, right_index=True, how="left")
    train_df.index.name = "Id"

    test_df = pd.merge(test_df, y_test, left_index=True, right_index=True, how="left")

    # Save the data
    if save_data:
        train_df.to_csv(path_or_buf="data/train_" + file_suffix + ".csv", header=True, index=True)
        test_df.to_csv(path_or_buf="data/test_" + file_suffix + ".csv", header=True, index=True)

    return train_df, test_df


if __name__ == "__main__":

    # Command lines
    parser_desc = "Main file to prepare data and features."
    parser = argparse.ArgumentParser(description=parser_desc)

    # Seed
    parser.add_argument(
        "--seed",
        default=8005,
        type=int,
        help="""
             Seed to use everywhere for reproducbility.
             Default: 42.
             """
    )

    # Correct missing values
    parser.add_argument(
        "--correct-feat",
        default=["Hillshade_3pm"],
        nargs="*",
        help="""
             List of features where a specific value has to be replaced.
             """
    )

    parser.add_argument(
        "--value",
        default=0,
        type=float,
        help="""
             Value to replace by predictions.
             Default: 0.
             """
    )

    # Merge binary features
    parser.add_argument(
        "--binary-feat",
        default=[],
        nargs="*",
        help="""
             List of binary features to merge. The name of a binary feature is expected to be the
             base and final feature name, for example "Soil_Type".
             """
    )

    # Logarithm features
    parser.add_argument(
        "--log-feat",
        default=[
            "Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points"
        ],
        nargs="*",
        help="""
             List of logarithm features. The name of a log feature is expected to be an existing
             feature.
             """
    )

    # Statistics features
    parser.add_argument(
        "--stat-feat",
        default=["Max", "Std"],
        nargs="*",
        help="""
             List of stat features. The name of a stat feature is expected to be either "Max",
             "Min", "Mean", "Median" or "Std".
             """
    )

    # Make some additions and differences of features
    parser.add_argument(
        "--pm-feat",
        default=[
            "Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points"
        ],
        nargs="*",
        help="""
             List of feature to sum and substract. The name of a pm feature is expected to be an
             existing features.
             """
    )

    # Knowledge-domain features
    parser.add_argument(
        "--kd-feat",
        default=[
            "Distance_To_Hydrology", "Mean_Distance_To_Points_Of_Interest",
            "Elevation_Shifted_Horizontal_Distance_To_Hydrology",
            "Elevation_Shifted_Vertical_Distance_To_Hydrology",
            "Elevation_Shifted_Horizontal_Distance_To_Roadways"
        ],
        nargs="*",
        help="""
             List of knowledge-domain features. The name of a knowledge-domain feature is expected
             to be either "Distance_To_Hydrology", "Mean_Distance_To_Points_Of_Interest",
             "Elevation_Shifted_Horizontal_Distance_To_Hydrology",
             "Elevation_Shifted_Vertical_Distance_To_Hydrology" or
             "Elevation_Shifted_Horizontal_Distance_To_Roadways".
             """
    )

    # Groups of families
    parser.add_argument(
        "--families-feat",
        default=["Ratake", "Vanet", "Catamount", "Leighan", "Bullwark", "Como", "Moran", "Other"],
        nargs="*",
        help="""
             List of families features to compute. The name of a binary feature is expected to be
             either "Ratake", "Vanet", "Catamount", "Leighan", "Bullwark", "Como", "Moran" or
             "Other".
             """
    )

    # Groups of soil types features
    parser.add_argument(
        "--soil-feat",
        default=["Stony", "Rubly"],
        nargs="*",
        help="""
             List of soil types features to compute. The name of a binary feature is expected to be
             either "Stony", "Rubly" or "Other".
             """
    )

    # Polynomial features
    parser.add_argument(
        "--fixed-poly-feat",
        default=[],
        nargs="*",
        help="""
             List of specific polynomial features. A fixed polynomial feature must be of the form
             "A B C". A, B and C can be features or powers of features, and their product will be
             computed.
             Example: --fixed-poly-feat "char_count^2 group_overlap".
             """
    )

    parser.add_argument(
        "--poly-feat",
        default=[
            "Horizontal_Distance_To_Roadways_Log", "Horizontal_Distance_To_Fire_Points_Log",
            "Elevation_Shifted_Vertical_Distance_To_Hydrology",
            "Elevation_Shifted_Horizontal_Distance_To_Hydrology"
        ],
        nargs="*",
        help="""
             List of polynomial features to compute of which interaction terms will be computed.
             Example: --poly-feat ldp_idf char_count.
             """
    )
    parser.add_argument(
        "--all-poly-feat",
        action="store_true",
        help="""
             Use this option to activate polynomial interaction of all features. Use with
             caution.
             Default: Deactivated.
             """
    )
    parser.set_defaults(all_poly_feat=False)
    parser.add_argument(
        "--poly-degree",
        default=2,
        type=int,
        help="""
             Define the degree until which products and powers of features are computed. If 1 or
             less, there will be no polynomial features.
             Default: 2.
             """
    )

    # Excluded features
    parser.add_argument(
        "--excl-feat",
        default=["Soil_Type15"],
        nargs="*",
        help="""
             List of features names to drop after computation.
             Example: --excl-feat "char_count^2 group_overlap".
             """
    )

    # Correlation threshold
    parser.add_argument(
        "--max-correlation",
        default=1.,
        type=float,
        help="""
             Correlation threshold to select features.
             Default: 1.0.
             """
    )

    # Rescale data
    parser.add_argument(
        "--rescale-data",
        action="store_true",
        help="""
             Use this option to activate rescaling the data sets.
             Default: Activated.
             """
    )
    parser.add_argument(
        "--no-rescale-data",
        action="store_false",
        dest="rescale-data",
        help="""
             Use this option to deactivate rescaling the data sets.
             Default: Activated.
             """
    )
    parser.set_defaults(rescale_data=True)

    parser.add_argument(
        "--scaling-method",
        default="standard",
        type=str,
        help="""
             If "standard", features are rescaled with zero mean and unit variance. If "positive",
             features are rescaled between zero and one.
             Default: "standard".
             """
    )

    # PCA ratio
    parser.add_argument(
        "--pca-ratio",
        default=1.,
        type=float,
        help="""
             Variance ratio parameter for the Principal Component Analysis.
             Default: 1.0.
             """
    )

    # Resampling
    parser.add_argument(
        "--nb-sampling",
        default=0,
        type=int,
        help="""
             Number of predicted resampling operations.
             Default: 0.
             """
    )

    parser.add_argument(
        "--manual-resampling",
        default="1:5500,2:7000,3:1000,4:50,5:450,6:500,7:650",
        type=lambda x: {int(k): int(v) for k, v in (i.split(":") for i in x.split(","))},
        help="""
             Manual selection of number of samples for each class.
             """
    )

    # Save data
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="""
             Use this option to activate saving the data sets.
             Default: Activated.
             """
    )
    parser.add_argument(
        "--no-save-data",
        action="store_false",
        dest="save-data",
        help="""
             Use this option to deactivate saving the data sets.
             Default: Activated.
             """
    )
    parser.set_defaults(save_data=True)

    parser.add_argument(
        "--file-suffix",
        default="final",
        type=str,
        help="""
             Suffix to append to the training and test files if **save_data** is True.
             Default: "final".
             """
    )

    # End of command lines
    args = parser.parse_args()

    create_features(
        seed=args.seed,
        correct_feat=args.correct_feat,
        value_to_replace=args.value,
        binary_feat=args.binary_feat,
        log_feat=args.log_feat,
        stat_feat=args.stat_feat,
        kd_feat=args.kd_feat,
        pm_feat=args.pm_feat,
        families_feat=args.families_feat,
        soil_feat=args.soil_feat,
        fixed_poly_feat=args.fixed_poly_feat,
        poly_feat=args.poly_feat,
        all_poly_feat=args.all_poly_feat,
        poly_degree=args.poly_degree,
        excl_feat=args.excl_feat,
        corr_threshold=float_zero_one(args.max_correlation),
        rescale_data=args.rescale_data,
        scaling_method=args.scaling_method,
        pca_ratio=float_zero_one(args.save_data),
        nb_sampling=args.nb_sampling,
        manual_resampling=args.manual_resampling,
        save_data=args.save_data,
        file_suffix=args.file_suffix
    )
