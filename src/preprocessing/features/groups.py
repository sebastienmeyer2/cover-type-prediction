"""Create features related to knowledge-domain groups."""


from typing import List, Optional

from pandas import DataFrame


FAMILIES = {
    "Ratake": [2, 4],
    "Vanet": [2, 5, 6],
    "Catamount": [10, 11, 13, 26, 31, 32, 33],
    "Leighan": [21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 33, 38],
    "Bullwark": [10, 11],
    "Como": [29, 30],
    "Moran": [38, 39, 40],
    "Other": [3, 14, 15, 16, 19, 20, 34, 35, 37]
}

SOIL_TYPES = {
    "Stony": [1, 2, 6, 9, 12, 18, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40],
    "Rubly": [3, 4, 5, 10, 11, 13],
    "Other": [7, 8, 14, 15, 16, 17, 19, 20, 21, 22]
}


def create_families_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    families_feat: Optional[List[str]] = None, excl_feat: Optional[List[str]] = None
):
    """Auxiliary function to create features related to groups of soil types.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    families_feat : optional list of str, default=None
        List of families features to compute. The name of a binary feature is expected to be either
        "Ratake", "Vanet", "Catamount", "Leighan", "Bullwark", "Como", "Moran" or "Other".

    excl_feat : optional list of str, default=None
        List of features names to drop.

    Raises
    ------
    ValueError
        If the family feature is unsupported.
    """
    if families_feat is None or len(families_feat) == 0:
        return

    print("\nComputing families features...")

    soil_type_feat = [f"Soil_Type{i}" for i in range(1, 41)]
    soil_type_feat = [x for x in soil_type_feat if x not in excl_feat]

    train_soil_types = train_df[soil_type_feat].idxmax(axis=1).str[9:].astype(int)
    if test_df is not None:
        test_soil_types = test_df[soil_type_feat].idxmax(axis=1).str[9:].astype(int)

    for family in families_feat:

        feat_name = family + "_Family_Type"

        try:

            family_dict = {i: 1 if i in FAMILIES[family] else 0 for i in range(1, 41)}

        except KeyError:

            err_msg = f"Unknown family {family}."
            raise ValueError(err_msg)  # pylint: disable=raise-missing-from

        train_df[feat_name] = train_soil_types.map(family_dict)
        if test_df is not None:
            test_df[feat_name] = test_soil_types.map(family_dict)

    print(f"Number of families features: {len(families_feat)}.")


def create_soil_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    soil_feat: Optional[List[str]] = None, excl_feat: Optional[List[str]] = None
):
    """Auxiliary function to create features related to groups of soil types.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    soil_feat : optional list of str, default=None
        List of soil types features to compute. The name of a binary feature is expected to be
        either "Stony", "Rubly" or "Other".

    excl_feat : optional list of str, default=None
        List of features names to drop.

    Raises
    ------
    ValueError
        If the soil type feature is unsupported.
    """
    if soil_feat is None or len(soil_feat) == 0:
        return

    print("\nComputing soil types features...")

    soil_type_feat = [f"Soil_Type{i}" for i in range(1, 41)]
    soil_type_feat = [x for x in soil_type_feat if x not in excl_feat]

    train_soil_types = train_df[soil_type_feat].idxmax(axis=1).str[9:].astype(int)
    if test_df is not None:
        test_soil_types = test_df[soil_type_feat].idxmax(axis=1).str[9:].astype(int)

    for soil_type in soil_feat:

        feat_name = soil_type + "_Soil_Type"

        try:

            soil_type_dict = {i: 1 if i in SOIL_TYPES[soil_type] else 0 for i in range(1, 41)}

        except KeyError:

            err_msg = f"Unknown soil type {soil_type}."
            raise ValueError(err_msg)  # pylint: disable=raise-missing-from

        train_df[feat_name] = train_soil_types.map(soil_type_dict)
        if test_df is not None:
            test_df[feat_name] = test_soil_types.map(soil_type_dict)

    print(f"Number of soil types features: {len(soil_feat)}.")
