"""Compute polynomial features by using the `scikit-learn` API."""


from typing import List, Optional

from pandas import DataFrame

from sklearn.preprocessing import PolynomialFeatures


def apply_poly(df: DataFrame, feat_name: str):
    """Compute a power feature.

    Parameters
    ----------
    df : DataFrame
        A dataframe containing the corresponding column, either "tags" if **summary** is True,
        or "d_tags" if **summary** is False.

    feat_name : str
        Name of the new feature to append to the dataframe. The feat_name must be of the form
        "A" or "A^x". A is the name of an existing feature in **df** and x must be an int.

    Raises
    ------
    ValueError
        If the power to apply is not an int.
    """
    if feat_name not in df.columns:

        feat_split = feat_name.split("^")

        if len(feat_split) <= 1:
            return

        initial_feat_name = feat_split[0]
        pw = feat_split[1]

        if not pw.isdigit():

            err_msg = f"From feature {feat_name}, non-integer power {pw}."
            raise ValueError(err_msg)

        df[feat_name] = df[initial_feat_name] ** int(pw)


def create_poly_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    fixed_poly_feat: Optional[List[str]] = None, poly_feat: Optional[List[str]] = None,
    poly_degree: int = 2
):
    """Auxiliary function to create polynomial features.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    fixed_poly_feat : optional list of str, default=None
        List of specific polynomial features. A fixed polynomial feature must be of the form
        "A B C". A, B and C can be features or powers of features, and their product will be
        computed.

    poly_feat : optional list of str, default=None
        List of polynomial features to compute of which interaction terms will be computed.

    poly_degree : int, default=2
        Define the degree until which products and powers of features are computed. If 1 or less,
        there will be no polynomial features.
    """
    if (
        (fixed_poly_feat is None or len(fixed_poly_feat) == 0) and
        (poly_feat is None or len(poly_feat) == 0)
    ):
        return

    print("\nComputing polynomial features...")

    init_feat = set(train_df.columns)
    new_feat = set()

    # Work on fixed polynomial features
    if fixed_poly_feat is not None:

        for feat_name in fixed_poly_feat:

            feat_split = feat_name.split(" ")

            for subfeat_name in feat_split:

                apply_poly(train_df, subfeat_name)
                if test_df is not None:
                    apply_poly(test_df, subfeat_name)

            train_df[feat_name] = train_df[feat_split].product(axis=1)
            if test_df is not None:
                test_df[feat_name] = test_df[feat_split].product(axis=1)

            new_feat.add(feat_name)

    # Work on interaction features
    if poly_degree >= 2 and poly_feat is not None and len(poly_feat) > 0:

        # Fit the handler and find the names of the new features
        pf = PolynomialFeatures(degree=(2, poly_degree), include_bias=False)

        pf.fit(train_df[poly_feat])

        new_poly_feat = pf.get_feature_names_out(poly_feat)

        # Actually compute these new features
        train_df[new_poly_feat] = pf.fit_transform(train_df[poly_feat])
        if test_df is not None:
            test_df[new_poly_feat] = pf.transform(test_df[poly_feat])

        new_feat = new_feat.union(set(new_poly_feat))

    # Drop intermediary features
    inter_feat = set(train_df.columns).difference(init_feat.union(new_feat))

    train_df.drop(columns=inter_feat, inplace=True)
    if test_df is not None:
        test_df.drop(columns=inter_feat, inplace=True)

    print(f"Number of polynomial features: {len(new_feat)}.")
