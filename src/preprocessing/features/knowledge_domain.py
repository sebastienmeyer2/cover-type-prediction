"""Compute knowledge-domain features."""


from typing import List, Optional

from pandas import DataFrame


def create_kd_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    kd_feat: Optional[List[str]] = None
):
    """Auxiliary function to create stat features.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing the initial features.

    test_df : optional DataFrame, default=None
        Test dataframe containing the initial features.

    kd_feat : optional list of str, default=None
        List of knowledge-domain features. The name of a knowledge-domain feature must be either
        "Distance_To_Hydrology", "Mean_Distance_To_Points_Of_Interest",
        "Elevation_Shifted_Horizontal_Distance_To_Hydrology",
        "Elevation_Shifted_Vertical_Distance_To_Hydrology" or
        "Elevation_Shifted_Horizontal_Distance_To_Roadways".
    """
    if kd_feat is None or len(kd_feat) == 0:
        return

    print("\nComputing knowledge-domain features...")

    dfs = [train_df]
    if test_df is not None:
        dfs.append(test_df)

    for df in dfs:

        for feat_name in kd_feat:

            if feat_name == "Distance_To_Hydrology":

                h_d = "Horizontal_Distance_To_Hydrology"
                v_d = "Vertical_Distance_To_Hydrology"
                df[feat_name] = (df[h_d].pow(2) + df[v_d].pow(2)).pow(0.5)

            elif feat_name == "Mean_Distance_To_Points_Of_Interest":

                hyd = "Horizontal_Distance_To_Hydrology"
                fpt = "Horizontal_Distance_To_Fire_Points"
                rdw = "Horizontal_Distance_To_Roadways"
                df[feat_name] = df[hyd] + df[fpt] + df[rdw]

            elif feat_name == "Elevation_Shifted_Horizontal_Distance_To_Hydrology":

                df[feat_name] = df["Elevation"] - 0.2*df["Horizontal_Distance_To_Hydrology"]

            elif feat_name == "Elevation_Shifted_Vertical_Distance_To_Hydrology":

                df[feat_name] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

            elif feat_name == "Elevation_Shifted_Horizontal_Distance_To_Roadways":

                df[feat_name] = df["Elevation"] - 0.02*df["Horizontal_Distance_To_Roadways"]

            else:

                err_msg = f"Unknown knowledge-domain feature {feat_name}."
                raise ValueError(err_msg)

    print(f"Number of knowledge-domain features: {len(kd_feat)}.")
