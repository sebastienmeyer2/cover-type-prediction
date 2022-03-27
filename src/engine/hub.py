"""Initialize a model based on its name and parameters."""


from typing import Any, Dict


from engine.models.base import BaseEstimator


BASE_MODELS_NAMES = ["rfc", "etc", "xgboost", "lightgbm", "catboost", "logreg", "stacking"]


def create_model(model_name: str, params: Dict[str, Any]) -> BaseEstimator:
    """Create a model.

    Parameters
    ----------
    model_name : str
        The name of model following project usage. See README.md for more information about
        available models.

    params : dict of {str: any}
        A dictionary of parameters for chosen **model_name**.

    Returns
    -------
    model : `BaseEstimator`
        Corresponding model from the catalogue.

    Raises
    ------
    ValueError
        If the **model_name** is not supported.
    """
    if model_name in BASE_MODELS_NAMES:

        model = BaseEstimator(model_name, params)

    else:

        err_msg = f"Unknown model {model_name}."
        raise ValueError(err_msg)

    return model
