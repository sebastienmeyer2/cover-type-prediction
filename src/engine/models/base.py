"""Initialize, fit and predict wrapper for general models."""


from typing import Any, Dict, List, Union

import warnings

from numpy import ndarray
from pandas import DataFrame

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore")


class BaseEstimator():
    """A wrapper for machine learning models."""

    def __init__(self, model_name: str, params: Dict[str, Any]):
        """The constructor of the class.

        Parameters
        ----------
        model_name : str
            The name of model following project usage. See README.md for more information.

        params : Dict[str, Any]
            A dictionary of parameters for chosen **model_name**. It contains all parameters to
            initialize and fit the model.
        """
        self.model_name = model_name

        # Model parameters
        self.model_params = params

        # Instantiate model
        self.model: Any

        # Label prediction
        if model_name == "rfc":
            self.model = RandomForestClassifier(**self.model_params)
        elif model_name == "etc":
            self.model = ExtraTreesClassifier(**self.model_params)
        elif self.model_name == "ova":
            self.model = OneVsRestClassifier(ExtraTreesClassifier(**self.model_params), n_jobs=-1)
        elif model_name == "xgboost":
            self.model = XGBClassifier(**self.model_params)
        elif model_name == "lightgbm":
            self.model = LGBMClassifier(**self.model_params)
        elif model_name == "catboost":
            self.model = CatBoostClassifier(**self.model_params)
        elif model_name == "logreg":
            self.model = LogisticRegression(**self.model_params)
        elif self.model_name == "stacking":
            final_est = LogisticRegression(
                solver="newton-cg", multi_class="auto",
                random_state=self.model_params["random_state"]
            )
            self.model = StackingClassifier(
                create_est(self.model_params),
                final_estimator=final_est, cv=5, n_jobs=-1, passthrough=True
            )
        elif self.model_name == "resampling":
            self.model = create_resampling_pipeline(self.model_params)
        else:
            raise ValueError("{} is not implemented. Check available models.".format(model_name))

    def fit(self, x_train: Union[ndarray, DataFrame], y_train: Union[ndarray, DataFrame]):
        """Wrapper of fit method.

        Parameters
        ----------
        x_train : Union[ndarray, DataFrame]
            Training features.

        y_train : Union[ndarray, DataFrame]
            Training labels.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_eval: Union[ndarray, DataFrame]) -> Union[ndarray, DataFrame]:
        """Wrapper of predict method.

        Parameters
        ----------
        x_eval : Union[ndarray, DataFrame]
            Evaluation features.

        Returns
        -------
        y_pred : Union[ndarray, DataFrame]
            Predicted labels on evaluation set.
        """
        y_pred = self.model.predict(x_eval)

        return y_pred

    def predict_proba(self, x_eval: Union[ndarray, DataFrame]) -> Union[ndarray, DataFrame]:
        """Wrapper of predict_proba method.

        Parameters
        ----------
        x_eval : Union[ndarray, DataFrame]
            Evaluation features.

        Returns
        -------
        y_pred_probs : Union[ndarray, DataFrame]
            Predicted probabilities on evaluation set.
        """
        y_pred_probs = self.model.predict_proba(x_eval)

        return y_pred_probs


def create_est(params: Dict[str, Any]) -> List[Any]:
    """Build up estimators for stacking.

    Parameters
    ----------
    params : dict of {str: any}
        A dictionary of parameters for stacking. It contains all parameters to initialize and fit
        the model.

    Returns
    -------
    est : list of any
        The complete sequence of estimators for stacking.
    """
    random_state = params["random_state"]

    # For now, manually estimated parameters
    etc = ExtraTreesClassifier(n_estimators=267, criterion="gini", max_depth=None,
                               min_samples_split=3, min_samples_leaf=1, max_features="auto",
                               max_leaf_nodes=None, min_impurity_decrease=8.08e-9, bootstrap=False,
                               ccp_alpha=3.11e-7, random_state=random_state, n_jobs=-1)
    rfc = RandomForestClassifier(n_estimators=156, criterion="entropy", min_samples_split=2,
                                 min_samples_leaf=1, max_features="auto", bootstrap=True,
                                 ccp_alpha=1.26e-7,  max_samples=None, random_state=random_state,
                                 n_jobs=-1)
    lgbm = LGBMClassifier(n_estimators=177, num_leaves=73, min_split_gain=1.22e-5,
                          min_child_weight=3.82e-5, min_child_samples=12, subsample=0.90,
                          subsample_freq=2, reg_alpha=1.14e-6, reg_lambda=5.37e-5,
                          random_state=random_state, n_jobs=-1)
    # xgb = XGBClassifier(n_estimators=250, use_label_encoder=False, random_state=random_state,
    #                     gamma=5.64e-2, subsample=0.78, colsample_bytree=0.94,
    #                     colsample_bylevel=0.85, colsample_bynode=0.76, max_delta_step=0.64,
    #                     reg_alpha=1.46e-7, reg_lambda=1.39e-5, grow_policy="depthwise")

    est = [("rfc", rfc), ("lgbm", lgbm), ("extra", etc)]

    return est


def create_resampling_pipeline(params: Dict[str, Any]) -> Pipeline:
    """Build up estimators for stacking.

    Parameters
    ----------
    params : Dict[str, Any]
        A dictionary of parameters for the feed-forward network. It contains all parameters to
        initialize and fit the model.

    Returns
    -------
    pipeline : Pipeline
        A complete *scikit-learn* `Pipeline` for resampling before fitting a model.
    """
    seed = params["random_state"]

    # Prepare the undersampling operation
    if "clust_1" not in params:
        error_msg = "Building a Pipeline requires a repartition for Undersampling."
        raise ValueError(error_msg)

    clust_dict = {i: params["clust_{}".format(i+1)] for i in range(7)}

    clust = ClusterCentroids(
        sampling_strategy=clust_dict, random_state=seed, estimator=KMeans(random_state=seed)
    )

    # Prepare the oversampling operation
    if "smote_1" not in params:
        error_msg = "Building a Pipeline requires a repartition for Oversampling."
        raise ValueError(error_msg)

    smote_dict = {i: params["smote_{}".format(i+1)] for i in range(7)}

    smote = SMOTE(sampling_strategy=smote_dict, random_state=seed, n_jobs=-1)

    # Prepare the model
    sampling_params = ["clust_{}".format(i+1) for i in range(7)] \
        + ["smote_{}".format(i+1) for i in range(7)]
    model_params = {k: v for k, v in params.items() if k not in sampling_params}
    model = ExtraTreesClassifier(**model_params)

    # Build the Pipeline
    pipeline = Pipeline([("undersampling", clust), ("oversampling", smote), ("model", model)])

    return pipeline
