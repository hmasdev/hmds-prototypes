from itertools import combinations
from typing import Callable, Iterable
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone, TransformerMixin, RegressorMixin  # noqa
try:
    # sklearn>=1.6
    from sklearn.utils._tags import Tags
except ImportError:
    Tags = None
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    check_X_y,
)
from mlprotobox.sklearn_utils import Identity


def _squared_diff_weight(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.ArrayLike:
    return (x - y) ** 2  # type: ignore


class PairwiseMoE(BaseEstimator, RegressorMixin):
    """A regressor that learns the relationship between pairs of features.

    This regressor learns the relationship between pairs of features by
    fitting a model to the target variable conditioned on the ratio of the
    two features. The target variable is created by dividing the target
    variable by the ratio of the two features. The regressor fits a model
    to the target variable conditioned on the ratio of the two features.

    This regressor predicts the target variable with :math:`D(D-1)` models by

    .. math::

        \\hat{y} = \\frac{1}{D(D-1)} \\sum_{1\\leq i,j\\leq D, i\\neq j} \\left( x_j + model_{ij}(x) \\times (x_i-x_j) \\right)

    This regressor is trained by minimizing the following loss function:

        L := \\frac{1}{N} \\sum_{1\\leq i,j\\leq D, i\\neq j} \\sum_{n=1}^N | y_n - x_{nj} - model_{ij}(x_n) \\times (x_{ni}-x_{nj}) |^2

        = \\frac{1}{N} \\sum_{1\\leq i,j\\leq D, i\\neq j} \\sum_{n=1}^N (x_{ni}-x_{nj})^2 | \\frac{y_n - x_{nj}}{x_{ni}-x_{nj}} - model_{ij}(x_n) |^2

        = \\frac{1}{N} \\sum_{1\\leq i,j\\leq D, i\\neq j} \\sum_{n=1}^N w_{nij} | \\varepsilon_{nij} - model_{ij}(x_n) |^2

    where :math:`N` is the number of samples, :math:`D` is the number of expert features,
    :math:`x` is the input data, :math:`y` is the target variable, :math:`model_{ij}` is the model
    fitted to the target variable conditioned on the ratio of the two features, :math:`w_{nij}` is
    the sample weight, and :math:`\\varepsilon_{nij}` is the alternative target variable.

    Parameters
    ----------
    base_estimator : BaseEstimator
        The base model to fit to the target variable conditioned on the
        ratio of the two features.

    experts_columns : Iterable[int] or None, default=None
        The indices of the expert columns, where an expert column is a column
        which is output by a expert model, like the first layer of a
        StackingRegressor. If None, all
        columns are used.

    invertible_transformer : TransformerMixin, default=Identity()
        The transformer to preprocess the target variable. This transformer
        should be invertible.

    weight_func : Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike], default=_squared_diff_weight  # noqa
        The function to compute the sample weights. The function should
        take two arrays of the same shape and return an array of the same
        shape.

    clip_alternative_target : bool, default=False
        Whether to clip the alternative target variable to the range [0, 1].

    Attributes
    ----------
    base_estimator : BaseEstimator
        The base model to fit to the target variable conditioned on the
        ratio of the two features.

    experts_columns_ : tuple[int]
        The indices of the expert columns. An expert column is a column
        which is output by a expert model, like the first layer of a
        StackingRegressor.

    n_experts_columns_ : int
        The number of expert columns.

    models_ : dict[tuple[int, int], BaseEstimator]
        The models fitted to the target variable conditioned on the ratio
        of the two features.

    invertible_transformers_ : dict[tuple[int, int], TransformerMixin]
        The transformers used to preprocess the target variable.

    weight_func : Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]
        The function to compute the sample weights.

    clip_alternative_target : bool
        Whether to clip the alternative target variable to the range [0, 1].
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        experts_columns: Iterable[int] | None = None,
        invertible_transformer: TransformerMixin = Identity(),
        weight_func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike] = _squared_diff_weight,  # noqa: E501
        clip_alternative_target: bool = False,
    ):
        super().__init__()
        self.base_estimator = base_estimator
        self.experts_columns = experts_columns
        self.invertible_transformer = invertible_transformer
        self.weight_func = weight_func
        self.clip_alternative_target = clip_alternative_target

    def __sklearn_tags__(self) -> "Tags":
        tags = super().__sklearn_tags__()
        tags.estimator_type = 'regressor'
        # TODO: check why estimator_type is not automatically set to 'regressor'.  # noqa
        return tags

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
    ) -> 'PairwiseMoE':
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target variable.

        sample_weight : array-like of shape (n_samples,) or None, default=None
            The sample weights.

        Returns
        -------
        self : PairwiseMoE
            The regressor itself.
        """
        X, y = check_X_y(X, y)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # init
        if self.experts_columns is None:
            self.experts_columns_ = tuple(range(X.shape[1]))  # type: ignore # noqa
        else:
            self.experts_columns_ = tuple(self.experts_columns)
        self.n_experts_columns_ = len(self.experts_columns_)
        self.models_: dict[tuple[int, int], BaseEstimator] = {}
        self.invertible_transformers_: dict[tuple[int, int], TransformerMixin] = {}  # noqa: E501

        # fit
        for i, j in combinations(self.experts_columns_, 2):

            # init
            xi = X[:, i]  # type: ignore
            xj = X[:, j]  # type: ignore
            model_ij = clone(self.base_estimator)
            transformer_ij = clone(self.invertible_transformer)
            model_ji = clone(self.base_estimator)
            transformer_ji = clone(self.invertible_transformer)

            # create target
            epsilon_ij = (y - xj) / (xi - xj)  # type: ignore
            epsilon_ji = (y - xi) / (xj - xi)  # type: ignore
            epsilon_ij = np.nan_to_num(epsilon_ij, nan=0.5, posinf=0.5, neginf=0.5)  # noqa
            epsilon_ji = np.nan_to_num(epsilon_ji, nan=0.5, posinf=0.5, neginf=0.5)  # noqa
            if self.clip_alternative_target:
                epsilon_ij = np.clip(epsilon_ij, 0, 1)
                epsilon_ji = np.clip(epsilon_ji, 0, 1)
            weight_ij = self.weight_func(xi, xj)
            weight_ji = self.weight_func(xj, xi)
            weight_ij = np.nan_to_num(weight_ij, nan=0, posinf=0, neginf=0)  # noqa
            weight_ji = np.nan_to_num(weight_ji, nan=0, posinf=0, neginf=0)  # noqa

            # TODO: transform weight_ij, weight_ji into a np.ndarray
            #       because they may not have ".reshape" method

            # preprocess targets
            z_ij = transformer_ij.fit_transform(epsilon_ij.reshape(-1, 1))
            z_ji = transformer_ji.fit_transform(epsilon_ji.reshape(-1, 1))

            # fit
            if sample_weight is None:
                model_ij.fit(X, z_ij, sample_weight=weight_ij)
                model_ji.fit(X, z_ji, sample_weight=weight_ji)
            else:
                # TODO: check if sample_weight can be multiplied by weight_ij, weight_ji  # noqa
                model_ij.fit(X, z_ij, sample_weight=sample_weight*weight_ij)
                model_ji.fit(X, z_ji, sample_weight=sample_weight*weight_ji)

            # store
            self.models_[(i, j)] = model_ij
            self.invertible_transformers_[(i, j)] = transformer_ij
            self.models_[(j, i)] = model_ji
            self.invertible_transformers_[(j, i)] = transformer_ji

        return self

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Predict the target variable.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target variable.
        """
        check_is_fitted(self)
        X = check_array(X)
        predictions: dict[tuple[int, int], np.ndarray] = {
            (i, j): X[:, j] + transformer_ij.inverse_transform(model_ij.predict(X).reshape(-1, 1)).flatten() * (X[:, i] - X[:, j])  # type: ignore # noqa
            for (i, j), model_ij in self.models_.items()
            if (transformer_ij := self.invertible_transformers_[(i, j)]) is not None  # noqa
        }
        prediction: np.ndarray = sum(predictions.values()) / len(predictions)  # type: ignore # noqa
        return prediction

    def get_weight(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Predict the sample weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        sample_weight_pred : array-like of shape (n_samples, n_experts_columns_)
            The predicted sample weights.
        """  # noqa
        check_is_fitted(self)
        X = check_array(X)
        epsilons = np.zeros((X.shape[0], self.n_experts_columns_))  # type: ignore # noqa
        for (i, j), model in self.models_.items():
            transformer = self.invertible_transformers_[(i, j)]
            epsilon = transformer.inverse_transform(model.predict(X).reshape(-1, 1)).flatten()  # noqa
            epsilons[:, i] += epsilon
            epsilons[:, j] += 1 - epsilon
        return epsilons / len(self.models_)
