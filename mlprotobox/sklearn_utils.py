import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin


class Identity(BaseEstimator, TransformerMixin):
    """A transformer that does nothing.

    This transformer is useful for debugging purposes and as a placeholder
    in pipelines where no transformation is needed.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Examples
    --------
    >>> import numpy as np
    >>> from hmds_prototypes import Identity
    >>> X = np.array([[1, 2], [3, 4]])
    >>> identity = Identity()
    >>> identity.fit_transform(X)
    array([[1, 2],
           [3, 4]])
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> 'Identity':  # noqa
        """Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,) or None, default=None
            The target variable.

        Returns
        -------
        self : Identity
            The transformer itself.
        """
        return self

    def transform(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> npt.ArrayLike:  # noqa
        """Transform the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,) or None, default=None
            The target variable.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            The input data, unchanged.
        """
        return X

    def inverse_transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Inverse transform the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            The input data, unchanged.
        """
        return X
