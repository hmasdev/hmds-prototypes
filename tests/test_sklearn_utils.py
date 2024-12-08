import pytest
from mlprotobox.sklearn_utils import Identity


@pytest.mark.parametrize(
    'X,y',
    [
        ([[1, 2], [3, 4]], None),
        ([[1, 2], [3, 4]], [5, 6]),
    ],
)
def test_identity_fit_transform(X, y):
    identity = Identity()
    assert identity.fit_transform(X, y) == X


@pytest.mark.parametrize(
    'X,y',
    [
        ([[1, 2], [3, 4]], None),
        ([[1, 2], [3, 4]], [5, 6]),
    ],
)
def test_identity_inverse_transform(X, y):
    identity = Identity()
    assert identity.fit_transform(X, y) == X
    assert identity.inverse_transform(X) == X
