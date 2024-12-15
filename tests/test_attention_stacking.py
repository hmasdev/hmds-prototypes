from itertools import product
from typing import Callable
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import pytest
from sklearn.base import is_regressor
from sklearn.linear_model import LinearRegression
from mlprotobox.attention_stacking import (
    _squared_diff_weight,
    PairwiseMoE,
)


@pytest.mark.parametrize(
    'X,y,experts_columns,weight_func,clip_alternative_target',  # noqa
    [
        sum(tups, ())
        for tups in product(
            [
                ([[1, 2, 3], [3, 4, 5]], [5, 6]),
                (np.array([[1, 2, 3], [3, 4, 5]]), np.array([5, 6])),
                (pd.DataFrame([[1, 2, 3], [3, 4, 5]]), pd.Series([5, 6])),  # noqa
                (pl.DataFrame([[1, 2, 3], [3, 4, 5]]).transpose(), pl.Series([5, 6])),  # noqa
            ],
            [
                (None, _squared_diff_weight, False),
                ([0, 1], _squared_diff_weight, False),
                (None, lambda x, y: x + y, False),
                (None, _squared_diff_weight, True),
            ],
        )
    ]
)
def test_pairwise_attention_regressor_init(
    X: np.ndarray | pd.DataFrame | pl.DataFrame,
    y: np.ndarray | pd.Series | pl.Series,
    experts_columns: list[int] | None,
    weight_func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike] | None,  # noqa
    clip_alternative_target: bool,
):
    model = PairwiseMoE(
        LinearRegression(),
        experts_columns=experts_columns,
        weight_func=weight_func,  # type: ignore
        clip_alternative_target=clip_alternative_target,
    )
    assert is_regressor(model)
    model.fit(X, y, sample_weight=None)
    model.fit(X, y, sample_weight=np.arange(len(X)))
    model.predict(X)
    if experts_columns is None:
        # FIXME: hard-coded
        assert model.get_weight(X).shape[1] == 3  # type: ignore
    else:
        assert model.get_weight(X).shape[1] == len(experts_columns)  # type: ignore # noqa
