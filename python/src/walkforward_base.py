"""
src/walkforward_base.py
~~~~~~~~~~~~~~~~~~~~~~~
Rolling (walk-forward) training / forecasting for XGBoost.

Core idea
---------
• Use a fixed `lookback` window (e.g. 8760 rows = 365 days of hours)
  to train at each step.
• Predict the next `horizon` rows (e.g. 24 hours).
• Slide the window forward by `horizon` and repeat until
  `X` is exhausted.

The heavy lifting is done *once* here so both CPU and GPU scripts
are just slim wrappers with different parameter dictionaries.
"""

from __future__ import annotations

from typing import Tuple, Dict, List

import numpy as np
import xgboost as xgb
from tqdm import tqdm


def walk_forward_predict(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int,
    params: Dict,
    num_boost_round: int = 200,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train / predict in rolling fashion.

    Parameters
    ----------
    X, y
        Full design matrix and target vector (NumPy `float32`).
    lookback
        Size of the training window (number of rows).
    horizon
        Number of rows to predict each iteration.
    params
        Parameters forwarded to `xgboost.train`.
    num_boost_round
        Number of boosting iterations.
    verbose
        If *True*, print step-level RMSE.

    Returns
    -------
    preds, actuals
        1-D arrays containing the concatenated forecasts and the
        matching ground-truth values.
    """
    assert X.shape[0] == y.shape[0], "X and y must have same length"

    preds: List[np.ndarray] = []
    actuals: List[np.ndarray] = []

    start = lookback
    end = X.shape[0]

    step = 0
    for start in tqdm(
        range(lookback, end - horizon + 1, horizon), desc="Walk-forward steps"
    ):
        # ------------------------------------------------------------------
        # 1. fit
        # ------------------------------------------------------------------
        train_X = X[start - lookback : start]
        train_y = y[start - lookback : start]
        dtrain = xgb.DMatrix(train_X, label=train_y)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False,
        )

        # ------------------------------------------------------------------
        # 2. predict next horizon
        # ------------------------------------------------------------------
        test_X = X[start : start + horizon]
        dtest = xgb.DMatrix(test_X)
        y_pred = booster.predict(dtest)

        preds.append(y_pred)
        actual_slice = y[start : start + horizon]
        actuals.append(actual_slice)

        if verbose:
            rmse = float(np.sqrt(np.mean((y_pred - actual_slice) ** 2)))
            tqdm.write(
                f"[step {step:03d}] rows {start}-{start+horizon-1}: RMSE={rmse:.4f}"
            )

        step += 1
        start += horizon

    return np.concatenate(preds), np.concatenate(actuals)
