"""
src/walkforward_cpu.py
~~~~~~~~~~~~~~~~~~~~~~
CPU benchmark of walk-forward XGBoost using the common engine.
"""

from __future__ import annotations

import numpy as np
from generate_data import create_dataset
from walkforward_base import walk_forward_predict
from time import time


def main() -> None:
    # ------------------------------------------------------------------
    # 1. generate synthetic data
    # ------------------------------------------------------------------
    X, y = create_dataset(random_state=42)

    # ------------------------------------------------------------------
    # 2. CPU-oriented XGBoost parameters
    # ------------------------------------------------------------------
    params_cpu = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "nthread": 0,  # use all logical cores
    }

    preds, actuals = walk_forward_predict(
        X=X,
        y=y,
        lookback=365 * 24,
        horizon=24,
        params=params_cpu,
        num_boost_round=200,
        verbose=True,
    )

    overall_rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    print(f"\n=== CPU overall RMSE: {overall_rmse:.4f} ===")


if __name__ == "__main__":
    # Measure execution time
    start_time = time()
    main()
    end_time = time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
