"""
src/generate_data.py
~~~~~~~~~~~~~~~~~~~~
Generate a synthetic time-series data set for walk-forward benchmarks.

• 1 year + 3 months of *hourly* observations  →  455 days → 10 920 rows
• 200 base features, of which ~10 % drive the target
• Add 24 lagged copies of every feature (plus the current-time column),
  yielding    (24 lags + current) × 200 = 5 000 predictors per row.
• Drop the first 24 rows so every sample has a full lag history.

The function returns:
    X : ndarray, shape (10 896, 5 000)
    y : ndarray, shape (10 896,)

Run this file directly (`python -m src.generate_data`) for a quick
sanity check that prints the shapes of X and y.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def create_dataset(
    hours: int | None = None,
    n_features: int = 200,
    n_lags: int = 24,
    random_state: int | None = None,
    fraction_relevant: float = 0.10,
    include_current: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a multivariate time series and add lagged columns.

    Parameters
    ----------
    hours
        Total number of hourly observations to simulate.
        *None* ≡ 1 year + 3 months ≈ 455 days.
    n_features
        Number of raw features at each timestamp.
    n_lags
        Maximum lag (in hours) to append for every feature.
    random_state
        Seed for NumPy's RNG for reproducibility.
    fraction_relevant
        Fraction of features that truly influence the target.
    include_current
        If *True*, keep the t-0 (current) features in the final matrix.

    Returns
    -------
    X, y
        2-D feature matrix and 1-D target vector, both float32.
        First ``n_lags`` rows are dropped so every row has complete lags.
    """
    # ------------------------------------------------------------------
    # 1. Create the raw feature matrix
    # ------------------------------------------------------------------
    if hours is None:
        days = 365 + 90  # 1 year + 3 months
        hours = days * 24

    rng = np.random.default_rng(random_state)
    X_raw = rng.standard_normal((hours, n_features)).astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Build the target y:   linear combo of ~10 % features + daily seasonality + noise
    # ------------------------------------------------------------------
    n_relevant = max(1, int(fraction_relevant * n_features))
    relevant_idx = rng.choice(n_features, n_relevant, replace=False)
    beta = rng.uniform(-1.0, 1.0, size=n_relevant)  # weights

    t = np.arange(hours)
    daily_cycle = np.sin(2.0 * np.pi * (t % 24) / 24)  # simple 24 h seasonality

    y = (
        X_raw[:, relevant_idx] @ beta  # signal
        + 0.3 * daily_cycle  # seasonality
        + rng.standard_normal(hours) * 0.1  # iid noise
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. Append lagged copies of every feature
    # ------------------------------------------------------------------
    start_lag = 0 if include_current else 1
    lags = range(start_lag, n_lags + 1)  # inclusive
    lagged_views = [X_raw[n_lags - k : hours - k] for k in lags]
    X_lagged = np.concatenate(
        lagged_views, axis=1
    )  # shape: (hours-n_lags, (|lags|)·n_features)

    # Match y to the lagged matrix (drop first n_lags rows)
    y = y[n_lags:]

    return X_lagged.astype(np.float32), y.astype(np.float32)


# ----------------------------------------------------------------------
# Command-line usage – quick smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    X, y = create_dataset(random_state=42)
    print(f"X shape: {X.shape}")  # -> (10896, 5000)
    print(f"y shape: {y.shape}")  # -> (10896,)
