"""Panel repeatability statistics used as data-release gates."""

from __future__ import annotations

import numpy as np


def krippendorff_alpha_nominal(values: np.ndarray) -> float:
    """Krippendorff's alpha for nominal ratings with missing values."""
    ratings = np.asarray(values, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("Ratings must be units by assessors")
    disagreements = 0.0
    observed_values = []
    for row in ratings:
        valid = row[np.isfinite(row)]
        if len(valid) < 2:
            continue
        observed_values.extend(valid.tolist())
        # Ordered coincidence pairs are weighted by 1/(m_u - 1), so units
        # with more raters do not contribute quadratically more weight.
        _, counts = np.unique(valid, return_counts=True)
        disagreements += (len(valid) ** 2 - float((counts ** 2).sum())) / (len(valid) - 1)
    pooled = np.asarray(observed_values)
    if len(pooled) < 2:
        return float("nan")
    observed = disagreements / len(pooled)
    _, counts = np.unique(pooled, return_counts=True)
    expected = (len(pooled) ** 2 - float((counts ** 2).sum())) / (len(pooled) * (len(pooled) - 1))
    if expected == 0:
        return float("nan")
    return float(1.0 - observed / expected)


def icc_2k(values: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement ICC(2,k) for complete ratings."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Intensity ratings must be units by assessors")
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    n, k = matrix.shape if matrix.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_columns = n * ((column_means - grand) ** 2).sum()
    residual = matrix - row_means[:, None] - column_means[None, :] + grand
    ss_error = (residual**2).sum()
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")
