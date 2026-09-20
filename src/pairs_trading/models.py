from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngleGrangerResult:
    intercept: float
    beta: float
    adf_t_stat: float
    residual: pd.Series


@dataclass(frozen=True)
class KalmanResult:
    intercept: pd.Series
    beta: pd.Series
    spread: pd.Series


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients, y - design @ coefficients


def engle_granger(y: pd.Series, x: pd.Series) -> EngleGrangerResult:
    pair = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(pair) < 30:
        raise ValueError("at least 30 observations are required")
    coefficients, residual_values = _ols(pair["y"].to_numpy(float), pair["x"].to_numpy(float))
    residual = pd.Series(residual_values, index=pair.index, name="residual")
    delta = residual.diff().dropna().to_numpy()
    lagged = residual.shift(1).dropna().to_numpy()
    adf_coefficients, adf_residuals = _ols(delta, lagged)
    design = np.column_stack([np.ones(len(lagged)), lagged])
    variance = float(np.dot(adf_residuals, adf_residuals) / (len(lagged) - 2))
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_error = np.sqrt(max(covariance[1, 1], 0.0))
    adf_t = float(adf_coefficients[1] / standard_error) if standard_error else float("nan")
    return EngleGrangerResult(float(coefficients[0]), float(coefficients[1]), adf_t, residual)


def kalman_hedge_ratio(y: pd.Series, x: pd.Series, process_variance: float = 1e-5, observation_variance: float = 1e-3) -> KalmanResult:
    pair = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(pair) < 2:
        raise ValueError("at least two observations are required")
    state = np.zeros(2)
    covariance = np.eye(2)
    transition_noise = np.eye(2) * process_variance
    intercepts, betas, spreads = [], [], []
    for observed_y, observed_x in pair.itertuples(index=False):
        design = np.array([1.0, observed_x])
        predicted_covariance = covariance + transition_noise
        prediction = float(design @ state)
        innovation = float(observed_y - prediction)
        innovation_variance = float(design @ predicted_covariance @ design + observation_variance)
        gain = predicted_covariance @ design / innovation_variance
        state = state + gain * innovation
        covariance = predicted_covariance - np.outer(gain, design @ predicted_covariance)
        intercepts.append(state[0]);betas.append(state[1]);spreads.append(innovation)
    index = pair.index
    return KalmanResult(pd.Series(intercepts, index=index, name="intercept"), pd.Series(betas, index=index, name="beta"), pd.Series(spreads, index=index, name="spread"))
