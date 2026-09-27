from __future__ import annotations

import math
from typing import Dict

import numpy as np
from skimage.metrics import structural_similarity


def _sam(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = reference.reshape(-1, reference.shape[-1])
    prediction = prediction.reshape(-1, prediction.shape[-1])
    numerator = np.sum(reference * prediction, axis=1)
    denominator = (
        np.linalg.norm(reference, axis=1)
        * np.linalg.norm(prediction, axis=1)
    )
    cosine = numerator / np.clip(denominator, 1e-12, None)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.mean(np.arccos(cosine)) * 180.0 / math.pi)


def _ergas(reference: np.ndarray, prediction: np.ndarray, scale: int) -> float:
    reference = reference.reshape(-1, reference.shape[-1])
    prediction = prediction.reshape(-1, prediction.shape[-1])
    rmse_bands = np.sqrt(np.mean((reference - prediction) ** 2, axis=0))
    means = np.mean(reference, axis=0)
    ratios = rmse_bands / np.clip(means, 1e-12, None)
    return float(100.0 / float(scale) * np.sqrt(np.mean(ratios ** 2)))


def _ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
    height, width, channels = reference.shape
    window = min(7, height, width)
    if window % 2 == 0:
        window -= 1
    window = max(window, 3)
    values = []
    for band in range(channels):
        values.append(
            structural_similarity(
                reference[..., band],
                prediction[..., band],
                data_range=1.0,
                win_size=window,
            )
        )
    return float(np.mean(values))


def _uiqi(reference: np.ndarray, prediction: np.ndarray) -> float:
    values = []
    for band in range(reference.shape[-1]):
        x = reference[..., band].reshape(-1).astype(np.float64)
        y = prediction[..., band].reshape(-1).astype(np.float64)
        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))
        var_x = float(np.var(x))
        var_y = float(np.var(y))
        covariance = float(np.mean((x - mean_x) * (y - mean_y)))
        numerator = 4.0 * covariance * mean_x * mean_y
        denominator = (
            (var_x + var_y)
            * (mean_x * mean_x + mean_y * mean_y)
        )
        values.append(numerator / max(abs(denominator), 1e-12))
    return float(np.mean(values))


def calculate_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    scale: int,
) -> Dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((reference - prediction) ** 2)))
    psnr = float("inf") if rmse == 0 else 10.0 * math.log10(1.0 / (rmse ** 2))
    return {
        "rmse": rmse,
        "psnr": psnr,
        "sam": _sam(reference, prediction),
        "ergas": _ergas(reference, prediction, scale),
        "ssim": _ssim(reference, prediction),
        "uiqi": _uiqi(reference, prediction),
    }

