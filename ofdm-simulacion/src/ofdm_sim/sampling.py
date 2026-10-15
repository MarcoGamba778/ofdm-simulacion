"""Adquisición de señales: Nyquist, aliasing, cuantización y resolución espectral."""

from __future__ import annotations

import numpy as np


def num_samples(duration: float, fs: float) -> int:
    """Tamaño de la muestra: N = T · fs."""
    return int(round(duration * fs))


def spectral_resolution(fs: float, n: int) -> float:
    """Resolución espectral: Δf = fs / N."""
    return fs / n


def satisfies_nyquist(fs: float, bandwidth: float) -> bool:
    """Criterio de Nyquist: fs >= 2B."""
    return fs >= 2.0 * bandwidth


def alias_frequency(f: float, fs: float) -> float:
    """Frecuencia aparente (en [0, fs/2]) de un tono de frecuencia f muestreado a fs."""
    return float(abs(f - fs * np.round(f / fs)))


def quantize(x: np.ndarray, n_bits: int, x_max: float = 1.0) -> np.ndarray:
    """Cuantizador uniforme 'mid-rise' de n_bits sobre el rango [-x_max, x_max]."""
    levels = 2 ** n_bits
    step = 2.0 * x_max / levels
    xq = step * (np.floor(x / step) + 0.5)
    return np.clip(xq, -x_max + step / 2, x_max - step / 2)


def sqnr_db(x: np.ndarray, xq: np.ndarray) -> float:
    """Relación señal a ruido de cuantización medida (dB)."""
    noise = x - xq
    return float(10 * np.log10(np.mean(x ** 2) / np.mean(noise ** 2)))


def sqnr_theory_db(n_bits: int) -> float:
    """SQNR teórica de una senoidal a plena escala: 6.02·n + 1.76 dB."""
    return 6.02 * n_bits + 1.76
