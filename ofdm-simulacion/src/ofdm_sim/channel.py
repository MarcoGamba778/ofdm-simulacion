"""Canal: desvanecimiento multitrayecto + ruido AWGN."""

from __future__ import annotations

import numpy as np


def example_channel() -> np.ndarray:
    """Canal multitrayecto fijo (3 trayectos, retardos 0, 2 y 5 muestras), energía unitaria."""
    h = np.zeros(6, dtype=complex)
    h[0] = 0.85
    h[2] = 0.45 * np.exp(1j * 0.9)
    h[5] = 0.25 * np.exp(-1j * 1.7)
    return h / np.linalg.norm(h)


def rayleigh_channel(n_taps: int, rng: np.random.Generator, decay: float = 2.0) -> np.ndarray:
    """Canal Rayleigh con perfil de potencia exponencial decreciente y energía media unitaria."""
    pdp = np.exp(-np.arange(n_taps) / decay)
    pdp /= pdp.sum()
    taps = rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)
    return taps * np.sqrt(pdp / 2.0)


def awgn_channel() -> np.ndarray:
    """Canal ideal (un solo trayecto): solo hay ruido."""
    return np.array([1.0 + 0j])


def multipath(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolución lineal con la respuesta al impulso h, truncada a la longitud de x."""
    return np.convolve(x, h)[: len(x)]


def awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator, signal_power: float | None = None) -> np.ndarray:
    """Suma ruido blanco gaussiano complejo para una SNR por muestra dada (dB)."""
    p = np.mean(np.abs(x) ** 2) if signal_power is None else signal_power
    sigma2 = p / (10 ** (snr_db / 10.0))
    noise = np.sqrt(sigma2 / 2.0) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x + noise


def frequency_response(h: np.ndarray, n_fft: int) -> np.ndarray:
    """H[k] = FFT(h) en las N subportadoras."""
    return np.fft.fft(h, n_fft)
