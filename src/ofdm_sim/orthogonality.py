"""Ortogonalidad de subportadoras: producto interno, ICI y síntesis por IFFT."""

from __future__ import annotations

import numpy as np


def subcarriers(n_sc: int, delta_f: float, duration: float, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Genera n_sc exponenciales complejas e^{j 2π k Δf t} en [0, T). Devuelve (t, matriz)."""
    t = np.arange(n_points) * duration / n_points
    k = np.arange(n_sc).reshape(-1, 1)
    return t, np.exp(2j * np.pi * k * delta_f * t)


def gram_matrix(carriers: np.ndarray) -> np.ndarray:
    """G[k,l] = (1/T) ∫ φ_k(t) φ_l*(t) dt. Es la identidad si y solo si son ortogonales."""
    return carriers @ carriers.conj().T / carriers.shape[1]


def ofdm_spectrum(freqs: np.ndarray, k: int, delta_f: float, duration: float) -> np.ndarray:
    """Espectro (sinc) de la subportadora k con ventana rectangular de duración T."""
    return np.sinc((freqs - k * delta_f) * duration)


def power_leakage(n_fft: int, epsilon: float, k0: int = 5) -> float:
    """Fracción de potencia que sale de su bin cuando la subportadora tiene un
    desplazamiento de frecuencia ε·Δf (ε en fracción del espaciamiento). Es la causa de la ICI."""
    n = np.arange(n_fft)
    x = np.exp(2j * np.pi * (k0 + epsilon) * n / n_fft)
    spectrum = np.fft.fft(x)
    p_total = np.sum(np.abs(spectrum) ** 2)
    return float(1.0 - np.abs(spectrum[k0]) ** 2 / p_total)


def oscillator_bank_synthesis(spectrum: np.ndarray) -> np.ndarray:
    """Síntesis 'a la antigua': suma de N osciladores. Resultado idéntico a la IFFT."""
    spectrum = np.asarray(spectrum, dtype=complex)
    n = len(spectrum)
    m = np.arange(n)
    out = np.zeros(n, dtype=complex)
    for k, xk in enumerate(spectrum):
        out += xk * np.exp(2j * np.pi * k * m / n)
    return out / n
