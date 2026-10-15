"""DFT / IDFT implementadas a mano, y una FFT radix-2 propia para comparar costos."""

from __future__ import annotations

import numpy as np


def dft_matrix(n: int) -> np.ndarray:
    """Matriz de Fourier W[k, m] = exp(-j 2π k m / N)."""
    idx = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(idx, idx) / n)


def dft(x: np.ndarray) -> np.ndarray:
    """DFT: X[k] = Σ x[n] e^{-j 2π k n / N}   (producto matricial, O(N²))."""
    x = np.asarray(x, dtype=complex)
    return dft_matrix(len(x)) @ x


def idft(spectrum: np.ndarray) -> np.ndarray:
    """IDFT: x[n] = (1/N) Σ X[k] e^{+j 2π k n / N}   (O(N²))."""
    spectrum = np.asarray(spectrum, dtype=complex)
    n = len(spectrum)
    return np.conj(dft_matrix(n)) @ spectrum / n


def dft_naive(x: np.ndarray) -> np.ndarray:
    """DFT con un lazo explícito sobre k. Costo O(N²), usada para medir tiempos."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    m = np.arange(n)
    out = np.empty(n, dtype=complex)
    for k in range(n):
        out[k] = np.sum(x * np.exp(-2j * np.pi * k * m / n))
    return out


def fft_radix2(x: np.ndarray) -> np.ndarray:
    """FFT recursiva radix-2 (Cooley-Tukey). Costo O(N log2 N); N debe ser potencia de 2."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    if n & (n - 1):
        raise ValueError("fft_radix2 requiere N potencia de 2")
    if n == 1:
        return x
    even = fft_radix2(x[0::2])
    odd = fft_radix2(x[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + twiddle, even - twiddle])
