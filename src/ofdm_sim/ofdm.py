"""Bloques OFDM: parámetros, modulación QAM (Gray), IFFT/FFT, prefijo cíclico, ecualización."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# Paso 1 - Parámetros del sistema
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OFDMParams:
    """Parámetros OFDM. Con muestreo crítico fs = B, el tiempo de muestra es Ts = 1/B."""

    n_fft: int = 64          # tamaño de la FFT = número de subportadoras
    cp_len: int = 16         # longitud del prefijo cíclico (muestras)
    mod_order: int = 16      # M-QAM (4, 16, 64, 256)
    bandwidth: float = 1e6   # ancho de banda B [Hz]

    def __post_init__(self) -> None:
        if self.n_fft < 2 or self.cp_len < 0:
            raise ValueError("n_fft >= 2 y cp_len >= 0")
        k = int(np.log2(self.mod_order))
        if 2 ** k != self.mod_order or k % 2 != 0:
            raise ValueError("mod_order debe ser 4, 16, 64, 256, ...")

    @property
    def fs(self) -> float:
        return self.bandwidth

    @property
    def sample_time(self) -> float:
        return 1.0 / self.fs

    @property
    def subcarrier_spacing(self) -> float:
        """Δf = fs / N = 1 / T."""
        return self.fs / self.n_fft

    @property
    def useful_duration(self) -> float:
        """T = N · Ts."""
        return self.n_fft * self.sample_time

    @property
    def cp_duration(self) -> float:
        return self.cp_len * self.sample_time

    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.mod_order))

    @property
    def bits_per_ofdm_symbol(self) -> int:
        return self.n_fft * self.bits_per_symbol

    @property
    def cp_efficiency(self) -> float:
        return self.n_fft / (self.n_fft + self.cp_len)

    @property
    def bit_rate(self) -> float:
        """Tasa binaria útil [bit/s]."""
        return self.bits_per_ofdm_symbol * self.cp_efficiency * self.fs / self.n_fft

    def summary(self) -> str:
        return (
            f"N={self.n_fft} | CP={self.cp_len} muestras | {self.mod_order}-QAM | B={self.bandwidth/1e6:.2f} MHz\n"
            f"Ts={self.sample_time*1e6:.3f} µs | Δf={self.subcarrier_spacing/1e3:.3f} kHz | "
            f"T={self.useful_duration*1e6:.1f} µs | T_cp={self.cp_duration*1e6:.1f} µs\n"
            f"Eficiencia por CP={self.cp_efficiency*100:.1f}% | Tasa útil={self.bit_rate/1e6:.3f} Mbit/s"
        )


# ----------------------------------------------------------------------------
# Paso 2 - Modulación QAM cuadrada con mapeo Gray
# ----------------------------------------------------------------------------
def _axis_tables(levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Tablas Gray por eje: gray_code -> nivel PAM, e índice binario -> gray_code."""
    binary = np.arange(levels)
    gray = binary ^ (binary >> 1)
    gray_to_level = np.empty(levels)
    gray_to_level[gray] = 2 * binary - (levels - 1)
    return gray_to_level, gray


def _norm(mod_order: int) -> float:
    """Factor para potencia media unitaria en la constelación."""
    return float(np.sqrt(2.0 * (mod_order - 1) / 3.0))


def qam_modulate(bits: np.ndarray, mod_order: int) -> np.ndarray:
    """Bits (0/1) -> símbolos M-QAM con potencia media 1."""
    k = int(np.log2(mod_order))
    half = k // 2
    bits = np.asarray(bits, dtype=np.int64)
    if bits.size % k:
        raise ValueError(f"El número de bits debe ser múltiplo de {k}")
    groups = bits.reshape(-1, k)
    weights = 1 << np.arange(half - 1, -1, -1)
    gray_i = groups[:, :half] @ weights
    gray_q = groups[:, half:] @ weights
    table, _ = _axis_tables(2 ** half)
    return (table[gray_i] + 1j * table[gray_q]) / _norm(mod_order)


def qam_demodulate(symbols: np.ndarray, mod_order: int) -> np.ndarray:
    """Símbolos recibidos -> bits (decisión dura por eje)."""
    k = int(np.log2(mod_order))
    half = k // 2
    levels = 2 ** half
    z = np.asarray(symbols) * _norm(mod_order)
    _, bin_to_gray = _axis_tables(levels)

    def axis_bits(v: np.ndarray) -> np.ndarray:
        idx = np.clip(np.round((v + levels - 1) / 2.0), 0, levels - 1).astype(np.int64)
        g = bin_to_gray[idx]
        return (g[:, None] >> np.arange(half - 1, -1, -1)) & 1

    return np.concatenate([axis_bits(z.real), axis_bits(z.imag)], axis=1).reshape(-1)


# ----------------------------------------------------------------------------
# Paso 2 y 4 - IFFT + prefijo cíclico (Tx) y quitar prefijo + FFT (Rx)
# ----------------------------------------------------------------------------
def ofdm_modulate(grid: np.ndarray, params: OFDMParams, add_cp: bool = True) -> np.ndarray:
    """grid (n_ofdm, N) de símbolos en frecuencia -> señal serie en el tiempo.

    IFFT con normalización unitaria (norm='ortho') para conservar la potencia.
    El prefijo cíclico copia la cola de cada símbolo al inicio."""
    x = np.fft.ifft(grid, axis=1, norm="ortho")
    if add_cp and params.cp_len > 0:
        x = np.concatenate([x[:, -params.cp_len:], x], axis=1)
    return x.reshape(-1)


def ofdm_demodulate(rx: np.ndarray, params: OFDMParams, has_cp: bool = True) -> np.ndarray:
    """Señal recibida -> grid (n_ofdm, N) en frecuencia: descarta CP y aplica FFT."""
    block = params.n_fft + (params.cp_len if has_cp else 0)
    n_blocks = len(rx) // block
    blocks = rx[: n_blocks * block].reshape(n_blocks, block)
    if has_cp:
        blocks = blocks[:, params.cp_len:]
    return np.fft.fft(blocks, axis=1, norm="ortho")


def equalize(received_grid: np.ndarray, channel_response: np.ndarray) -> np.ndarray:
    """Ecualización escalar (zero-forcing): una división por subportadora, Y[k]/H[k]."""
    return received_grid / channel_response[None, :]
