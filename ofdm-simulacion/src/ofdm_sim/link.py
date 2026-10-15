"""Enlace completo (los 4 pasos del transceptor) y utilidades de BER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import erfc

from .channel import awgn, frequency_response, multipath
from .ofdm import (OFDMParams, equalize, ofdm_demodulate, ofdm_modulate,
                   qam_demodulate, qam_modulate)


@dataclass
class LinkResult:
    n_errors: int
    n_bits: int
    tx_symbols: np.ndarray      # (n_ofdm, N)
    rx_symbols: np.ndarray      # (n_ofdm, N), ya ecualizados

    @property
    def ber(self) -> float:
        return self.n_errors / self.n_bits


def run_link(params: OFDMParams, h: np.ndarray, snr_db: float | None, rng: np.random.Generator,
             n_ofdm: int = 100, use_cp: bool = True) -> LinkResult:
    """Tx -> canal -> Rx. snr_db=None significa canal sin ruido."""
    n = params.n_fft
    # 1-2) Transmisor: bits -> QAM -> IFFT -> prefijo cíclico
    bits = rng.integers(0, 2, n_ofdm * params.bits_per_ofdm_symbol)
    tx_grid = qam_modulate(bits, params.mod_order).reshape(n_ofdm, n)
    tx = ofdm_modulate(tx_grid, params, add_cp=use_cp)
    # 3) Canal: multitrayecto + AWGN (SNR respecto a la potencia transmitida)
    rx = multipath(tx, h)
    if snr_db is not None:
        rx = awgn(rx, snr_db, rng, signal_power=np.mean(np.abs(tx) ** 2))
    # 4) Receptor: quitar CP -> FFT -> ecualización escalar -> demodulación
    rx_grid = equalize(ofdm_demodulate(rx, params, has_cp=use_cp), frequency_response(h, n))
    rx_bits = qam_demodulate(rx_grid.reshape(-1), params.mod_order)
    return LinkResult(int(np.sum(bits != rx_bits)), bits.size, tx_grid, rx_grid)


def ber_curve(params: OFDMParams, snr_db_values: np.ndarray, channel_fn: Callable[[np.random.Generator], np.ndarray],
              use_cp: bool, rng: np.random.Generator, trials: int = 30, n_ofdm: int = 100) -> np.ndarray:
    """BER promedio por cada SNR; un canal nuevo por prueba (desvanecimiento por bloques)."""
    out = np.empty(len(snr_db_values))
    for i, snr in enumerate(snr_db_values):
        errors = bits = 0
        for _ in range(trials):
            res = run_link(params, channel_fn(rng), snr, rng, n_ofdm=n_ofdm, use_cp=use_cp)
            errors += res.n_errors
            bits += res.n_bits
        out[i] = errors / bits
    return out


def qam_ber_awgn(mod_order: int, snr_db: np.ndarray) -> np.ndarray:
    """BER teórica aproximada de M-QAM (Gray) en AWGN, con SNR = Es/N0."""
    k = np.log2(mod_order)
    snr = 10 ** (np.asarray(snr_db) / 10.0)
    q = 0.5 * erfc(np.sqrt(3.0 * snr / (mod_order - 1)) / np.sqrt(2.0))
    return (4.0 / k) * (1.0 - 1.0 / np.sqrt(mod_order)) * q
