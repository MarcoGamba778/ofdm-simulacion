"""Demostración extremo a extremo: se transmite un texto por el canal OFDM y se recupera."""

import numpy as np
from _common import plt  # noqa: F401  (configura rutas del paquete)

from ofdm_sim.channel import awgn, example_channel, frequency_response, multipath
from ofdm_sim.ofdm import (OFDMParams, equalize, ofdm_demodulate, ofdm_modulate,
                           qam_demodulate, qam_modulate)

MESSAGE = "Hola OFDM: la FFT hace posible esta transmision."
SNR_DB = 30.0
params = OFDMParams(n_fft=64, cp_len=16, mod_order=16)
rng = np.random.default_rng(5)
h = example_channel()


def transmit(text: str, use_cp: bool) -> str:
    bits = np.unpackbits(np.frombuffer(text.encode("utf-8"), dtype=np.uint8)).astype(int)
    per = params.bits_per_ofdm_symbol
    padded = np.concatenate([bits, np.zeros((-len(bits)) % per, dtype=int)])
    grid = qam_modulate(padded, params.mod_order).reshape(-1, params.n_fft)
    tx = ofdm_modulate(grid, params, add_cp=use_cp)
    rx = awgn(multipath(tx, h), SNR_DB, rng, signal_power=np.mean(np.abs(tx) ** 2))
    eq = equalize(ofdm_demodulate(rx, params, has_cp=use_cp), frequency_response(h, params.n_fft))
    rx_bits = qam_demodulate(eq.reshape(-1), params.mod_order)[: len(bits)]
    return np.packbits(rx_bits.astype(np.uint8)).tobytes().decode("utf-8", errors="replace")


print(f"Mensaje original      : {MESSAGE}")
print(f"Recibido SIN prefijo  : {transmit(MESSAGE, False)}")
print(f"Recibido CON prefijo  : {transmit(MESSAGE, True)}")
