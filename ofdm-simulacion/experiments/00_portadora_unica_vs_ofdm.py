"""Diapositiva 'Portadora Única vs FDM': la ISI destruye la portadora única sin ecualizador;
OFDM la absorbe con prefijo cíclico + ecualización de un solo coeficiente por subportadora."""

import numpy as np
from _common import ACCENT, plt, save

from ofdm_sim.channel import awgn, example_channel, multipath
from ofdm_sim.link import run_link
from ofdm_sim.ofdm import OFDMParams, qam_modulate

SNR_DB = 30.0
rng = np.random.default_rng(1)
params = OFDMParams(n_fft=64, cp_len=16, mod_order=16)
h = example_channel()

# Portadora única: una sola portadora a la tasa completa B, sin ecualizador temporal
n_sym = 64 * 100
bits = rng.integers(0, 2, n_sym * params.bits_per_symbol)
sc_tx = qam_modulate(bits, params.mod_order)
sc_rx = awgn(multipath(sc_tx, h), SNR_DB, rng, signal_power=1.0)

# OFDM con CP y ecualización escalar sobre el mismo canal y la misma SNR
ofdm = run_link(params, h, SNR_DB, rng, n_ofdm=100, use_cp=True)

ts_single = params.sample_time
tau = (len(h) - 1) * params.sample_time
print(f"Retardo máximo del canal: {tau*1e6:.1f} µs")
print(f"Portadora única : duración de símbolo {ts_single*1e6:.2f} µs -> la ISI abarca {len(h)-1} símbolos")
print(f"OFDM (N={params.n_fft})   : duración de símbolo {params.useful_duration*1e6:.2f} µs -> "
      f"el retardo es {100*tau/params.useful_duration:.1f}% del símbolo (lo absorbe el CP)")
print(f"BER OFDM+CP a {SNR_DB:.0f} dB: {ofdm.ber:.2e}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
axes[0].plot(sc_rx.real, sc_rx.imag, ".", ms=2, color="#e0533d", alpha=0.5)
axes[0].set_title("Portadora única sin ecualizador (ISI)")
axes[1].plot(ofdm.rx_symbols.real.ravel(), ofdm.rx_symbols.imag.ravel(), ".", ms=2, color=ACCENT, alpha=0.5)
axes[1].set_title("OFDM + prefijo cíclico + ecualización escalar")
for ax in axes:
    ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal"); ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
fig.suptitle(f"16-QAM sobre el mismo canal multitrayecto, SNR = {SNR_DB:.0f} dB")
save(fig, "00_portadora_unica_vs_ofdm.png")
