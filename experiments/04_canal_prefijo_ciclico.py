"""Diapositivas 'Canales con Ruido' y 'Diseño del Transceptor': multitrayecto, AWGN y prefijo cíclico."""

import numpy as np
from _common import ACCENT, OK, WARN, plt, save

from ofdm_sim.channel import example_channel, frequency_response
from ofdm_sim.link import run_link
from ofdm_sim.ofdm import OFDMParams

SNR_DB = 28.0
params = OFDMParams(n_fft=64, cp_len=16, mod_order=16, bandwidth=1e6)
h = example_channel()

print("=== Paso 1: parámetros ===")
print(params.summary())

rng = np.random.default_rng(7)
with_cp = run_link(params, h, SNR_DB, rng, n_ofdm=200, use_cp=True)
no_cp = run_link(params, h, SNR_DB, rng, n_ofdm=200, use_cp=False)
print(f"\n=== Pasos 2-4 (SNR = {SNR_DB:.0f} dB, canal de {len(h)} taps) ===")
print(f"BER sin prefijo cíclico : {no_cp.ber:.3e}")
print(f"BER con prefijo cíclico : {with_cp.ber:.3e}")

H = frequency_response(h, params.n_fft)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
ax = axes[0, 0]
ax.stem(np.abs(h), basefmt=" ")
ax.set_title("Respuesta al impulso |h[n]| (multitrayecto)")
ax.set_xlabel("retardo [muestras]")
ax = axes[0, 1]
ax.plot(20 * np.log10(np.abs(H)), color=ACCENT)
ax.set_title("Desvanecimiento selectivo: |H[k]| por subportadora")
ax.set_xlabel("subportadora k"); ax.set_ylabel("dB")
for ax, res, title, color in [(axes[1, 0], no_cp, "Sin prefijo cíclico (ISI + ICI)", WARN),
                              (axes[1, 1], with_cp, "Con prefijo cíclico + ecualización escalar", OK)]:
    ax.plot(res.rx_symbols.real.ravel(), res.rx_symbols.imag.ravel(), ".", ms=2, alpha=0.5, color=color)
    ax.set_title(f"{title}\nBER = {res.ber:.2e}")
    ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal"); ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
fig.suptitle(f"Constelaciones 16-QAM recibidas, SNR = {SNR_DB:.0f} dB")
save(fig, "04_canal_prefijo_ciclico.png")
