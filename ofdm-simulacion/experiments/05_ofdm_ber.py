"""Diapositiva 'Diseño del Transceptor': curva BER vs SNR de la cadena OFDM completa."""

import argparse

import numpy as np
from _common import ACCENT, GREY, OK, WARN, plt, save

from ofdm_sim.channel import awgn_channel, rayleigh_channel
from ofdm_sim.link import ber_curve, qam_ber_awgn
from ofdm_sim.ofdm import OFDMParams

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mod", type=int, default=16, help="orden de la QAM (4, 16, 64)")
parser.add_argument("--trials", type=int, default=40, help="realizaciones de canal por punto de SNR")
parser.add_argument("--n-ofdm", type=int, default=100, help="símbolos OFDM por realización")
parser.add_argument("--quick", action="store_true", help="versión rápida (pocas pruebas)")
args = parser.parse_args()
trials = 6 if args.quick else args.trials

params = OFDMParams(n_fft=64, cp_len=16, mod_order=args.mod)
snr = np.arange(0, 36, 3.0)
rng = np.random.default_rng(2024)
fading = lambda r: rayleigh_channel(8, r)     # 8 taps <= CP + 1

print(f"Simulando {args.mod}-QAM, {trials} canales x {args.n_ofdm} símbolos OFDM por punto...")
ber_awgn = ber_curve(params, snr, lambda r: awgn_channel(), True, rng, trials, args.n_ofdm)
ber_cp = ber_curve(params, snr, fading, True, rng, trials, args.n_ofdm)
ber_nocp = ber_curve(params, snr, fading, False, rng, trials, args.n_ofdm)

print(" SNR[dB] | teoría AWGN | OFDM/AWGN | multitrayecto+CP | multitrayecto sin CP")
for s, a, b, c, d in zip(snr, qam_ber_awgn(args.mod, snr), ber_awgn, ber_cp, ber_nocp):
    print(f"{s:8.0f} | {a:11.2e} | {b:9.2e} | {c:16.2e} | {d:.2e}")

floor = 1e-6
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.semilogy(snr, np.maximum(qam_ber_awgn(args.mod, snr), floor), "k--", label="Teoría QAM en AWGN")
ax.semilogy(snr, np.maximum(ber_awgn, floor), "o", color=GREY, label="OFDM simulado, canal AWGN")
ax.semilogy(snr, np.maximum(ber_cp, floor), "s-", color=OK, label="Multitrayecto + prefijo cíclico + ecualizador")
ax.semilogy(snr, np.maximum(ber_nocp, floor), "^-", color=WARN, label="Multitrayecto SIN prefijo cíclico")
ax.set_ylim(floor, 1); ax.set_xlabel("SNR [dB]"); ax.set_ylabel("BER")
ax.set_title(f"OFDM {args.mod}-QAM, N={params.n_fft}, CP={params.cp_len}"); ax.legend(loc="lower left")
save(fig, f"05_ofdm_ber_{args.mod}qam.png")
