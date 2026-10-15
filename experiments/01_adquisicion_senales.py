"""Diapositiva 'Adquisición de Señales': Nyquist/aliasing, cuantización y resolución Δf = fs/N."""

import numpy as np
from _common import ACCENT, GREY, OK, WARN, plt, save

from ofdm_sim.sampling import (alias_frequency, num_samples, quantize, spectral_resolution,
                               sqnr_db, sqnr_theory_db)

# ---------------------------------------------------------------- Teorema de Nyquist
f0, T = 40.0, 1.0                      # tono de 40 Hz -> B = 40 Hz, Nyquist exige fs >= 80 Hz
t_dense = np.linspace(0, 0.15, 3000)
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
for col, (fs, color) in enumerate([(200.0, OK), (60.0, WARN)]):
    n = num_samples(T, fs)
    ts = np.arange(n) / fs
    x = np.sin(2 * np.pi * f0 * ts)
    fa = alias_frequency(f0, fs)
    ok = fs >= 2 * f0
    ax = axes[0, col]
    ax.plot(t_dense, np.sin(2 * np.pi * f0 * t_dense), color=GREY, lw=1, label=f"señal real ({f0:.0f} Hz)")
    m = ts <= 0.15
    ax.plot(ts[m], x[m], "o", color=color, label=f"muestras (fs={fs:.0f} Hz)")
    if not ok:
        ax.plot(t_dense, -np.sin(2 * np.pi * fa * t_dense), "--", color=color, label=f"alias aparente ({fa:.0f} Hz)")
    ax.set_title(f"fs = {fs:.0f} Hz  ({'cumple' if ok else 'NO cumple'} fs ≥ 2B)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("amplitud"); ax.legend(loc="upper right")
    spec = np.abs(np.fft.rfft(x)) / (n / 2)
    ax = axes[1, col]
    ax.plot(np.fft.rfftfreq(n, 1 / fs), spec, color=color)
    ax.axvline(fa, ls=":", color="k", lw=0.8)
    ax.set_title(f"Espectro: pico en {fa:.0f} Hz" + ("" if ok else f" (¡debería estar en {f0:.0f} Hz!)"))
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("|X|")
fig.suptitle("Teorema de Nyquist y aliasing")
save(fig, "01a_nyquist_aliasing.png")

# ---------------------------------------------------------------- Cuantización
fs = 48000.0
t = np.arange(int(fs)) / fs
x = np.sin(2 * np.pi * 137.0 * t)
bits_range = np.arange(2, 13)
measured = [sqnr_db(x, quantize(x, b)) for b in bits_range]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
tt = t[:400]
axes[0].plot(tt * 1e3, x[:400], color=GREY, label="continua")
axes[0].step(tt * 1e3, quantize(x, 3)[:400], where="mid", color=ACCENT, label="cuantizada (3 bits, 8 niveles)")
axes[0].set_xlabel("t [ms]"); axes[0].set_ylabel("amplitud"); axes[0].legend(); axes[0].set_title("Amplitudes continuas → niveles discretos")
axes[1].plot(bits_range, measured, "o-", color=ACCENT, label="medida")
axes[1].plot(bits_range, [sqnr_theory_db(b) for b in bits_range], "--", color=WARN, label="6.02·n + 1.76 dB")
axes[1].set_xlabel("bits por muestra"); axes[1].set_ylabel("SQNR [dB]"); axes[1].legend()
axes[1].set_title("Cada bit adicional aporta ≈ 6 dB")
save(fig, "01b_cuantizacion.png")

# ---------------------------------------------------------------- Resolución espectral
fs = 1000.0
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, T_obs in zip(axes, (0.2, 1.0)):
    n = num_samples(T_obs, fs)
    tt = np.arange(n) / fs
    x = np.sin(2 * np.pi * 100 * tt) + np.sin(2 * np.pi * 103 * tt)
    df = spectral_resolution(fs, n)
    ax.plot(np.fft.rfftfreq(n, 1 / fs), np.abs(np.fft.rfft(x)) / n * 2, color=ACCENT)
    ax.set_xlim(85, 120)
    ax.set_title(f"N = T·fs = {n}  →  Δf = fs/N = {df:.0f} Hz  ({'resueltos' if df < 3 else 'NO resueltos'})")
    ax.set_xlabel("f [Hz]")
axes[0].set_ylabel("|X|")
fig.suptitle("Dos tonos a 100 y 103 Hz")
save(fig, "01c_resolucion_espectral.png")
