"""Diapositiva 'Ortogonalidad en OFDM': Δf = 1/T, ICI y síntesis con IFFT en vez de osciladores."""

import numpy as np
from _common import ACCENT, GREY, OK, WARN, plt, save

from ofdm_sim.orthogonality import (gram_matrix, ofdm_spectrum, oscillator_bank_synthesis,
                                    power_leakage, subcarriers)

T = 1.0
n_pts = 1024

# ---------------------------------------------------------------- Figura A: subportadoras y espectros
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
t, car = subcarriers(3, 1 / T, T, n_pts)
for k, c in enumerate(car):
    axes[0, 0].plot(t, c.real, label=f"k={k}")
axes[0, 0].set_title("Subportadoras con Δf = 1/T: ciclos enteros en T")
axes[0, 0].set_xlabel("t / T"); axes[0, 0].legend(loc="upper right")

f = np.linspace(-1.5, 6.5, 2000) / T
for k in range(6):
    axes[0, 1].plot(f, ofdm_spectrum(f, k, 1 / T, T), lw=1.2)
axes[0, 1].plot(np.arange(6) / T, np.ones(6), "ko", ms=4)
axes[0, 1].plot(np.arange(6) / T, np.zeros(6), "kx", ms=5)
axes[0, 1].set_title("OFDM: espectros sinc solapados; cada pico cae en los ceros de los demás")
axes[0, 1].set_xlabel("f · T")

# FDM clásico con bandas de guarda vs OFDM
g = 0.5
for k in range(6):
    axes[1, 0].plot(f, ofdm_spectrum(f, k, (1 + g) / T, T), color=WARN, lw=1.2)
axes[1, 0].set_title(f"FDM clásico con bandas de guarda ({int(g*100)}%): ocupa {(1+g)*6:.0f}/T")
axes[1, 0].set_xlabel("f · T"); axes[1, 0].set_xlim(-1.5, 8.5)
for k in range(6):
    axes[1, 1].plot(f, ofdm_spectrum(f, k, 1 / T, T), color=OK, lw=1.2)
axes[1, 1].set_title("OFDM: ocupa 6/T (≈ 33% menos ancho de banda para 6 subportadoras)")
axes[1, 1].set_xlabel("f · T"); axes[1, 1].set_xlim(-1.5, 8.5)
fig.suptitle("Solapamiento ortogonal vs multiplexación en frecuencia convencional")
save(fig, "03a_ortogonalidad_espectros.png")

# ---------------------------------------------------------------- Figura B: Gram + ICI
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, ratio, title in [(axes[0], 1.0, "Δf = 1/T  →  G = I (ortogonales)"),
                         (axes[1], 0.7, "Δf = 0.7/T  →  G ≠ I (interferencia)")]:
    _, car = subcarriers(8, ratio / T, T, n_pts)
    G = np.abs(gram_matrix(car))
    im = ax.imshow(G, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10); ax.set_xlabel("l"); ax.set_ylabel("k"); ax.grid(False)
    print(f"Δf = {ratio}/T: máx |G[k,l]| fuera de la diagonal = {np.max(G - np.diag(np.diag(G))):.3e}")
fig.colorbar(im, ax=axes[:2], shrink=0.8, label="|⟨φk, φl⟩|")

eps = np.linspace(0, 1, 101)
leak = [power_leakage(64, e) for e in eps]
axes[2].plot(eps, leak, color=ACCENT, label="simulación (N=64)")
axes[2].plot(eps, 1 - np.sinc(eps) ** 2, "--", color=WARN, label="1 − sinc²(ε)")
axes[2].set_xlabel("desajuste de frecuencia ε (fracción de Δf)")
axes[2].set_ylabel("potencia fugada a otras subportadoras (ICI)")
axes[2].set_title("Pérdida de ortogonalidad por desajuste"); axes[2].legend()
save(fig, "03b_gram_ici.png")

# ---------------------------------------------------------------- IFFT = banco de osciladores
rng = np.random.default_rng(3)
X = rng.standard_normal(64) + 1j * rng.standard_normal(64)
err = np.max(np.abs(np.fft.ifft(X) - oscillator_bank_synthesis(X)))
print(f"\nIFFT vs banco de 64 osciladores: error máximo = {err:.2e}  (idénticos)")
