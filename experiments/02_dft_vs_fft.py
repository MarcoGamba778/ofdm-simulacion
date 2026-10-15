"""Diapositivas 'Análisis Matemático' y 'Eficiencia Computacional': DFT/IDFT y O(N²) vs O(N log N)."""

import time

import numpy as np
from _common import ACCENT, GREY, OK, WARN, plt, save

from ofdm_sim.dft import dft, dft_naive, fft_radix2, idft

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Verificación numérica
x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
print("Verificación (N=256):")
print(f"  max|DFT propia  - np.fft.fft| = {np.max(np.abs(dft(x) - np.fft.fft(x))):.2e}")
print(f"  max|FFT radix-2 - np.fft.fft| = {np.max(np.abs(fft_radix2(x) - np.fft.fft(x))):.2e}")
print(f"  max|IDFT(DFT(x)) - x|         = {np.max(np.abs(idft(dft(x)) - x)):.2e}")


# ---------------------------------------------------------------- Medición de tiempos
def best_time(fn, arg, repeats=3):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(arg)
        best = min(best, time.perf_counter() - t0)
    return best


sizes = 2 ** np.arange(4, 13)            # 16 ... 4096
t_dft, t_fft_own, t_fft_np = [], [], []
print("\n   N   |  DFT O(N²) [ms] | FFT propia [ms] | np.fft [ms]")
for n in sizes:
    sig = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    a, b, c = best_time(dft_naive, sig), best_time(fft_radix2, sig), best_time(np.fft.fft, sig, 20)
    t_dft.append(a); t_fft_own.append(b); t_fft_np.append(c)
    print(f"{n:6d} | {a*1e3:15.3f} | {b*1e3:15.3f} | {c*1e3:10.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
ax.loglog(sizes, t_dft, "o-", color=WARN, label="DFT directa  O(N²)")
ax.loglog(sizes, t_fft_own, "s-", color=ACCENT, label="FFT radix-2 propia  O(N log N)")
ax.loglog(sizes, t_fft_np, "^-", color=OK, label="np.fft.fft")
ax.set_xlabel("N"); ax.set_ylabel("tiempo [s]"); ax.legend(); ax.set_title("Tiempo de cómputo medido")

ax = axes[1]
n_f = sizes.astype(float)
ops_dft, ops_fft = n_f ** 2, n_f * np.log2(n_f)
ax.loglog(sizes, ops_dft, color=WARN, label="N²")
ax.loglog(sizes, ops_fft, color=ACCENT, label="N·log₂N")
ax.set_xlabel("N"); ax.set_ylabel("operaciones (orden)"); ax.legend()
ax.set_title(f"Para N=4096 la FFT necesita ≈ {ops_dft[-1]/ops_fft[-1]:.0f}× menos operaciones")
save(fig, "02_dft_vs_fft.png")
