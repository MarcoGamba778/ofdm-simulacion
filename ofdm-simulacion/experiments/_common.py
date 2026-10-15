"""Configuración compartida por los experimentos (rutas y estilo de las figuras)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ACCENT = "#0b9be3"
WARN = "#e0533d"
OK = "#2a9d63"
GREY = "#6b7280"

plt.rcParams.update({
    "figure.constrained_layout.use": True,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
})


def save(fig, name: str) -> None:
    path = RESULTS / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> figura guardada: results/{name}")
