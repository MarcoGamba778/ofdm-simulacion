"""Ejecuta todos los experimentos en orden y deja las figuras en results/.

Uso:  python run_all.py            (completo)
      python run_all.py --quick    (más rápido, menos pruebas Monte Carlo)
"""

import runpy
import sys
from pathlib import Path

EXPERIMENTS = sorted(Path(__file__).parent.joinpath("experiments").glob("0*.py"))

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    for script in EXPERIMENTS:
        print(f"\n{'=' * 70}\n{script.name}\n{'=' * 70}")
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script)] + (["--quick"] if quick and script.name.startswith("05") else [])
        runpy.run_path(str(script), run_name="__main__")
    print("\nListo. Revisa la carpeta results/")
