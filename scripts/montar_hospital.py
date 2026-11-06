"""Monta o hospital 3D em `build/space/hospital.html`.

Uso, a partir da raiz do projeto:

    python scripts/montar_hospital.py

Le `results/` e a vitrine ja montada em `build/space/index.html`. Usa so a
biblioteca padrao, entao roda na maquina, sem o container.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.ui import hospital  # noqa: E402


def main() -> None:
    dados = hospital.montar_dados(hospital.casos_da_vitrine())
    html = hospital.pagina(dados, hospital.embutir_figuras(dados))
    config.HOSPITAL_PAGINA.parent.mkdir(parents=True, exist_ok=True)
    config.HOSPITAL_PAGINA.write_text(html, encoding="utf-8")
    tamanho = config.HOSPITAL_PAGINA.stat().st_size / 1e6
    print(f"{config.HOSPITAL_PAGINA} ({tamanho:.1f} MB)")


if __name__ == "__main__":
    main()
