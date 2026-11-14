"""Monta a clinica ficticia: gera, anonimiza, grava e constroi o banco.

Saidas, em `data/clinica/` por padrao:
- `prontuarios.jsonl`: registros anonimizados, versionados no Git;
- `gabarito_paaf.json`: caso da Fase 2, pseudonimo e diagnostico de referencia,
  usado so na avaliacao (o prontuario nao traz a resposta);
- `prontuarios.sqlite`: banco de consultas, reconstruido a cada execucao;
- `bruto/`: dados com identificadores falsos e o mapa de pseudonimos, fora do Git.

    python scripts/gerar_clinica.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.clinica import anonimizacao, banco, gerador, paaf  # noqa: E402


def _serializar(objeto):
    if isinstance(objeto, date):
        return objeto.isoformat()
    raise TypeError(type(objeto))


def montar(*, destino: Path = config.CLINICA_DIR,
           n_pacientes: int = config.CLINICA_N_PACIENTES) -> dict:
    destino = Path(destino)
    (destino / "bruto").mkdir(parents=True, exist_ok=True)

    casos = paaf.casos_de_teste()
    clinica = gerador.gerar(n_pacientes=n_pacientes, casos_paaf=casos, semente=config.SEED)
    resultado = anonimizacao.anonimizar(clinica)

    (destino / "bruto" / "clinica.json").write_text(
        json.dumps(dataclasses.asdict(clinica), default=_serializar, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (destino / "bruto" / "pseudonimos.json").write_text(
        json.dumps(resultado.mapa, indent=1), encoding="utf-8"
    )
    with open(destino / "prontuarios.jsonl", "w", encoding="utf-8", newline="\n") as saida:
        for registro in resultado.registros:
            saida.write(json.dumps(registro, ensure_ascii=False, sort_keys=True) + "\n")

    referencias = {c.caso: c.referencia for c in casos}
    gabarito = sorted(
        ({"caso": e.caso_paaf, "paciente": resultado.mapa[e.paciente_id],
          "referencia": referencias[e.caso_paaf]}
         for e in clinica.exames if e.caso_paaf is not None),
        key=lambda g: g["caso"],
    )
    (destino / "gabarito_paaf.json").write_text(
        json.dumps(gabarito, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    banco.criar(destino / "prontuarios.sqlite", resultado.registros)
    return {"pacientes": len(clinica.pacientes), "exames": len(clinica.exames),
            "paaf": len(gabarito), "registros": len(resultado.registros)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--destino", type=Path, default=config.CLINICA_DIR)
    args = parser.parse_args()
    print(json.dumps(montar(destino=args.destino), ensure_ascii=False))


if __name__ == "__main__":
    main()
