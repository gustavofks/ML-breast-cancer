"""Roda o fluxo de verdade em algumas pacientes, contra o modelo ajustado.

Verificacao da semana 5, nao avaliacao: a decisao do medico e simulada e
marcada como tal na auditoria. Grava results/fase3/fluxo/execucoes.json com o
estado final de cada execucao e os eventos de auditoria.

    FASE3_AJUSTADO_URL=http://host.docker.internal:8092 python scripts/rodar_fluxo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.api.assistente import ServicoAssistente  # noqa: E402
from src.api.servico import Servico, carregar_especificacao  # noqa: E402

MEDICO = "verificacao-automatica-semana5"


def _pacientes() -> list[str]:
    """Duas com PAAF maligna e duas benignas pela referencia, e uma so com pendencia."""
    gabarito = json.loads(config.CLINICA_GABARITO_FILE.read_text(encoding="utf-8"))
    malignas = [g["paciente"] for g in gabarito if g["referencia"] == "Maligno"][:2]
    benignas = [g["paciente"] for g in gabarito if g["referencia"] == "Benigno"][:2]
    com_paaf = {g["paciente"] for g in gabarito}
    registros = [json.loads(linha) for linha in
                 config.CLINICA_PRONTUARIOS_FILE.read_text(encoding="utf-8").splitlines()]
    sem_paaf = sorted({r["paciente"] for r in registros if r["tipo"] == "exame"
                       and r["resultado"] is None and r["paciente"] not in com_paaf})[:1]
    return malignas + benignas + sem_paaf


def main() -> None:
    modelo = Servico.construir(carregar_especificacao())
    s = ServicoAssistente.construir(modelo.prever)
    execucoes = []
    for paciente in _pacientes():
        estado = s.fluxo.iniciar(paciente)
        if estado["status"] == "aguardando_medico":
            estado = s.fluxo.decidir(estado["execucao"], {"acao": "aprovar", "medico": MEDICO})
        execucoes.append({**estado, "auditoria": s.auditoria.ler(estado["execucao"])})
        print(paciente, estado["status"], estado["verificacao"], estado["alertas"], flush=True)
    destino = config.FASE3_DIR / "fluxo" / "execucoes.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(execucoes, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
