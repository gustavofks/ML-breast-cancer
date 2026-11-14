"""Os casos de teste da Fase 2, reaproveitados como PAAF da clinica ficticia.

Uso ilustrativo, depois da avaliacao final: nenhuma decisao de modelo e tomada
sobre estes casos. A numeracao e a posicao no conjunto de teste, a mesma da
vitrine e do relatorio da Fase 2, para que "caso 16" signifique o mesmo
paciente nas tres fases.
"""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.data import load_dataset
from src.preprocessing import split_data


@dataclass(frozen=True)
class CasoPaaf:
    caso: int
    medidas: dict[str, float]
    referencia: str


def casos_de_teste() -> list[CasoPaaf]:
    X, y = load_dataset()
    _, X_teste, _, y_teste = split_data(X, y)
    return [
        CasoPaaf(
            caso=indice,
            medidas={nome: float(valor) for nome, valor in linha.items()},
            referencia=config.CLASS_NAMES[int(rotulo)],
        )
        for indice, (linha, rotulo) in enumerate(
            zip(X_teste.to_dict("records"), y_teste)
        )
    ]
