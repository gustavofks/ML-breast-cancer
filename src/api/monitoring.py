"""Sentinela de desvio e contadores operacionais no formato do Prometheus.

A Fase 1 nomeou a degradacao silenciosa como risco de implantacao: troca de
equipamento, protocolo ou populacao desloca a distribuicao das medidas sem
gerar erro visivel. A sentinela implementa a mitigacao que aquele relatorio
prescreveu: compara cada medida recebida com a distribuicao do treino e registra
alerta quando ela foge por mais de `k` desvios padrao, para cima ou para baixo.

A sentinela registra, nao bloqueia. Quem decide se confia no resultado e o
medico; a API so garante que ele e o operador fiquem sabendo.

O formato Prometheus e escrito a mao. Tres metricas nao justificam uma
dependencia com registro global, que alem disso vazaria estado entre testes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Alerta:
    feature: str
    valor: float
    desvios: float


@dataclass(frozen=True)
class Sentinela:
    medias: dict[str, float]
    desvios_padrao: dict[str, float]
    k: float

    @classmethod
    def de_treino(cls, X_train: pd.DataFrame, k: float) -> "Sentinela":
        return cls(
            medias={c: float(X_train[c].mean()) for c in X_train.columns},
            # Desvio populacional: e a distribuicao do treino, nao uma estimativa dela.
            desvios_padrao={c: float(X_train[c].std(ddof=0)) for c in X_train.columns},
            k=k,
        )

    def verificar(self, medidas: dict[str, float]) -> list[Alerta]:
        """Medidas que fogem da distribuicao do treino, na ordem recebida."""
        alertas = []
        for feature, valor in medidas.items():
            desvio_padrao = self.desvios_padrao.get(feature)
            if not desvio_padrao:
                continue
            desvios = abs(float(valor) - self.medias[feature]) / desvio_padrao
            if desvios > self.k:
                alertas.append(Alerta(feature, float(valor), round(desvios, 2)))
        return alertas


class Monitor:
    """Contadores em memoria. Nada persiste: a API e sem estado por desenho."""

    def __init__(self) -> None:
        self._requisicoes: Counter = Counter()
        self._latencia_soma: defaultdict[str, float] = defaultdict(float)
        self._latencia_contagem: Counter = Counter()
        self._alertas: Counter = Counter()

    def registrar(self, rota: str, status: int, segundos: float) -> None:
        self._requisicoes[(rota, status)] += 1
        self._latencia_soma[rota] += segundos
        self._latencia_contagem[rota] += 1

    def registrar_alertas(self, alertas: list[Alerta]) -> None:
        for alerta in alertas:
            self._alertas[alerta.feature] += 1

    def exportar(self) -> str:
        """Texto no formato de exposicao do Prometheus, pronto para o coletor."""
        linhas = [
            "# HELP api_requisicoes_total Requisicoes atendidas, por rota e status.",
            "# TYPE api_requisicoes_total counter",
        ]
        for (rota, status), n in sorted(self._requisicoes.items()):
            linhas.append(f'api_requisicoes_total{{rota="{rota}",status="{status}"}} {n}')

        linhas += [
            "# HELP api_latencia_segundos Latencia por rota.",
            "# TYPE api_latencia_segundos summary",
        ]
        for rota in sorted(self._latencia_contagem):
            soma = round(self._latencia_soma[rota], 6)
            linhas.append(f'api_latencia_segundos_sum{{rota="{rota}"}} {soma}')
            linhas.append(
                f'api_latencia_segundos_count{{rota="{rota}"}} {self._latencia_contagem[rota]}'
            )

        linhas += [
            "# HELP api_alertas_de_desvio_total Medidas fora da distribuicao do treino.",
            "# TYPE api_alertas_de_desvio_total counter",
        ]
        for feature, n in sorted(self._alertas.items()):
            linhas.append(f'api_alertas_de_desvio_total{{feature="{feature}"}} {n}')

        return "\n".join(linhas) + "\n"
