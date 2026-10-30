"""Presenca deste arquivo faz o pytest incluir a raiz do projeto no sys.path,
permitindo `from src import ...` nos testes sem instalar o pacote.

Guarda tambem as fixtures compartilhadas entre arquivos de teste. Importar de um
arquivo de teste para outro exigiria transformar `tests/` em pacote, com
`__init__.py`, e o projeto nao faz isso.
"""

import pytest

from src.genetic.chromosome import PADROES


@pytest.fixture
def payload_ag():
    """Fabrica de payload no formato de `src.genetic.engine.para_dicionario`.

    Os testes de agregacao e de figuras precisam de execucoes sinteticas com
    valores controlados: afirmar qual media a agregacao devolve exige saber qual
    media esperar, e isso uma execucao real do algoritmo genetico nao da.
    """

    def _montar(
        *,
        experimento="E1 base",
        familia="Regressão Logística",
        semente=42,
        braco="AG, completo",
        fitness=0.95,
        fn=2,
        fp=16,
        n=455,
        features=("texture_worst", "area_worst"),
        segmentos=("hiperparametros", "limiar", "mascara"),
    ):
        return {
            "experimento": experimento,
            "familia": familia,
            "semente": semente,
            "braco": braco,
            "segmentos": sorted(segmentos),
            "historico": [
                {
                    "geracao": 0,
                    "melhor_fitness": fitness - 0.01,
                    "fitness_medio": fitness - 0.03,
                    "desvio_fitness": 0.01,
                    "diversidade": 0.4,
                },
                {
                    "geracao": 1,
                    "melhor_fitness": fitness,
                    "fitness_medio": fitness - 0.01,
                    "desvio_fitness": 0.005,
                    "diversidade": 0.3,
                },
            ],
            "melhor": {
                "familia": familia,
                "n_features": len(features),
                "features": list(features),
                "mascara": [1] * len(features) + [0] * (30 - len(features)),
                # Hiperparametros coerentes com a familia: um payload de KNN com
                # o `C` da regressao logistica passaria nos testes de agregacao,
                # que so leem numeros, e quebraria no primeiro teste que de fato
                # construisse o modelo.
                "hiperparametros": dict(PADROES[familia]),
                "limiar": 0.25,
            },
            "avaliacao": {
                "fitness": fitness,
                "custo": (10 * fn + fp) / n,
                "recall": 0.98,
                "falsos_negativos": fn,
                "falsos_positivos": fp,
                "n_amostras": n,
                "n_features": len(features),
            },
        }

    return _montar
