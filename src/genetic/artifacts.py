"""Agregacao dos resultados das buscas nos artefatos que o relatorio consome.

O modulo e puro: recebe os dicionarios que `engine.para_dicionario` produziu e
devolve tabelas. Nao treina, nao le disco e nao conhece o conjunto de teste.

A decisao central aqui e **separar o custo clinico do termo de parcimonia**.
Comparar bracos por fitness mistura duas perguntas diferentes: quantos casos o
modelo erra, e quantas medidas ele exige do patologista. O piloto mostrou por que
isso importa — o braco completo vence no fitness com custo clinico pior que o
braco que so otimiza o limiar.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev
from typing import Any

from src import config


def custo_clinico(payload: dict[str, Any]) -> float:
    """Custo por amostra, sem o desconto de parcimonia.

    Recalculado a partir de falsos negativos e positivos em vez de lido do
    fitness: e o unico numero comparavel entre bracos que selecionam quantidades
    diferentes de features.
    """
    avaliacao = payload["avaliacao"]
    total = (
        config.CUSTO_FALSO_NEGATIVO * avaliacao["falsos_negativos"]
        + config.CUSTO_FALSO_POSITIVO * avaliacao["falsos_positivos"]
    )
    return float(total / avaliacao["n_amostras"])


def _desvio(valores: list[float]) -> float:
    """Desvio populacional, que vale zero com uma observacao so."""
    return float(pstdev(valores)) if len(valores) > 1 else 0.0


def agregar_por_experimento(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Uma linha por (experimento, familia), com media e desvio entre sementes.

    Algoritmo genetico e estocastico, e resultado de execucao unica nao e
    resultado. A tabela do relatorio e esta.
    """
    grupos: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for payload in payloads:
        grupos.setdefault((payload["experimento"], payload["familia"]), []).append(payload)

    linhas = []
    for (experimento, familia), execucoes in grupos.items():
        fitnesses = [p["avaliacao"]["fitness"] for p in execucoes]
        custos = [custo_clinico(p) for p in execucoes]
        features = [p["avaliacao"]["n_features"] for p in execucoes]

        linhas.append(
            {
                "experimento": experimento,
                "familia": familia,
                "sementes": len(execucoes),
                "fitness_medio": round(mean(fitnesses), 6),
                "fitness_desvio": round(_desvio(fitnesses), 6),
                "custo_medio": round(mean(custos), 6),
                "custo_desvio": round(_desvio(custos), 6),
                "features_medio": round(mean(features), 2),
                "geracoes": max(len(p["historico"]) for p in execucoes),
            }
        )

    return sorted(linhas, key=lambda linha: (linha["experimento"], linha["familia"]))


def tabela_de_bracos(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Uma linha por braco, ordenada pelo menor custo clinico.

    A ordenacao e por custo, e nao por fitness, porque a pergunta da secao 4.6 e
    qual braco erra menos. O numero de features aparece ao lado para que o leitor
    veja o que cada braco cobra do patologista.
    """
    grupos: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        grupos.setdefault(payload.get("braco", "AG, completo"), []).append(payload)

    linhas = []
    for braco, execucoes in grupos.items():
        custos = [custo_clinico(p) for p in execucoes]
        linhas.append(
            {
                "braco": braco,
                "sementes": len(execucoes),
                "custo_medio": round(mean(custos), 6),
                "custo_desvio": round(_desvio(custos), 6),
                "fitness_medio": round(mean(p["avaliacao"]["fitness"] for p in execucoes), 6),
                "recall_medio": round(mean(p["avaliacao"]["recall"] for p in execucoes), 6),
                "falsos_negativos_medio": round(
                    mean(p["avaliacao"]["falsos_negativos"] for p in execucoes), 2
                ),
                "falsos_positivos_medio": round(
                    mean(p["avaliacao"]["falsos_positivos"] for p in execucoes), 2
                ),
                "features_medio": round(
                    mean(p["avaliacao"]["n_features"] for p in execucoes), 2
                ),
            }
        )

    return sorted(linhas, key=lambda linha: linha["custo_medio"])


def frequencia_de_features(
    payloads: list[dict[str, Any]],
    shap_global: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Quantas execucoes selecionaram cada medida, cruzado com o ranking SHAP.

    A Fase 1 encerrou a explicabilidade mostrando tres tecnicas independentes
    convergindo nas mesmas medidas. O algoritmo genetico e um quarto caminho,
    matematicamente distinto: se ele reencontra as mesmas features, isso e
    confirmacao independente, e e por isso que a posicao no ranking SHAP viaja
    junto com a contagem.
    """
    posicoes = {item["feature"]: posicao for posicao, item in enumerate(shap_global, start=1)}

    contador: Counter = Counter()
    for payload in payloads:
        contador.update(payload["melhor"]["features"])

    return [
        {
            "feature": feature,
            "selecoes": vezes,
            "execucoes": len(payloads),
            "frequencia": round(vezes / len(payloads), 4),
            "posicao_shap": posicoes.get(feature),
        }
        for feature, vezes in contador.most_common()
    ]


def montar_summary(
    payloads: list[dict[str, Any]],
    shap_global: list[dict[str, Any]],
) -> dict[str, Any]:
    """Consolidado de `results/genetic/summary.json`.

    O relatorio nao digita numero a mao: ele le este arquivo, pela mesma regra
    que `results/metrics.json` segue desde a Fase 1.
    """
    return {
        "execucoes": len(payloads),
        "experimentos": agregar_por_experimento(payloads),
        "bracos": tabela_de_bracos(payloads),
        "features": frequencia_de_features(payloads, shap_global),
    }
