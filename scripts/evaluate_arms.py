"""Avaliacao final de todos os bracos contra o conjunto de teste intocado.

Uso, a partir da raiz do projeto:

    python scripts/evaluate_arms.py                      # todos os bracos
    python scripts/evaluate_arms.py --somente-referencia # so Original e grade

**Este e o unico ponto da Fase 2 que le o conjunto de teste, e ele roda uma vez,
depois de toda a busca estar concluida.** Nenhum modulo de `src/genetic/` tem
acesso a estes dados, e existe teste que falha se algum passar a ter — e por isso
que este codigo vive em `scripts/`, e nao dentro do pacote.

Os bracos de referencia saem de `src.models`, e nao de uma reconstrucao
equivalente: se a referencia nao for o modelo da Fase 1 exatamente, o ganho
reportado e artefato de implementacao, nao resultado.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config, models  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.genetic.chromosome import Individuo, decodificar  # noqa: E402
from src.preprocessing import split_data  # noqa: E402

FAMILIA_DE_REFERENCIA = "Regressão Logística"

# Os bracos que nao otimizam limiar usam 0,5, o padrao do sklearn. O 0,30 da
# Fase 1 foi calibrado olhando o conjunto de teste, e a Fase 2 existe em parte
# para corrigir essa fragilidade: herdar aquele numero contaminaria justamente a
# comparacao que a correcao deveria tornar honesta.
LIMIAR_DE_REFERENCIA = 0.5


def bracos_de_referencia() -> dict[str, Pipeline]:
    """Os dois bracos que nao usam algoritmo genetico, direto da Fase 1.

    Cada braco recebe uma copia propria por `clone`: `Pipeline` guarda estado
    depois do `fit`, e devolver o mesmo objeto nas duas chaves faria o ajuste de
    um braco contaminar o outro — divergencia silenciosa, que nao levanta
    excecao e so aparece como numero estranho na tabela final.
    """
    catalogo = models.get_models()
    return {
        "Original": clone(catalogo[FAMILIA_DE_REFERENCIA]),
        "Busca em grade": clone(catalogo[FAMILIA_DE_REFERENCIA]),
    }


def _metricas(y_true, probabilidades, limiar: float, n_features: int, braco: str) -> dict:
    """Metricas tradicionais mais o custo clinico, que e o que decide aqui."""
    previsoes = (probabilidades >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, previsoes, labels=[0, 1]).ravel()
    custo = (
        config.CUSTO_FALSO_NEGATIVO * fn + config.CUSTO_FALSO_POSITIVO * fp
    ) / len(y_true)

    return {
        "braco": braco,
        "recall": round(float(recall_score(y_true, previsoes, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, previsoes, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilidades)), 4),
        "falsos_negativos": int(fn),
        "falsos_positivos": int(fp),
        "verdadeiros_negativos": int(tn),
        "verdadeiros_positivos": int(tp),
        "custo_clinico": round(float(custo), 6),
        "n_features": int(n_features),
        "limiar": round(float(limiar), 4),
    }


def avaliar_braco_de_referencia(braco: str) -> dict:
    """Treina o braco de referencia no treino e avalia no teste."""
    X, y = load_dataset()
    X_train, X_avaliacao, y_train, y_avaliacao = split_data(X, y)

    pipeline = bracos_de_referencia()[braco]
    if braco == "Busca em grade":
        ajustados, _ = models.tune_models({FAMILIA_DE_REFERENCIA: pipeline}, X_train, y_train)
        modelo = ajustados[FAMILIA_DE_REFERENCIA]
    else:
        modelo = pipeline.fit(X_train, y_train)

    probabilidades = modelo.predict_proba(X_avaliacao)[:, 1]
    return _metricas(
        y_avaliacao, probabilidades, LIMIAR_DE_REFERENCIA, X_train.shape[1], braco
    )


def avaliar_individuo(payload: dict, braco: str) -> dict:
    """Treina o melhor individuo de um braco do AG e avalia no teste.

    O individuo e reconstruido a partir do artefato e decodificado pelo mesmo
    `decodificar` que a busca usou, entao o modelo avaliado aqui e o mesmo que o
    fitness pontuou — nao uma reconstrucao parecida.
    """
    X, y = load_dataset()
    X_train, X_avaliacao, y_train, y_avaliacao = split_data(X, y)

    melhor = payload["melhor"]
    individuo = Individuo(
        familia=melhor["familia"],
        mascara=tuple(melhor["mascara"]),
        hiperparametros=melhor["hiperparametros"],
        limiar=melhor["limiar"],
    )
    pipeline, selecionadas = decodificar(individuo, list(X_train.columns))
    modelo = pipeline.fit(X_train.loc[:, selecionadas], y_train)

    probabilidades = modelo.predict_proba(X_avaliacao.loc[:, selecionadas])[:, 1]
    return _metricas(
        y_avaliacao, probabilidades, individuo.limiar, len(selecionadas), braco
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliacao final no conjunto de teste.")
    parser.add_argument("--execucoes", type=Path, default=None)
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--somente-referencia", action="store_true")
    argumentos = parser.parse_args()

    tabela = [avaliar_braco_de_referencia(braco) for braco in bracos_de_referencia()]

    if not argumentos.somente_referencia:
        origem = argumentos.execucoes or (config.GENETIC_DIR / "execucoes.json")
        payloads = json.loads(origem.read_text(encoding="utf-8"))

        # Um braco por nome, usando a execucao de melhor fitness entre as
        # sementes. A media por semente vive no summary.json; aqui a pergunta e
        # qual solucao de fato seria implantada.
        melhores: dict[str, dict] = {}
        for payload in payloads:
            braco = payload.get("braco", "AG, completo")
            atual = melhores.get(braco)
            if atual is None or payload["avaliacao"]["fitness"] > atual["avaliacao"]["fitness"]:
                melhores[braco] = payload

        tabela += [avaliar_individuo(payload, braco) for braco, payload in melhores.items()]

    tabela.sort(key=lambda linha: linha["custo_clinico"])

    destino = argumentos.saida or (config.GENETIC_DIR / "comparacao_final.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(tabela, indent=2, ensure_ascii=False), encoding="utf-8")

    for linha in tabela:
        print(
            f"{linha['braco']:<24} custo {linha['custo_clinico']:.4f} | "
            f"recall {linha['recall']:.4f} | FN {linha['falsos_negativos']:>2} | "
            f"FP {linha['falsos_positivos']:>2} | features {linha['n_features']:>2}"
        )
    print(f"Gravado em {destino}")


if __name__ == "__main__":
    main()
