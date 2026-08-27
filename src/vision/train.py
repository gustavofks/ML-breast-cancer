"""Treinamento das redes convolucionais.

O protocolo espelha o do pipeline tabular: os modelos sao comparados usando
treino e validacao, e o conjunto de teste so aparece na avaliacao final
(`src/vision/evaluate.py`).

Duas particularidades do caso de imagens:

- **Pesos de classe.** A base e desbalanceada (56% benignas, 27% malignas, 17%
  normais). Sem correcao, a rede aprende a favorecer a classe majoritaria,
  justamente o oposto do que interessa quando a classe rara e a maligna.
- **Parada antecipada.** Com poucas centenas de imagens, o sobreajuste chega em
  poucas epocas; treinar ate o fim costuma piorar o resultado de validacao.
"""

from __future__ import annotations

import time

import pandas as pd

from src import config
from src.vision import model as vmodel


def train_model(
    modelo,
    treino,
    validacao,
    class_weight: dict[int, float] | None = None,
    epochs: int = config.EPOCHS,
    verbose: int = 2,
):
    """Treina um modelo e devolve `(historico, segundos)`.

    O tempo de treino e devolvido junto porque, em uma entrega que compara uma
    rede pequena com uma pre-treinada, o custo computacional faz parte da
    comparacao.
    """
    inicio = time.perf_counter()
    historico = modelo.fit(
        treino,
        validation_data=validacao,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=vmodel.callbacks(),
        verbose=verbose,
    )
    return historico, round(time.perf_counter() - inicio, 1)


def train_all(
    modelos: dict,
    treino,
    validacao,
    class_weight: dict[int, float] | None = None,
    epochs: int = config.EPOCHS,
    verbose: int = 2,
) -> tuple[dict, dict, pd.DataFrame]:
    """Treina todos os modelos do catalogo.

    Returns:
        `(modelos_treinados, historicos, tabela_de_treino)`.
    """
    historicos = {}
    linhas = []

    for nome, modelo in modelos.items():
        print(f"   treinando: {nome}")
        historico, segundos = train_model(
            modelo, treino, validacao, class_weight, epochs, verbose
        )
        historicos[nome] = historico

        melhor_epoca = int(pd.Series(historico.history["val_loss"]).idxmin()) + 1
        linhas.append(
            {
                "modelo": nome,
                "parametros": int(modelo.count_params()),
                "epocas_treinadas": len(historico.history["loss"]),
                "melhor_epoca": melhor_epoca,
                "val_accuracy": round(float(historico.history["val_accuracy"][melhor_epoca - 1]), 4),
                "val_loss": round(float(historico.history["val_loss"][melhor_epoca - 1]), 4),
                "segundos": segundos,
            }
        )

    return modelos, historicos, pd.DataFrame(linhas)
