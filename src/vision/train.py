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


def configure_determinism(seed: int = config.SEED) -> None:
    """Fixa as sementes do TensorFlow para que o treino seja reproduzivel.

    Sem isso, cada execucao produz metricas diferentes: a inicializacao dos
    pesos, a ordem do embaralhamento e o aumento de dados usam geradores
    proprios. `enable_op_determinism` cobra um pouco de desempenho, aceitavel
    numa base deste tamanho, e e o que garante que rodar duas vezes devolva o
    mesmo resultado — a mesma promessa que o pipeline tabular ja cumpre.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


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


def fine_tune(
    modelo,
    treino,
    validacao,
    class_weight: dict[int, float] | None = None,
    epochs: int = 20,
    n_camadas: int = 40,
    learning_rate: float = 1e-5,
    verbose: int = 2,
):
    """Segunda etapa do treino: adapta as camadas finais da base pre-treinada.

    Deve ser chamada com um modelo **ja treinado** com a base congelada. Rodar o
    ajuste fino a partir de pesos aleatorios na cabeca destruiria a base, porque
    os gradientes iniciais seriam enormes.

    Returns:
        `(historico, segundos)`.
    """
    vmodel.enable_fine_tuning(modelo, n_camadas=n_camadas, learning_rate=learning_rate)
    return train_model(modelo, treino, validacao, class_weight, epochs, verbose)


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
