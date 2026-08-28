"""Explicabilidade das previsoes por imagem: Grad-CAM.

O pipeline tabular responde "por que esta paciente?" com SHAP, decompondo a
previsao em contribuicoes por medida. Em imagem nao existem medidas — existem
pixels — e a pergunta equivalente e **onde a rede olhou**.

O Grad-CAM responde exatamente isso. Ele pega o ultimo mapa de ativacao
convolucional, mede o quanto cada canal desse mapa influencia a pontuacao da
classe prevista (pelo gradiente) e soma os canais ponderados por essa
influencia. O resultado e um mapa de calor na resolucao do mapa de ativacao,
que ampliado sobre a imagem original mostra a regiao que sustentou a decisao.

Vale a ressalva honesta: o mapa mostra *onde*, nao *por que*. Ele nao afirma que
a rede reconheceu uma margem espiculada — apenas que aquela regiao pesou. Ainda
assim, e o que permite a um medico discordar de forma fundamentada: um mapa
centrado fora da lesao denuncia uma previsao correta pelo motivo errado.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src import config
from src.plotting import INK, INK_MUTED, save_figure
from src.vision.dataset import positive_index

# Rampa sequencial de uma cor so, do transparente ao vermelho escuro. Evita o
# arco-iris do mapa "jet" tradicional, em que o meio da escala vira uma cor
# propria e o olho le regioes intermediarias como categorias distintas.
CALOR = LinearSegmentedColormap.from_list(
    "atencao",
    [
        (0.0, (1.0, 1.0, 1.0, 0.0)),
        (0.35, (0.93, 0.41, 0.20, 0.35)),
        (0.70, (0.85, 0.23, 0.12, 0.65)),
        (1.0, (0.56, 0.05, 0.05, 0.85)),
    ],
)


def last_conv_layer(modelo) -> str:
    """Nome da ultima camada que produz um mapa de ativacao 2D.

    Em um modelo de transferencia, essa camada e a propria base pre-treinada,
    que devolve um tensor `(altura, largura, canais)`. Em uma CNN construida do
    zero, e a ultima convolucao.
    """
    for camada in reversed(modelo.layers):
        # No Keras 3 a maioria das camadas nao expoe mais `output_shape`; o
        # formato vem do tensor de saida. O bloco de aumento de dados tambem
        # devolve 4 dimensoes, mas fica antes de qualquer convolucao, entao a
        # varredura de tras para frente nunca chega nele em um modelo valido.
        saida = getattr(camada, "output", None)
        formato = getattr(saida, "shape", None)
        if formato is not None and len(formato) == 4 and camada.name != "aumento":
            return camada.name
    raise TypeError("Modelo sem mapa de ativação convolucional: Grad-CAM não se aplica.")


def gradcam(
    modelo,
    imagem: np.ndarray,
    classe: int | None = None,
    nome_camada: str | None = None,
):
    """Mapa de calor Grad-CAM de uma imagem.

    Args:
        modelo: rede treinada.
        imagem: imagem em escala 0-255, no formato `(altura, largura, 3)`.
        classe: indice da classe a explicar. Usa a classe prevista se omitido.
        nome_camada: camada de ativacao a inspecionar. Detectada se omitida.

    Returns:
        `(mapa, classe_explicada, probabilidade)`, com o mapa normalizado em
        0-1 e no tamanho do mapa de ativacao (nao da imagem).
    """
    import tensorflow as tf

    alvo = nome_camada or last_conv_layer(modelo)
    entrada = tf.convert_to_tensor(imagem[None], dtype=tf.float32)

    with tf.GradientTape() as fita:
        x = entrada
        ativacao = None
        for camada in modelo.layers:
            if camada.__class__.__name__ == "InputLayer":
                continue
            x = camada(x, training=False)
            if camada.name == alvo:
                ativacao = x
                fita.watch(ativacao)

        previsao = x
        if previsao.shape[-1] == 1:  # binario
            indice = 0
            probabilidade = float(previsao[0, 0])
        else:
            indice = int(tf.argmax(previsao[0])) if classe is None else classe
            probabilidade = float(previsao[0, indice])
        pontuacao = previsao[:, indice]

    gradientes = fita.gradient(pontuacao, ativacao)
    pesos = tf.reduce_mean(gradientes, axis=(0, 1, 2))

    mapa = tf.reduce_sum(ativacao[0] * pesos, axis=-1)
    mapa = tf.nn.relu(mapa)  # so contribuicoes que empurram *a favor* da classe

    maximo = float(tf.reduce_max(mapa))
    mapa = mapa / maximo if maximo > 0 else mapa
    return mapa.numpy(), indice, probabilidade


def _sobrepor(ax, imagem: np.ndarray, mapa: np.ndarray) -> None:
    """Desenha a imagem em cinza com o mapa de calor por cima."""
    ax.imshow(imagem.astype("uint8"), cmap="gray")
    ax.imshow(
        mapa,
        cmap=CALOR,
        extent=(0, imagem.shape[1], imagem.shape[0], 0),
        interpolation="bilinear",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)


def collect_cases(
    modelo,
    conjunto,
    class_names: list[str],
    acertos: bool = True,
    limite: int = 3,
) -> list[tuple[np.ndarray, int, int]]:
    """Coleta casos malignos para explicar.

    Args:
        acertos: `True` junta malignos detectados corretamente; `False` junta os
            que escaparam.

    Returns:
        Lista de `(imagem, classe_real, classe_prevista)`.
    """
    positiva = positive_index(class_names)
    casos = []

    for imagens, alvos in conjunto:
        probabilidades = modelo.predict(imagens, verbose=0)
        preditos = (
            probabilidades.argmax(axis=1)
            if probabilidades.shape[1] > 1
            else (probabilidades.ravel() >= 0.5).astype(int)
        )
        reais = (
            alvos.numpy().argmax(axis=1)
            if alvos.shape[-1] > 1
            else alvos.numpy().ravel().astype(int)
        )

        condicao = (preditos == positiva) if acertos else (preditos != positiva)
        for indice in np.where((reais == positiva) & condicao)[0]:
            casos.append((imagens[indice].numpy(), positiva, int(preditos[indice])))
            if len(casos) >= limite:
                return casos

    return casos


def plot_gradcam_grid(
    modelo,
    casos: list[tuple[np.ndarray, int, int]],
    class_names: list[str],
    titulo: str,
    filename: str,
) -> Path | None:
    """Grade com a imagem original e o Grad-CAM sobreposto, um caso por coluna."""
    if not casos:
        return None

    fig, axes = plt.subplots(2, len(casos), figsize=(3.1 * len(casos), 6.4), squeeze=False)

    for coluna, (imagem, real, previsto) in enumerate(casos):
        mapa, _, probabilidade = gradcam(modelo, imagem, classe=real)

        axes[0][coluna].imshow(imagem.astype("uint8"), cmap="gray")
        axes[0][coluna].set_xticks([])
        axes[0][coluna].set_yticks([])
        axes[0][coluna].grid(visible=False)
        axes[0][coluna].set_title(
            f"real: {class_names[real]}\nprevisto: {class_names[previsto]}",
            fontsize=9,
            color=INK if real == previsto else INK_MUTED,
        )

        _sobrepor(axes[1][coluna], imagem, mapa)
        axes[1][coluna].set_xlabel(
            f"confiança em {class_names[real]}: {probabilidade:.2f}", fontsize=8, color=INK_MUTED
        )

    axes[0][0].set_ylabel("ultrassom original", fontsize=9, color=INK)
    axes[1][0].set_ylabel("Grad-CAM", fontsize=9, color=INK)

    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    return save_figure(fig, filename)


def plot_gradcam_examples(
    modelo,
    conjunto,
    class_names: list[str],
    limite: int = 3,
) -> tuple[Path | None, Path | None]:
    """Gera as duas grades: casos detectados e casos que escaparam."""
    detectados = collect_cases(modelo, conjunto, class_names, acertos=True, limite=limite)
    perdidos = collect_cases(modelo, conjunto, class_names, acertos=False, limite=limite)

    return (
        plot_gradcam_grid(
            modelo,
            detectados,
            class_names,
            "Grad-CAM — casos malignos detectados corretamente",
            "25_gradcam_acertos.png",
        ),
        plot_gradcam_grid(
            modelo,
            perdidos,
            class_names,
            "Grad-CAM — casos malignos que escaparam",
            "26_gradcam_erros.png",
        ),
    )
