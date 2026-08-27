"""Arquiteturas de rede para o diagnostico por imagem.

Duas alternativas, pela mesma razao que o pipeline tabular compara tres modelos:
o contraste entre elas e o que torna a conclusao informativa.

1. **CNN do zero** — pequena, treinada apenas com as imagens do dataset. Serve de
   baseline e mostra o que se consegue sem conhecimento externo.
2. **Transfer learning (MobileNetV2)** — reaproveita filtros aprendidos em
   milhoes de imagens naturais. Costuma vencer com folga em bases medicas
   pequenas, onde treinar do zero leva a sobreajuste.

Ambas recebem a mesma camada de aumento de dados e sao compiladas com as mesmas
metricas, para que a comparacao seja justa.
"""

from __future__ import annotations

from src import config


def _augmentation():
    """Aumento de dados aplicado somente durante o treino.

    Giros e rotacoes leves sao seguros em imagens medicas: nao alteram o que a
    lesao e. Distorcoes agressivas de cor seriam arriscadas, porque a
    intensidade carrega informacao diagnostica.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="aumento",
    )


def _saida(n_classes: int):
    """Camada final e funcao de perda adequadas ao numero de classes."""
    if n_classes == 2:
        return 1, "sigmoid", "binary_crossentropy"
    return n_classes, "softmax", "categorical_crossentropy"


def _metricas(n_classes: int):
    """Metricas acompanhadas no treino.

    Recall e precisao so sao calculaveis diretamente no caso binario; com tres
    ou mais classes, a avaliacao detalhada fica para `src/vision/evaluate.py`,
    sobre as previsoes finais.
    """
    from tensorflow import keras

    if n_classes == 2:
        return [
            "accuracy",
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.AUC(name="auc"),
        ]
    return ["accuracy"]


def build_cnn(
    n_classes: int,
    image_size: tuple[int, int] = config.IMAGE_SIZE,
    learning_rate: float = 1e-3,
):
    """CNN pequena treinada do zero: tres blocos convolucionais e uma cabeca densa."""
    from tensorflow import keras
    from tensorflow.keras import layers

    unidades, ativacao, perda = _saida(n_classes)

    modelo = keras.Sequential(
        [
            keras.Input(shape=(*image_size, 3)),
            _augmentation(),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(unidades, activation=ativacao),
        ],
        name="cnn_do_zero",
    )

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=perda,
        metrics=_metricas(n_classes),
    )
    return modelo


def build_transfer_model(
    n_classes: int,
    image_size: tuple[int, int] = config.IMAGE_SIZE,
    learning_rate: float = 1e-3,
    trainable_base: bool = False,
):
    """MobileNetV2 pre-treinada na ImageNet, com cabeca nova.

    Args:
        trainable_base: quando `True`, libera a base para ajuste fino. Deve ser
            usado apenas em uma segunda etapa, com taxa de aprendizado bem menor,
            depois que a cabeca ja convergiu.
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    unidades, ativacao, perda = _saida(n_classes)

    base = MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights="imagenet")
    base.trainable = trainable_base

    entradas = keras.Input(shape=(*image_size, 3))
    x = _augmentation()(entradas)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    saidas = layers.Dense(unidades, activation=ativacao)(x)

    modelo = keras.Model(entradas, saidas, name="mobilenetv2_transferencia")
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=perda,
        metrics=_metricas(n_classes),
    )
    return modelo


def enable_fine_tuning(modelo, n_camadas: int = 40, learning_rate: float = 1e-5):
    """Libera as ultimas camadas da base pre-treinada para ajuste fino.

    Treinar a cabeca com a base congelada aprende *o que fazer* com as features
    da ImageNet; o ajuste fino adapta as proprias features ao dominio — texturas
    de ultrassom nao se parecem com fotografias naturais.

    Tres cuidados que evitam destruir o que ja foi aprendido:

    - **Somente as camadas finais.** As primeiras detectam bordas e texturas
      genericas, uteis em qualquer imagem; as ultimas e que sao especificas da
      ImageNet.
    - **Taxa de aprendizado muito menor** (1e-5 contra 1e-3). Com a taxa
      original, os gradientes da cabeca recem-treinada apagariam os pesos
      pre-treinados na primeira epoca.
    - **BatchNormalization permanece congelada.** Atualizar suas estatisticas com
      lotes pequenos degrada a normalizacao aprendida em milhoes de imagens.

    Returns:
        O proprio modelo, recompilado e pronto para continuar o treino.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    # Cuidado necessario: a camada de aumento de dados tambem e um modelo
    # (Sequential herda de Model). A base pre-treinada e a unica submodelo
    # funcional, entao e por ai que ela se distingue.
    base = next(
        (
            camada
            for camada in modelo.layers
            if isinstance(camada, keras.Model) and not isinstance(camada, keras.Sequential)
        ),
        None,
    )
    if base is None:
        raise TypeError("Modelo sem base pré-treinada: nada a liberar para ajuste fino.")

    base.trainable = True
    for camada in base.layers[:-n_camadas]:
        camada.trainable = False
    for camada in base.layers:
        if isinstance(camada, layers.BatchNormalization):
            camada.trainable = False

    n_classes = 2 if modelo.output_shape[-1] == 1 else modelo.output_shape[-1]
    _, _, perda = _saida(n_classes)

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=perda,
        metrics=_metricas(n_classes),
    )
    return modelo


def trainable_layers(modelo) -> int:
    """Quantidade de camadas treinaveis, para registrar o efeito do ajuste fino."""
    return sum(1 for camada in modelo.layers if camada.trainable) + sum(
        1
        for camada in modelo.layers
        if hasattr(camada, "layers")
        for sub in camada.layers
        if sub.trainable
    )


def callbacks(patience: int = 6):
    """Parada antecipada e reducao de taxa de aprendizado.

    A parada monitora a perda de validacao e restaura os melhores pesos: em base
    pequena, o sobreajuste chega rapido, e treinar ate a ultima epoca costuma
    piorar o resultado.
    """
    from tensorflow import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
        ),
    ]


def get_models(n_classes: int, image_size: tuple[int, int] = config.IMAGE_SIZE) -> dict:
    """Catalogo de modelos comparados na entrega extra."""
    return {
        "CNN do zero": build_cnn(n_classes, image_size),
        "MobileNetV2 (transferência)": build_transfer_model(n_classes, image_size),
    }
