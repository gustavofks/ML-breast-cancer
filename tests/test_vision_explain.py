"""Testes do Grad-CAM."""

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="entrega extra: requer requirements-vision.txt")

from src.vision import explain as vexplain  # noqa: E402
from src.vision import model as vmodel  # noqa: E402

TAMANHO = (64, 64)


@pytest.fixture(scope="module")
def imagem():
    return np.random.default_rng(7).integers(0, 255, (*TAMANHO, 3)).astype("float32")


def test_camada_alvo_e_convolucional_nao_o_aumento():
    """O bloco de aumento também devolve 4 dimensões e não pode ser escolhido."""
    cnn = vmodel.build_cnn(n_classes=3, image_size=TAMANHO)

    alvo = vexplain.last_conv_layer(cnn)

    assert alvo != "aumento"
    camada = cnn.get_layer(alvo)
    assert len(camada.output.shape) == 4


def test_camada_alvo_de_transferencia_e_a_base_pre_treinada():
    modelo = vmodel.build_transfer_model(n_classes=3, image_size=(96, 96))

    assert vexplain.last_conv_layer(modelo).startswith("mobilenetv2")


def test_percorrer_as_camadas_reproduz_a_previsao_do_modelo(imagem):
    """Garante que o Grad-CAM vê exatamente a mesma conta que o `predict`.

    Uma função aplicada ao tensor durante a construção do modelo — como a
    normalização da MobileNetV2 — é absorvida pelo grafo e não aparece em
    `modelo.layers`. Percorrer as camadas pularia essa etapa e alimentaria a
    rede com a escala errada, produzindo mapas de calor sem relação com a
    previsão real. O modelo declara a normalização como camada justamente para
    que esta igualdade valha.
    """
    import numpy as np
    import tensorflow as tf

    modelo = vmodel.build_transfer_model(n_classes=3, image_size=(96, 96))
    entrada = np.random.default_rng(5).integers(0, 255, (96, 96, 3)).astype("float32")

    esperado = modelo.predict(entrada[None], verbose=0)[0]

    x = tf.convert_to_tensor(entrada[None], dtype=tf.float32)
    for camada in modelo.layers:
        if camada.__class__.__name__ == "InputLayer":
            continue
        x = camada(x, training=False)

    assert np.allclose(esperado, x.numpy()[0], atol=1e-5)


def test_mapa_normalizado_entre_zero_e_um(imagem):
    modelo = vmodel.build_cnn(n_classes=3, image_size=TAMANHO)

    mapa, classe, probabilidade = vexplain.gradcam(modelo, imagem)

    assert mapa.ndim == 2
    assert 0.0 <= mapa.min() and mapa.max() <= 1.0
    assert classe in (0, 1, 2)
    assert 0.0 <= probabilidade <= 1.0


def test_mapa_tem_a_resolucao_do_mapa_de_ativacao(imagem):
    """Três blocos de pooling reduzem 64x64 para 8x8."""
    modelo = vmodel.build_cnn(n_classes=3, image_size=TAMANHO)

    mapa, _, _ = vexplain.gradcam(modelo, imagem)

    assert mapa.shape == (8, 8)


def test_classe_explicada_pode_ser_forcada(imagem):
    modelo = vmodel.build_cnn(n_classes=3, image_size=TAMANHO)

    _, classe, _ = vexplain.gradcam(modelo, imagem, classe=2)

    assert classe == 2


def test_grade_vazia_nao_gera_figura():
    modelo = vmodel.build_cnn(n_classes=3, image_size=TAMANHO)

    assert vexplain.plot_gradcam_grid(modelo, [], ["a", "b", "c"], "t", "x.png") is None
