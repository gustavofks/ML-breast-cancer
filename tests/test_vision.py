"""Testes da entrega extra (imagens).

Usam uma base sintetica criada em disco, sem depender do dataset real: o que se
verifica aqui e a convencao de pastas e a montagem dos conjuntos, que precisam
funcionar com qualquer base organizada por classe.
"""

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="entrega extra: requer requirements-vision.txt")

from src.vision import dataset as vdata  # noqa: E402
from src.vision import model as vmodel  # noqa: E402

TAMANHO = (64, 64)


@pytest.fixture(scope="module")
def base_sintetica(tmp_path_factory):
    """Cria 2 classes com imagens aleatorias, na convencao de uma pasta por classe."""
    from PIL import Image

    raiz = tmp_path_factory.mktemp("imagens")
    gerador = np.random.default_rng(42)

    for classe, quantidade in (("benign", 24), ("malignant", 12)):
        (raiz / classe).mkdir()
        for indice in range(quantidade):
            pixels = gerador.integers(0, 255, (*TAMANHO, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(raiz / classe / f"{classe}_{indice:03d}.png")

    return raiz


def test_classes_vem_das_subpastas(base_sintetica):
    assert vdata.discover_classes(base_sintetica) == ["benign", "malignant"]


def test_base_ausente_falha_com_mensagem_clara(tmp_path):
    with pytest.raises(vdata.ImageDatasetError, match="não encontrada"):
        vdata.discover_classes(tmp_path / "inexistente")


def test_contagem_por_classe(base_sintetica):
    contagens = vdata.count_images(base_sintetica)

    assert contagens["imagens"].sum() == 36
    assert dict(zip(contagens["classe"], contagens["imagens"])) == {"benign": 24, "malignant": 12}
    assert abs(contagens["proporcao"].sum() - 1.0) < 1e-6


def test_mascaras_nao_sao_contadas_como_amostras(tmp_path):
    """O BUSI distribui a máscara ao lado da imagem; ela não é um exame."""
    from PIL import Image

    for classe in ("benign", "malignant"):
        (tmp_path / classe).mkdir()
        for indice in range(3):
            imagem = Image.new("RGB", (16, 16))
            imagem.save(tmp_path / classe / f"{classe} ({indice}).png")
            imagem.save(tmp_path / classe / f"{classe} ({indice})_mask.png")

    contagens = vdata.count_images(tmp_path)
    assert contagens["imagens"].sum() == 6  # 6 imagens, 6 máscaras ignoradas
    assert len(vdata.find_masks(tmp_path)) == 6


def test_carregamento_falha_se_houver_mascaras_na_pasta(tmp_path):
    """Falhar aqui é melhor que treinar com o conjunto contaminado em silêncio."""
    from PIL import Image

    for classe in ("benign", "malignant"):
        (tmp_path / classe).mkdir()
        for indice in range(3):
            imagem = Image.new("RGB", (16, 16))
            imagem.save(tmp_path / classe / f"{classe}_{indice}.png")
    Image.new("RGB", (16, 16)).save(tmp_path / "benign" / "benign_0_mask.png")

    with pytest.raises(vdata.ImageDatasetError, match="máscaras"):
        vdata.load_datasets(tmp_path, image_size=(16, 16), batch_size=2)


def test_classe_positiva_e_a_maligna(base_sintetica):
    classes = vdata.discover_classes(base_sintetica)

    assert classes[vdata.positive_index(classes)] == "malignant"


def test_pesos_compensam_o_desbalanceamento(base_sintetica):
    classes = vdata.discover_classes(base_sintetica)
    pesos = vdata.class_weights(vdata.count_images(base_sintetica), classes)

    # A classe com metade das imagens recebe o dobro de peso.
    assert pesos[1] > pesos[0]
    assert round(pesos[1] / pesos[0], 2) == 2.0


def test_conjuntos_sao_disjuntos_e_nao_vazios(base_sintetica):
    treino, validacao, teste, classes = vdata.load_datasets(
        base_sintetica, image_size=TAMANHO, batch_size=4, validation_split=0.4
    )

    assert classes == ["benign", "malignant"]
    for conjunto in (treino, validacao, teste):
        assert conjunto.cardinality().numpy() > 0

    imagens, rotulos = next(iter(treino))
    assert imagens.shape[1:] == (*TAMANHO, 3)
    assert rotulos.shape[0] == imagens.shape[0]


def test_cnn_binaria_tem_uma_saida_sigmoide(base_sintetica):
    modelo = vmodel.build_cnn(n_classes=2, image_size=TAMANHO)

    assert modelo.output_shape[-1] == 1
    assert modelo.loss == "binary_crossentropy"

    previsao = modelo.predict(np.zeros((2, *TAMANHO, 3), dtype=np.float32), verbose=0)
    assert previsao.shape == (2, 1)
    assert ((previsao >= 0) & (previsao <= 1)).all()


def test_cnn_multiclasse_tem_uma_saida_por_classe():
    modelo = vmodel.build_cnn(n_classes=3, image_size=TAMANHO)

    assert modelo.output_shape[-1] == 3
    assert modelo.loss == "categorical_crossentropy"

    previsao = modelo.predict(np.zeros((2, *TAMANHO, 3), dtype=np.float32), verbose=0)
    assert np.allclose(previsao.sum(axis=1), 1.0, atol=1e-5)


def test_callbacks_incluem_parada_antecipada():
    nomes = [type(c).__name__ for c in vmodel.callbacks()]

    assert "EarlyStopping" in nomes
    assert "ReduceLROnPlateau" in nomes


# ---------------------------------------------------------------------------
# Treino e avaliacao
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def conjuntos(base_sintetica):
    return vdata.load_datasets(
        base_sintetica, image_size=TAMANHO, batch_size=4, validation_split=0.4
    )


def test_treino_registra_historico_e_tempo(conjuntos):
    from src.vision import train as vtrain

    treino, validacao, _, class_names = conjuntos
    modelo = vmodel.build_cnn(n_classes=len(class_names), image_size=TAMANHO)

    historico, segundos = vtrain.train_model(modelo, treino, validacao, epochs=1, verbose=0)

    assert len(historico.history["loss"]) == 1
    assert "val_loss" in historico.history
    assert segundos > 0


def test_previsoes_cobrem_todo_o_conjunto(conjuntos):
    from src.vision import evaluate as vevaluate

    treino, validacao, teste, class_names = conjuntos
    modelo = vmodel.build_cnn(n_classes=len(class_names), image_size=TAMANHO)
    modelo.fit(treino, validation_data=validacao, epochs=1, verbose=0)

    y_true, y_pred, y_proba = vevaluate.predict_dataset(modelo, teste)

    assert len(y_true) == len(y_pred) == len(y_proba)
    assert set(np.unique(y_pred)) <= set(range(len(class_names)))


def test_metricas_de_imagem_sao_consistentes(conjuntos):
    from src.vision import evaluate as vevaluate

    treino, validacao, teste, class_names = conjuntos
    modelo = vmodel.build_cnn(n_classes=len(class_names), image_size=TAMANHO)
    modelo.fit(treino, validation_data=validacao, epochs=1, verbose=0)

    metricas = vevaluate.evaluate(modelo, teste, class_names)

    for chave in ("accuracy", "recall", "f1", "recall_maligno"):
        assert 0.0 <= metricas[chave] <= 1.0

    # Malignos perdidos nunca podem exceder os malignos presentes no conjunto.
    assert metricas["malignos_nao_detectados"] <= metricas["malignos_no_teste"]

    if metricas["malignos_no_teste"]:
        detectados = metricas["malignos_no_teste"] - metricas["malignos_nao_detectados"]
        esperado = round(detectados / metricas["malignos_no_teste"], 4)
        assert abs(metricas["recall_maligno"] - esperado) < 1e-4
