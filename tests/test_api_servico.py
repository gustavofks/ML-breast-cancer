"""Testes do nucleo de dominio da API: o que e servido e como se preve."""

import json

import pytest

from src import config
from src.api import servico


def test_sem_arquivo_serve_a_referencia_da_fase_1_e_diz_isso(tmp_path):
    """Servir a referencia em silencio faria /model-info mentir sobre a origem."""
    especificacao = servico.carregar_especificacao(tmp_path / "nao_existe.json")

    assert especificacao == servico.referencia_da_fase_1()
    assert "Fase 1" in especificacao.origem


def test_referencia_usa_as_30_medidas_e_o_limiar_padrao():
    referencia = servico.referencia_da_fase_1()

    assert referencia.familia == "Regressão Logística"
    assert len(referencia.features) == config.EXPECTED_N_FEATURES
    assert referencia.limiar == 0.5


def test_arquivo_no_formato_do_melhor_individuo_e_aceito(tmp_path):
    """O usuario copia o `melhor` de uma execucao do AG, sem traducao."""
    caminho = tmp_path / "modelo.json"
    caminho.write_text(
        json.dumps(
            {
                "familia": "Regressão Logística",
                "hiperparametros": {"C": 0.5, "class_weight": "balanced"},
                "features": ["texture_worst", "area_worst"],
                "limiar": 0.3,
                "origem": "AG, completo, E1, semente 42",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    especificacao = servico.carregar_especificacao(caminho)

    assert especificacao.features == ("texture_worst", "area_worst")
    assert especificacao.limiar == 0.3


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("familia", "SVM"),
        ("features", []),
        ("features", ["nao_existe"]),
        ("limiar", 1.5),
        ("limiar", 0.0),
    ],
)
def test_especificacao_invalida_falha_na_carga(campo, valor):
    """Erro na inicializacao, nao na primeira requisicao de um medico."""
    dados = servico.referencia_da_fase_1().para_dicionario()
    dados[campo] = valor

    with pytest.raises(ValueError):
        servico.EspecificacaoDoModelo.de_dicionario(dados)


def test_especificacao_ida_e_volta_pelo_dicionario_e_identidade():
    referencia = servico.referencia_da_fase_1()

    assert servico.EspecificacaoDoModelo.de_dicionario(referencia.para_dicionario()) == referencia


# ---------------------------------------------------------------------------
# Treino na inicializacao e previsao
# ---------------------------------------------------------------------------
from src import models  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.preprocessing import split_data  # noqa: E402


@pytest.fixture(scope="module")
def referencia():
    return servico.Servico.construir(servico.referencia_da_fase_1())


@pytest.fixture(scope="module")
def teste():
    X, y = load_dataset()
    _, X_test, _, _ = split_data(X, y)
    return X_test


def test_referencia_preve_o_mesmo_que_o_modelo_da_fase_1(referencia, teste):
    """E o que torna legitimo chamar o modelo servido de "a referencia"."""
    X, y = load_dataset()
    X_train, _, y_train, _ = split_data(X, y)
    fase_1 = models.get_models()["Regressão Logística"].fit(X_train, y_train)

    esperado = fase_1.predict_proba(teste.head(10))[:, 1]
    obtido = [referencia.prever(linha.to_dict()).probabilidade for _, linha in teste.head(10).iterrows()]

    assert obtido == pytest.approx(list(esperado))


def test_classe_sai_da_comparacao_com_o_limiar_servido(teste):
    """Limiar 0,05 torna maligno quase tudo; e a regra servida, nao o 0,5 do sklearn."""
    baixo = servico.EspecificacaoDoModelo.de_dicionario(
        {**servico.referencia_da_fase_1().para_dicionario(), "limiar": 0.05}
    )
    modelo = servico.Servico.construir(baixo)

    for _, linha in teste.head(30).iterrows():
        previsao = modelo.prever(linha.to_dict())
        esperada = "Maligno" if previsao.probabilidade >= 0.05 else "Benigno"
        assert previsao.classe == esperada
        assert previsao.limiar == 0.05


def test_medida_faltante_e_nomeada_no_erro(referencia, teste):
    incompleto = {k: v for k, v in teste.iloc[0].to_dict().items() if k != "area_worst"}

    with pytest.raises(servico.MedidasInvalidas, match="area_worst"):
        referencia.prever(incompleto)


def test_medida_desconhecida_e_recusada(referencia, teste):
    """Nome errado ignorado em silencio esconderia um erro de digitacao."""
    with pytest.raises(servico.MedidasInvalidas, match="area_wrost"):
        referencia.prever({**teste.iloc[0].to_dict(), "area_wrost": 1.0})


def test_medida_nao_finita_e_recusada(referencia, teste):
    with pytest.raises(servico.MedidasInvalidas, match="radius_mean"):
        referencia.prever({**teste.iloc[0].to_dict(), "radius_mean": float("nan")})


def test_modelo_com_subconjunto_aceita_as_30_medidas(teste):
    """O painel manda o caso inteiro; as medidas a mais so alimentam a sentinela."""
    reduzido = servico.EspecificacaoDoModelo.de_dicionario(
        {
            **servico.referencia_da_fase_1().para_dicionario(),
            "features": ["area_worst", "texture_worst"],
        }
    )
    modelo = servico.Servico.construir(reduzido)

    previsao = modelo.prever(teste.iloc[0].to_dict())

    assert 0.0 <= previsao.probabilidade <= 1.0
    # Ordem da base, nao a do arquivo: e a ordem que o modelo avaliado viu.
    assert modelo.selecionadas == ["texture_worst", "area_worst"]


def test_modelo_servido_nao_ve_o_conjunto_de_teste(referencia):
    """Treinado so no treino, como o modelo da avaliacao final."""
    X, y = load_dataset()
    X_train, _, _, _ = split_data(X, y)

    assert len(referencia.X_train) == len(X_train)
    assert referencia.X_train.index.equals(X_train.index)
