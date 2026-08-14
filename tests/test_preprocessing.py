"""Testes do pipeline de pre-processamento e do split treino/teste."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data import load_dataset
from src.preprocessing import build_pipeline, build_preprocessor, split_data


def test_split_preserva_proporcao_das_classes():
    X, y = load_dataset()
    _, _, y_train, y_test = split_data(X, y)

    proporcao_original = (y == 1).mean()
    assert abs((y_train == 1).mean() - proporcao_original) < 0.01
    assert abs((y_test == 1).mean() - proporcao_original) < 0.01


def test_split_respeita_o_tamanho_configurado():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_test) == round(len(X) * config.TEST_SIZE)
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)


def test_split_e_reprodutivel_com_a_mesma_semente():
    X, y = load_dataset()
    primeiro, _, _, _ = split_data(X, y)
    segundo, _, _, _ = split_data(X, y)

    assert primeiro.index.equals(segundo.index)


def test_scaler_e_ajustado_apenas_com_dados_de_treino():
    """Garante que nao ha vazamento: o scaler nunca ve o conjunto de teste."""
    X, y = load_dataset()
    X_train, _, _, _ = split_data(X, y)

    preprocessor = build_preprocessor().fit(X_train)
    medias_do_scaler = preprocessor.named_steps["scaler"].mean_

    assert np.allclose(medias_do_scaler, X_train.mean().to_numpy())
    assert not np.allclose(medias_do_scaler, X.mean().to_numpy())


def test_preprocessador_padroniza_as_features():
    X, y = load_dataset()
    X_train, _, _, _ = split_data(X, y)

    transformado = build_preprocessor().fit_transform(X_train)

    assert np.allclose(transformado.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(transformado.std(axis=0), 1, atol=1e-9)


def test_build_pipeline_encadeia_preprocessamento_e_estimador():
    X, y = load_dataset()
    X_train, X_test, y_train, _ = split_data(X, y)

    pipeline = build_pipeline(LogisticRegression(max_iter=5000))
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "estimator"]

    pipeline.fit(X_train, y_train)
    previsoes = pipeline.predict(X_test)

    assert len(previsoes) == len(X_test)
    assert set(np.unique(previsoes)) <= {0, 1}
