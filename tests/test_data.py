"""Testes do carregamento e da limpeza da base."""

import pandas as pd
import pytest

from src import config
from src.data import (
    DatasetValidationError,
    clean,
    load_dataset,
    load_raw,
    split_features_target,
)


def test_load_raw_traz_arquivo_original_com_coluna_fantasma():
    raw = load_raw()
    assert raw.shape[0] == config.EXPECTED_N_SAMPLES
    # O cabecalho do CSV termina em virgula: o pandas cria uma coluna extra nula.
    assert any(str(c).startswith("Unnamed") for c in raw.columns)
    assert config.ID_COLUMN in raw.columns


def test_clean_remove_id_e_coluna_fantasma():
    limpo = clean(load_raw())
    assert limpo.shape == (config.EXPECTED_N_SAMPLES, config.EXPECTED_N_FEATURES + 1)
    assert config.ID_COLUMN not in limpo.columns
    assert not any(str(c).startswith("Unnamed") for c in limpo.columns)
    assert limpo.isna().sum().sum() == 0


def test_clean_codifica_alvo_como_binario():
    limpo = clean(load_raw())
    alvo = limpo[config.TARGET_COLUMN]
    assert set(alvo.unique()) == {0, 1}
    # Distribuicao conhecida da base: 357 benignos, 212 malignos.
    assert (alvo == 0).sum() == 357
    assert (alvo == 1).sum() == 212


def test_clean_rejeita_valores_de_alvo_desconhecidos():
    raw = load_raw()
    raw.loc[0, config.TARGET_COLUMN] = "X"
    with pytest.raises(DatasetValidationError, match="Valores inesperados"):
        clean(raw)


def test_clean_rejeita_base_com_numero_de_amostras_errado():
    raw = load_raw().head(10)
    with pytest.raises(DatasetValidationError, match="amostras"):
        clean(raw)


def test_split_features_target_separa_alvo_das_features():
    X, y = split_features_target(clean(load_raw()))
    assert X.shape == (config.EXPECTED_N_SAMPLES, config.EXPECTED_N_FEATURES)
    assert y.name == config.TARGET_COLUMN
    assert config.TARGET_COLUMN not in X.columns
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def test_load_dataset_devolve_X_e_y_prontos():
    X, y = load_dataset()
    assert len(X) == len(y) == config.EXPECTED_N_SAMPLES
    assert set(y.unique()) == {0, 1}
