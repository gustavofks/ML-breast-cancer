"""Carregamento, limpeza e validacao do dataset Breast Cancer Wisconsin.

Este modulo e a unica porta de entrada dos dados. Ele devolve um DataFrame
sempre no mesmo formato, ja validado, para que os modulos seguintes nao
precisem repetir verificacoes.

Armadilha conhecida do arquivo original: o cabecalho do CSV termina em virgula,
o que faz o pandas criar uma coluna extra `Unnamed: 32` inteiramente nula. Ela e
removida em `clean()`.
"""

from __future__ import annotations

import pandas as pd

from src import config


class DatasetValidationError(RuntimeError):
    """Erro levantado quando a base carregada nao tem o formato esperado."""


def load_raw(path=None) -> pd.DataFrame:
    """Le o CSV original, sem nenhuma transformacao.

    Args:
        path: caminho alternativo do arquivo. Usa `config.RAW_DATA_FILE` se omitido.

    Returns:
        DataFrame exatamente como esta no disco.
    """
    csv_path = path or config.RAW_DATA_FILE
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado em {csv_path}. "
            "Baixe o arquivo do Kaggle e salve em data/raw/data.csv."
        )
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas inuteis, codifica o alvo e valida o resultado.

    Passos:
        1. descarta colunas sem nome (`Unnamed: *`) e colunas inteiramente nulas;
        2. descarta a coluna `id`, que e identificador e nao tem valor preditivo;
        3. codifica `diagnosis` em 0 (benigno) e 1 (maligno);
        4. valida forma, ausencia de nulos e tipos.

    Args:
        df: DataFrame cru vindo de `load_raw()`.

    Returns:
        DataFrame com o alvo numerico na primeira coluna e 30 features numericas.

    Raises:
        DatasetValidationError: se a base nao corresponder ao esperado.
    """
    data = df.copy()

    unnamed = [c for c in data.columns if str(c).startswith("Unnamed")]
    all_null = [c for c in data.columns if data[c].isna().all()]
    data = data.drop(columns=set(unnamed) | set(all_null))

    if config.ID_COLUMN in data.columns:
        data = data.drop(columns=[config.ID_COLUMN])

    if config.TARGET_COLUMN not in data.columns:
        raise DatasetValidationError(
            f"Coluna alvo '{config.TARGET_COLUMN}' ausente na base."
        )

    unexpected = set(data[config.TARGET_COLUMN].unique()) - set(config.TARGET_MAPPING)
    if unexpected:
        raise DatasetValidationError(
            f"Valores inesperados em '{config.TARGET_COLUMN}': {sorted(unexpected)}. "
            f"Esperado: {sorted(config.TARGET_MAPPING)}."
        )
    data[config.TARGET_COLUMN] = data[config.TARGET_COLUMN].map(config.TARGET_MAPPING)

    _validate(data)
    return data


def _validate(data: pd.DataFrame) -> None:
    """Confere forma, nulos e tipos da base limpa."""
    n_features = data.shape[1] - 1

    if data.shape[0] != config.EXPECTED_N_SAMPLES:
        raise DatasetValidationError(
            f"Esperadas {config.EXPECTED_N_SAMPLES} amostras, encontradas {data.shape[0]}."
        )
    if n_features != config.EXPECTED_N_FEATURES:
        raise DatasetValidationError(
            f"Esperadas {config.EXPECTED_N_FEATURES} features, encontradas {n_features}."
        )

    missing = int(data.isna().sum().sum())
    if missing:
        raise DatasetValidationError(f"Base contem {missing} valores ausentes.")

    non_numeric = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]
    if non_numeric:
        raise DatasetValidationError(f"Colunas nao numericas apos a limpeza: {non_numeric}.")


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de features do vetor alvo."""
    y = data[config.TARGET_COLUMN]
    X = data.drop(columns=[config.TARGET_COLUMN])
    return X, y


def load_dataset(path=None) -> tuple[pd.DataFrame, pd.Series]:
    """Atalho: carrega, limpa e ja devolve `(X, y)` prontos para modelagem."""
    return split_features_target(clean(load_raw(path)))
