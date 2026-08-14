"""Testes das analises numericas da exploracao de dados."""

from src import eda
from src.data import load_dataset


def test_grupos_de_features_cobrem_as_trinta_colunas():
    X, _ = load_dataset()
    grupos = [eda.feature_group(X, sufixo) for sufixo in eda.FEATURE_GROUPS]

    assert [len(g) for g in grupos] == [10, 10, 10]
    assert sorted(sum(grupos, [])) == sorted(X.columns)


def test_separation_ranking_ordena_por_magnitude_do_efeito():
    X, y = load_dataset()
    ranking = eda.separation_ranking(X, y)

    assert len(ranking) == len(X.columns)
    assert ranking["cohens_d"].abs().is_monotonic_decreasing
    # A feature mais discriminante da base e conhecida da literatura.
    assert ranking.iloc[0]["feature"] == "concave points_worst"


def test_pares_altamente_correlacionados_nao_se_repetem():
    X, _ = load_dataset()
    pares = eda.highly_correlated_pairs(X, threshold=0.9)

    assert not pares.empty
    assert (pares["correlacao"].abs() > 0.9).all()
    # Cada par aparece uma unica vez, em uma unica ordem.
    combinacoes = {frozenset((a, b)) for a, b in zip(pares["feature_a"], pares["feature_b"])}
    assert len(combinacoes) == len(pares)
    # Nenhuma feature correlacionada consigo mesma.
    assert (pares["feature_a"] != pares["feature_b"]).all()


def test_correlacao_com_alvo_fica_no_intervalo_valido():
    X, y = load_dataset()
    corr = eda.correlation_with_target(X, y)

    assert len(corr) == len(X.columns)
    assert corr["correlacao"].abs().le(1).all()
    assert corr["correlacao"].abs().is_monotonic_decreasing
