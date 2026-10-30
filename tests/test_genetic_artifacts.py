"""Testes da agregacao dos resultados em artefatos do relatorio."""

import pytest

from src.genetic import artifacts


def test_custo_clinico_ignora_o_termo_de_parcimonia(payload_ag):
    """Comparar bracos por fitness mistura erro clinico com numero de medidas.

    O braco completo vence no fitness por usar menos features, nao por errar
    menos. Sem separar os dois termos, o relatorio creditaria ao algoritmo um
    ganho que o limiar entrega sozinho.
    """
    com_poucas = payload_ag(fn=2, fp=16, features=("a",))
    com_todas = payload_ag(fn=2, fp=16, features=tuple("abcdefghij"))

    assert artifacts.custo_clinico(com_poucas) == artifacts.custo_clinico(com_todas)
    assert artifacts.custo_clinico(payload_ag(fn=2, fp=16)) == pytest.approx(36 / 455)


def test_agregacao_entre_sementes_traz_media_e_desvio(payload_ag):
    """Execucao unica nao e resultado: o relatorio reporta media e desvio."""
    payloads = [
        payload_ag(semente=42, fitness=0.90),
        payload_ag(semente=43, fitness=0.92),
        payload_ag(semente=44, fitness=0.94),
    ]

    linhas = artifacts.agregar_por_experimento(payloads)

    assert len(linhas) == 1
    linha = linhas[0]
    assert linha["experimento"] == "E1 base"
    assert linha["sementes"] == 3
    assert linha["fitness_medio"] == pytest.approx(0.92)
    assert linha["fitness_desvio"] > 0


def test_agregacao_separa_experimentos_e_familias(payload_ag):
    payloads = [
        payload_ag(experimento="E1 base", familia="KNN"),
        payload_ag(experimento="E1 base", familia="Regressão Logística"),
        payload_ag(experimento="E2 exploração", familia="KNN"),
    ]

    linhas = artifacts.agregar_por_experimento(payloads)

    assert len(linhas) == 3


def test_agregacao_separa_bracos_dentro_do_mesmo_experimento(payload_ag):
    """Os bracos de ablacao rodam com o nome e a familia de E1.

    Sem separar por braco, as 12 execucoes de E1 na regressao logistica — tres do
    braco completo e nove das ablacoes — virariam um grupo so, e a tabela
    reportaria "12 sementes" para uma configuracao que rodou com tres.
    """
    payloads = [
        payload_ag(braco="AG, completo", semente=42),
        payload_ag(braco="AG, completo", semente=43),
        payload_ag(braco="AG, só limiar", semente=42, segmentos=("limiar",)),
    ]

    linhas = artifacts.agregar_por_experimento(payloads)

    assert len(linhas) == 2
    por_braco = {linha["braco"]: linha for linha in linhas}
    assert por_braco["AG, completo"]["sementes"] == 2
    assert por_braco["AG, só limiar"]["sementes"] == 1


def test_tabela_de_bracos_compara_so_onde_as_ablacoes_rodaram(payload_ag):
    """Comparar 3 execucoes de um braco com 36 de outro nao e comparacao.

    As ablacoes rodam apenas em E1 sobre a regressao logistica. O braco completo
    roda nos quatro experimentos e nas tres familias, e incluir tudo na mesma
    tabela confrontaria conjuntos de tamanhos e condicoes diferentes.
    """
    payloads = [
        payload_ag(braco="AG, só limiar", experimento="E1 base", segmentos=("limiar",)),
        payload_ag(braco="AG, completo", experimento="E1 base"),
        payload_ag(braco="AG, completo", experimento="E2 exploração"),
        payload_ag(braco="AG, completo", experimento="E1 base", familia="KNN"),
    ]

    tabela = artifacts.tabela_de_bracos(payloads)

    por_braco = {linha["braco"]: linha for linha in tabela}
    assert set(por_braco) == {"AG, só limiar", "AG, completo"}
    assert por_braco["AG, completo"]["sementes"] == 1


def test_tabela_de_bracos_sem_ablacao_usa_tudo_que_recebeu(payload_ag):
    """Sem braco de ablacao nao ha recorte a inferir, e o smoke tem que funcionar."""
    payloads = [payload_ag(braco="AG, completo"), payload_ag(braco="AG, completo")]

    tabela = artifacts.tabela_de_bracos(payloads)

    assert len(tabela) == 1
    assert tabela[0]["sementes"] == 2


def test_tabela_de_bracos_ordena_pelo_menor_custo_clinico(payload_ag):
    """A pergunta da secao 4.6 e qual braco erra menos, nao qual tem maior fitness."""
    payloads = [
        payload_ag(braco="AG, completo", fn=2, fp=20, fitness=0.98),
        payload_ag(braco="AG, só limiar", fn=2, fp=16, fitness=0.94),
    ]

    tabela = artifacts.tabela_de_bracos(payloads)

    assert [linha["braco"] for linha in tabela] == ["AG, só limiar", "AG, completo"]
    assert tabela[0]["custo_medio"] < tabela[1]["custo_medio"]


def test_frequencia_de_features_cruza_com_o_ranking_shap(payload_ag):
    """Secao 4.9: se o AG reencontra as features do SHAP, e a quarta confirmacao."""
    shap_global = [
        {"feature": "texture_worst", "shap_medio": 1.25},
        {"feature": "area_worst", "shap_medio": 0.80},
        {"feature": "radius_mean", "shap_medio": 0.10},
    ]
    payloads = [
        payload_ag(semente=42, features=("texture_worst", "area_worst")),
        payload_ag(semente=43, features=("texture_worst",)),
        payload_ag(semente=44, features=("texture_worst", "radius_mean")),
    ]

    linhas = artifacts.frequencia_de_features(payloads, shap_global)

    primeira = linhas[0]
    assert primeira["feature"] == "texture_worst"
    assert primeira["selecoes"] == 3
    assert primeira["execucoes"] == 3
    assert primeira["posicao_shap"] == 1


def test_frequencia_ignora_execucoes_com_a_mascara_congelada(payload_ag):
    """Braco que nao otimiza features "seleciona" as 30 por construcao.

    Contar essas execucoes infla toda frequencia e faz o grafico dizer algo
    diferente do que o rotulo promete: nao e escolha do algoritmo, e o valor
    fixo do braco. O numero sairia plausivel e errado, sem erro nenhum.
    """
    buscadas = [
        payload_ag(semente=42, features=("texture_worst",)),
        payload_ag(semente=43, features=("texture_worst",)),
    ]
    congeladas = [
        payload_ag(
            semente=42,
            braco="AG, só limiar",
            segmentos=("limiar",),
            features=tuple(f"m{i}" for i in range(30)),
        ),
    ]

    linhas = artifacts.frequencia_de_features(buscadas + congeladas, shap_global=[])

    assert linhas[0]["feature"] == "texture_worst"
    assert linhas[0]["execucoes"] == 2
    assert linhas[0]["frequencia"] == pytest.approx(1.0)
    assert all(not linha["feature"].startswith("m") for linha in linhas)


def test_frequencia_sem_nenhuma_execucao_de_mascara_devolve_tabela_vazia(payload_ag):
    """Varredura so de limiar nao responde nada sobre selecao de features."""
    congeladas = [payload_ag(segmentos=("limiar",), features=("texture_worst",))]

    assert artifacts.frequencia_de_features(congeladas, shap_global=[]) == []


def test_feature_fora_do_ranking_shap_nao_quebra_a_tabela(payload_ag):
    """Base defensiva: `metrics.json` traz um recorte, nao as 30 medidas."""
    linhas = artifacts.frequencia_de_features(
        [payload_ag(features=("desconhecida",))],
        [{"feature": "texture_worst", "shap_medio": 1.25}],
    )

    assert linhas[0]["posicao_shap"] is None


def test_summary_reune_as_tres_tabelas(payload_ag):
    resumo = artifacts.montar_summary([payload_ag()], shap_global=[])

    assert set(resumo) == {"experimentos", "bracos", "features", "execucoes"}
    assert resumo["execucoes"] == 1


def _da_varredura(payload_ag, *, razao, alfa, semente=42, limiar=0.25, features=("a",), fn=2):
    """Payload de varredura: o da fabrica com a politica de custo registrada."""
    payload = payload_ag(semente=semente, features=features, fn=fn)
    payload["melhor"]["limiar"] = limiar
    payload["custo"] = {
        "falso_negativo": float(razao),
        "falso_positivo": 1.0,
        "parcimonia": alfa,
    }
    return payload


def test_varredura_agrega_uma_linha_por_politica_de_custo(payload_ag):
    """Sementes da mesma politica viram uma linha; politicas diferentes, nao."""
    payloads = [
        _da_varredura(payload_ag, razao=5, alfa=0.0, semente=s, limiar=limiar)
        for s, limiar in ((42, 0.30), (43, 0.40))
    ] + [_da_varredura(payload_ag, razao=20, alfa=0.0)]

    linhas = artifacts.tabela_da_varredura(payloads)

    assert [(l["razao_custo"], l["parcimonia"], l["sementes"]) for l in linhas] == [
        (5.0, 0.0, 2),
        (20.0, 0.0, 1),
    ]
    assert linhas[0]["limiar_medio"] == pytest.approx(0.35)


def test_varredura_ordena_pela_razao_dentro_de_cada_parcimonia(payload_ag):
    """A leitura da tabela e a curva: como a solucao se desloca quando a razao sobe."""
    payloads = [
        _da_varredura(payload_ag, razao=razao, alfa=alfa)
        for razao in (20, 5, 10)
        for alfa in (0.15, 0.0)
    ]

    chaves = [(l["parcimonia"], l["razao_custo"]) for l in artifacts.tabela_da_varredura(payloads)]

    assert chaves == sorted(chaves)


def test_varredura_nao_compara_custo_entre_politicas(payload_ag):
    """Cada politica mede o erro na propria moeda.

    Custo de 0,05 sob razao 5 e custo de 0,05 sob razao 20 nao sao o mesmo erro,
    e uma coluna de custo ou fitness convidaria a ordenar politicas por um numero
    que nao as compara. A tabela traz o que e comparavel: limiar, medidas, erros.
    """
    linha = artifacts.tabela_da_varredura([_da_varredura(payload_ag, razao=5, alfa=0.0)])[0]

    assert not any("custo_medio" in chave or "fitness" in chave for chave in linha)
    assert {"limiar_medio", "features_medio", "falsos_negativos_medio"} <= set(linha)


def test_varredura_sem_politica_registrada_falha_em_vez_de_assumir_uma(payload_ag):
    """Um default de 10:1 juntaria todas as execucoes numa politica so, sem erro."""
    with pytest.raises(KeyError):
        artifacts.tabela_da_varredura([payload_ag()])
