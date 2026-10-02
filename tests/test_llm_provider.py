"""Testes do provedor de modelo de linguagem, do cache e da chave."""

import pytest

from src import config
from src.llm import provider as prov


@pytest.fixture
def falso():
    return prov.FakeProvider(["primeira resposta", "segunda resposta"])


def test_config_publica_os_caminhos_e_limites_da_camada_de_llm():
    assert config.LLM_DIR.parent == config.RESULTS_DIR
    assert config.LLM_CACHE_DIR.parent == config.LLM_DIR
    assert config.LLM_MODELO
    # Temperatura zero e semente fixa: o texto publicado tem que ser o mesmo a
    # cada execucao, pela mesma regra que vale para results/metrics.json.
    assert config.LLM_TEMPERATURA == 0.0
    assert config.LLM_SEED == config.SEED
    assert config.LLM_LIMITE_DE_PALAVRAS > 0


def test_provedor_falso_atende_ao_protocolo(falso):
    """Todo o resto da camada depende so do Protocol, nunca de um cliente."""
    assert isinstance(falso, prov.Provider)

    texto = falso.generate("oi", system="voce e um assistente", seed=42)

    assert texto == "primeira resposta"


def test_cache_evita_a_segunda_chamada(falso, tmp_path):
    cache = prov.CachedProvider(falso, tmp_path, modelo="teste")

    primeira = cache.generate("mesmo prompt", system="s", seed=42)
    segunda = cache.generate("mesmo prompt", system="s", seed=42)

    assert primeira == segunda == "primeira resposta"
    assert falso.chamadas == 1


def test_mudar_o_prompt_muda_a_chave(falso, tmp_path):
    """Prompt novo nao pode reaproveitar resposta antiga em silencio."""
    cache = prov.CachedProvider(falso, tmp_path, modelo="teste")

    primeira = cache.generate("prompt A", system="s", seed=42)
    segunda = cache.generate("prompt B", system="s", seed=42)

    assert primeira != segunda
    assert falso.chamadas == 2


@pytest.mark.parametrize("mudanca", [{"system": "outro sistema"}, {"seed": 7}])
def test_qualquer_parametro_de_geracao_entra_na_chave(falso, tmp_path, mudanca):
    """Modelo, prompt de sistema, prompt do usuario e parametros, em conjunto."""
    cache = prov.CachedProvider(falso, tmp_path, modelo="teste")
    base = {"system": "s", "seed": 42}

    cache.generate("p", **base)
    cache.generate("p", **{**base, **mudanca})

    assert falso.chamadas == 2


def test_mudar_o_modelo_muda_a_chave(falso, tmp_path):
    prov.CachedProvider(falso, tmp_path, modelo="modelo A").generate("p", system="s", seed=42)
    prov.CachedProvider(falso, tmp_path, modelo="modelo B").generate("p", system="s", seed=42)

    assert falso.chamadas == 2


def test_falta_de_cache_sem_backend_nomeia_o_script_a_executar(tmp_path):
    """Erro explicito em vez de texto vazio: o container publicado nao tem LLM.

    Devolver string vazia aqui produziria um relatorio com um paragrafo em branco
    e nenhum erro — defeito silencioso, que e o que esta suite existe para
    impedir.
    """
    somente_cache = prov.CachedProvider(None, tmp_path, modelo="teste")

    with pytest.raises(prov.CacheVazioError, match="run_llm.py"):
        somente_cache.generate("p", system="s", seed=42)


def test_cache_sobrevive_a_troca_de_instancia(falso, tmp_path):
    """O cache e o artefato versionado que o container publicado consome."""
    prov.CachedProvider(falso, tmp_path, modelo="teste").generate("p", system="s", seed=42)

    somente_cache = prov.CachedProvider(None, tmp_path, modelo="teste")

    assert somente_cache.generate("p", system="s", seed=42) == "primeira resposta"
