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


# ---------------------------------------------------------------------------
# Provedor do Claude: custo travado por construcao
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402


class ClienteFalso:
    """Imita `anthropic.Anthropic().messages`, sem rede e sem chave."""

    def __init__(self, texto="texto gerado", entrada=500, saida=300, parada="end_turn"):
        self.pedidos = []
        self._resposta = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=texto),
            ],
            usage=SimpleNamespace(input_tokens=entrada, output_tokens=saida),
            stop_reason=parada,
        )
        self.messages = self

    def create(self, **pedido):
        self.pedidos.append(pedido)
        return self._resposta


def _claude(cliente, **extras):
    return prov.ClaudeProvider(cliente, **extras)


def test_provedor_do_claude_atende_ao_protocolo():
    assert isinstance(_claude(ClienteFalso()), prov.Provider)


def test_pedido_usa_o_modelo_configurado_com_esforco_baixo_e_sem_temperatura():
    """Haiku 5.5 recusa `temperature` com 400; pensamento e cobrado como saida."""
    cliente = ClienteFalso()

    _claude(cliente).generate("prompt", system="sistema", seed=1)

    pedido = cliente.pedidos[0]
    assert pedido["model"] == config.LLM_MODELO
    assert pedido["system"] == "sistema"
    assert pedido["messages"] == [{"role": "user", "content": "prompt"}]
    assert pedido["output_config"] == {"effort": config.CLAUDE_ESFORCO}
    assert pedido["max_tokens"] == config.CLAUDE_MAX_TOKENS
    assert not {"temperature", "top_p", "top_k"} & set(pedido)


def test_devolve_so_o_texto_e_ignora_o_bloco_de_pensamento():
    assert _claude(ClienteFalso(texto="  resposta  ")).generate("p", system="s", seed=1) == (
        "resposta"
    )


def test_custo_acumulado_sai_do_uso_informado_pela_api():
    """O numero do relatorio vem da resposta, nao de estimativa."""
    claude = _claude(ClienteFalso(entrada=1_000_000, saida=1_000_000))

    claude.generate("p", system="s", seed=1)

    esperado = config.CLAUDE_PRECO_ENTRADA_USD + config.CLAUDE_PRECO_SAIDA_USD
    assert claude.gasto_usd == pytest.approx(esperado)
    assert claude.ultimo_uso["tokens_entrada"] == 1_000_000
    assert claude.ultimo_uso["custo_usd"] == pytest.approx(esperado)


def test_chamada_que_poderia_estourar_o_orcamento_nao_sai():
    """A trava age antes da chamada: o pior caso tem de caber no que sobra."""
    cliente = ClienteFalso()
    claude = _claude(cliente, orcamento_usd=0.0000001)

    with pytest.raises(prov.OrcamentoEsgotado):
        claude.generate("p", system="s", seed=1)

    assert cliente.pedidos == []


def test_pior_caso_de_uma_chamada_cabe_com_folga_no_orcamento_padrao():
    """Uma chamada sozinha, no maximo de tokens, custa uma fracao minima do teto."""
    claude = _claude(ClienteFalso())

    assert claude.pior_caso("x" * 4000, "y" * 2000) < config.LLM_ORCAMENTO_USD / 500


@pytest.mark.parametrize("parada", ["max_tokens", "refusal"])
def test_resposta_incompleta_levanta_em_vez_de_ir_para_o_cache(parada, tmp_path):
    """Texto cortado no meio, gravado no cache, seria servido para sempre."""
    claude = _claude(ClienteFalso(parada=parada))
    cache = prov.CachedProvider(claude, tmp_path, modelo="teste")

    with pytest.raises(prov.GeracaoIncompleta):
        cache.generate("p", system="s", seed=1)

    assert not list(tmp_path.glob("*.json"))
    # Foi cobrada mesmo assim, e o gasto tem de registrar isso.
    assert claude.gasto_usd > 0


def test_cache_registra_o_uso_e_nao_cobra_de_novo(tmp_path):
    """Rodar o lote duas vezes custa uma vez so; o custo fica gravado no artefato."""
    cliente = ClienteFalso()
    cache = prov.CachedProvider(_claude(cliente), tmp_path, modelo="teste")

    cache.generate("p", system="s", seed=1)
    cache.generate("p", system="s", seed=1)

    assert len(cliente.pedidos) == 1
    total = prov.custo_do_cache(tmp_path)
    assert total["textos"] == 1
    assert total["custo_usd"] > 0
    assert total["tokens_saida"] == 300


def test_custo_do_cache_ignora_entradas_sem_uso(falso, tmp_path):
    """Entradas gravadas por provedor sem custo (falso, Ollama) somam zero."""
    prov.CachedProvider(falso, tmp_path, modelo="teste").generate("p", system="s", seed=1)

    assert prov.custo_do_cache(tmp_path)["custo_usd"] == 0
