"""Testes do painel: logica pura primeiro, a tela do Gradio por ultimo."""

import matplotlib

matplotlib.use("Agg")  # backend sem janela: o teste roda em terminal

import pytest  # noqa: E402

from src import config  # noqa: E402
from src.plotting import COLOR_BENIGNO, COLOR_MALIGNO  # noqa: E402
from src.ui import app as painel  # noqa: E402


def test_casos_de_teste_sao_o_conjunto_de_teste_com_o_rotulo_registrado():
    """O painel mostra acerto e erro; para isso precisa do diagnostico registrado."""
    casos = painel.casos_de_teste()

    assert len(casos) == 114
    assert set(casos["diagnostico_registrado"]) == set(config.CLASS_NAMES)
    assert len(casos.columns) == config.EXPECTED_N_FEATURES + 1


def test_rotulo_do_caso_nomeia_indice_e_diagnostico_registrado():
    casos = painel.casos_de_teste()

    rotulo = painel.rotulo_do_caso(casos, 0)

    assert rotulo.startswith("Caso 0")
    assert casos["diagnostico_registrado"].iloc[0] in rotulo


CONTRIBUICOES = [
    {"feature": f"medida_{i}", "contribuicao": valor}
    for i, valor in enumerate([2.0, -1.5, 1.0, -0.8, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1, 0.05, -0.01])
]


def test_grafico_mostra_as_dez_maiores_contribuicoes():
    figura = painel.figura_de_contribuicoes(CONTRIBUICOES)
    barras = figura.axes[0].patches

    assert len(barras) == 10


def test_maior_contribuicao_fica_no_topo_do_grafico():
    """barh desenha de baixo para cima; sem inverter, a mais importante fica escondida no pe."""
    figura = painel.figura_de_contribuicoes(CONTRIBUICOES)
    rotulos = [t.get_text() for t in figura.axes[0].get_yticklabels()]

    assert rotulos[-1] == "medida_0"


def test_cor_da_barra_segue_o_sinal_da_contribuicao():
    """Laranja empurra para maligno, azul para benigno: as cores da Fase 1."""
    figura = painel.figura_de_contribuicoes(CONTRIBUICOES[:2])
    cores = {
        round(b.get_width(), 6): matplotlib.colors.to_hex(b.get_facecolor())
        for b in figura.axes[0].patches
    }

    assert cores[2.0] == matplotlib.colors.to_hex(COLOR_MALIGNO)
    assert cores[-1.5] == matplotlib.colors.to_hex(COLOR_BENIGNO)


# ---------------------------------------------------------------------------
# Analise de um caso, contra a API real
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def api():
    """A API de verdade, servida em processo: o painel fala com ela por HTTP."""
    pytest.importorskip("fastapi", reason="Fase 2: requer requirements-api.txt")
    from fastapi.testclient import TestClient

    from src.api import main, servico

    modelo = servico.Servico.construir(servico.referencia_da_fase_1())
    return TestClient(main.criar_app(modelo, diretorio_cache=config.LLM_CACHE_DIR / "_vazio"))


@pytest.fixture(scope="module")
def casos():
    return painel.casos_de_teste()


def test_analise_mostra_o_que_a_api_respondeu(api, casos):
    """O painel nao recalcula nada: o numero da tela e o numero da API."""
    resumo, figura, _ = painel.analisar(api, casos, 0)

    medidas = casos.drop(columns="diagnostico_registrado").iloc[0].to_dict()
    resposta = api.post("/explain", json={"medidas": medidas}).json()

    assert f"{resposta['probabilidade']:.1%}".replace(".", ",") in resumo
    assert resposta["classe"] in resumo
    assert casos["diagnostico_registrado"].iloc[0] in resumo
    assert len(figura.axes[0].patches) == painel.TOP_CONTRIBUICOES


def test_sem_texto_gerado_o_painel_diz_por_que(api, casos):
    """Paragrafo em branco pareceria defeito; o motivo diz que e esperado."""
    _, _, texto = painel.analisar(api, casos, 0)

    assert "run_llm.py" in texto


def test_caso_com_medida_fora_da_distribuicao_mostra_o_alerta(api, casos):
    deslocados = casos.copy()
    deslocados.loc[0, "area_worst"] = 1e6

    resumo, _, _ = painel.analisar(api, deslocados, 0)

    assert "area_worst" in resumo
    assert "fora da distribuição" in resumo


def test_erro_da_api_vira_mensagem_e_nao_excecao(casos):
    """Tela quebrada no meio da demonstracao e pior que uma mensagem clara."""

    class ApiFora:
        def post(self, *args, **kwargs):
            raise ConnectionError("API indisponivel")

    resumo, figura, texto = painel.analisar(ApiFora(), casos, 0)

    assert "indisponível" in resumo
    assert figura is None


def test_analisar_muitos_casos_nao_acumula_figuras_abertas(api, casos):
    """Painel em uso continuo: figura esquecida aberta e vazamento de memoria."""
    import matplotlib.pyplot as plt

    antes = len(plt.get_fignums())
    for indice in range(5):
        painel.analisar(api, casos, indice)

    assert len(plt.get_fignums()) == antes


# ---------------------------------------------------------------------------
# A tela e o servidor unico
# ---------------------------------------------------------------------------
def test_aviso_de_apoio_a_decisao_fica_fixo_na_tela(api):
    """Secao 7: aviso permanente, fora do texto gerado, que pode faltar."""
    pytest.importorskip("gradio", reason="Fase 2: requer requirements-api.txt")
    import gradio as gr

    tela = painel.criar_painel(api)

    textos = [b.value for b in tela.blocks.values() if isinstance(b, gr.Markdown)]
    assert any("diagnóstico final é sempre do médico" in (t or "") for t in textos)


def test_painel_e_api_no_mesmo_servidor(api):
    """Um container, uma porta: o que o Hugging Face Spaces exige."""
    pytest.importorskip("gradio", reason="Fase 2: requer requirements-api.txt")
    from fastapi.testclient import TestClient

    from src.api import servico

    modelo = servico.Servico.construir(servico.referencia_da_fase_1())
    cliente = TestClient(painel.criar_app_com_painel(modelo, url_da_api="http://teste"))

    assert cliente.get("/health").status_code == 200
    assert cliente.get("/painel/").status_code == 200


def test_raiz_serve_a_vitrine(api):
    """Quem abre o endereco do servidor cai na vitrine, nao num JSON."""
    pytest.importorskip("gradio", reason="Fase 2: requer requirements-api.txt")
    from fastapi.testclient import TestClient

    from src.api import servico

    modelo = servico.Servico.construir(servico.referencia_da_fase_1())
    cliente = TestClient(painel.criar_app_com_painel(modelo, url_da_api="http://teste"))

    resposta = cliente.get("/", follow_redirects=False)

    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/html")
    assert "Console da API" in resposta.text


def test_painel_aponta_para_a_porta_do_proprio_processo(monkeypatch):
    """No Spaces a porta e 7860; localmente, 8000. Vem do ambiente."""
    monkeypatch.setenv("PORT", "7860")

    assert painel.url_local_da_api() == "http://127.0.0.1:7860"


def test_metricas_separam_o_painel_da_raiz(api):
    """O Gradio montado sobrescreve a rota do escopo com a dele, que e "/".

    Sem o prefixo da montagem, cada pagina do painel seria contada como a raiz,
    e a raiz apareceria com duas visitas em vez de uma.
    """
    pytest.importorskip("gradio", reason="Fase 2: requer requirements-api.txt")
    from fastapi.testclient import TestClient

    from src.api import servico

    modelo = servico.Servico.construir(servico.referencia_da_fase_1())
    cliente = TestClient(painel.criar_app_com_painel(modelo, url_da_api="http://teste"))

    cliente.get("/painel/")
    cliente.get("/", follow_redirects=False)
    texto = cliente.get("/metrics").text

    assert 'api_requisicoes_total{rota="/painel/",status="200"} 1' in texto
    assert 'api_requisicoes_total{rota="/",status="200"} 1' in texto
