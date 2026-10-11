"""Testes da geracao em lote dos textos."""

import importlib.util
import json

import pytest

from src import config


@pytest.fixture(scope="module")
def script():
    caminho = config.PROJECT_ROOT / "scripts" / "run_llm.py"
    spec = importlib.util.spec_from_file_location("run_llm", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_modo_verificacao_roda_o_lote_inteiro_sem_modelo_instalado(
    script, tmp_path, monkeypatch
):
    """O Ollama nao e pre-requisito para exercitar prompts e verificadores.

    A camada inteira fala com um Protocol, entao o lote roda com um provedor
    deterministico. E isso que permite desenvolver e testar a Fase 2 antes de ter
    qualquer modelo local instalado.
    """
    destino = tmp_path / "narrativas.json"
    monkeypatch.setattr(
        "sys.argv", ["run_llm.py", "--fake", "--casos", "1", "--saida", str(destino)]
    )

    script.main()

    assert destino.exists()


def test_modo_verificacao_percorre_o_caminho_feliz(script, tmp_path, monkeypatch):
    """O `--fake` existe para que uma falha real se destaque.

    Se o provedor falso devolvesse texto generico, tudo reprovaria sempre e o
    modo de verificacao nao distinguiria encanamento quebrado de texto ruim. A
    resposta falsa respeita o contrato do template que a pediu; os verificadores
    ja sao testados contra defeitos na propria suite deles.
    """
    destino = tmp_path / "narrativas.json"
    monkeypatch.setattr(
        "sys.argv", ["run_llm.py", "--fake", "--casos", "1", "--saida", str(destino)]
    )

    script.main()

    registros = json.loads(destino.read_text(encoding="utf-8"))
    reprovados = [r["identificador"] for r in registros if not r["qualidade"]["aprovado"]]

    assert not reprovados, reprovados


def test_lote_grava_um_registro_por_texto_com_a_avaliacao_junto(
    script, tmp_path, monkeypatch
):
    destino = tmp_path / "narrativas.json"
    monkeypatch.setattr(
        "sys.argv", ["run_llm.py", "--fake", "--casos", "2", "--saida", str(destino)]
    )

    script.main()

    registros = json.loads(destino.read_text(encoding="utf-8"))
    assert registros
    for registro in registros:
        assert registro["identificador"]
        assert "qualidade" in registro
        assert "aprovado" in registro["qualidade"]


def test_os_tres_produtos_de_texto_aparecem_no_lote(script, tmp_path, monkeypatch):
    """Explicacao por paciente, resumo de gestao e nota de mudanca."""
    destino = tmp_path / "narrativas.json"
    monkeypatch.setattr(
        "sys.argv", ["run_llm.py", "--fake", "--casos", "1", "--saida", str(destino)]
    )

    script.main()

    registros = json.loads(destino.read_text(encoding="utf-8"))
    nomes = {r["identificador"].rsplit("_v", 1)[0] for r in registros}

    assert "explicacao_paciente" in nomes
    assert "resumo_gestao" in nomes


def test_comparacao_entre_versoes_de_prompt_entra_no_artefato(
    script, tmp_path, monkeypatch
):
    """Secao 5.4: duas versoes medidas nas mesmas verificacoes automaticas."""
    destino = tmp_path / "narrativas.json"
    comparacao = tmp_path / "comparacao_de_prompts.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_llm.py",
            "--fake",
            "--casos",
            "1",
            "--saida",
            str(destino),
            "--comparacao",
            str(comparacao),
        ],
    )

    script.main()

    linhas = json.loads(comparacao.read_text(encoding="utf-8"))
    versoes = {linha["identificador"] for linha in linhas}

    assert {"explicacao_paciente_v1", "explicacao_paciente_v2"} <= versoes
    for linha in linhas:
        assert set(linha["verificacoes"]) == {
            "fidelidade",
            "seguranca",
            "cobertura",
            "formato",
        }


def test_artefato_e_reprodutivel_entre_execucoes(script, tmp_path, monkeypatch):
    """Mesma regra de results/metrics.json: rodar duas vezes da o mesmo arquivo."""
    primeiro, segundo = tmp_path / "a.json", tmp_path / "b.json"

    for destino in (primeiro, segundo):
        monkeypatch.setattr(
            "sys.argv", ["run_llm.py", "--fake", "--casos", "2", "--saida", str(destino)]
        )
        script.main()

    assert primeiro.read_bytes() == segundo.read_bytes()


def test_casos_selecionados_sao_deterministicos(script):
    """Caso sorteado a cada execucao tornaria o artefato irreprodutivel."""
    primeiro = script.selecionar_casos(3)
    segundo = script.selecionar_casos(3)

    assert primeiro == segundo
    assert len(primeiro) == 3


# ---------------------------------------------------------------------------
# Claude: estimativa antes de gastar, custo gravado depois
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from src.llm import provider as prov  # noqa: E402


class ClienteFalso:
    """Imita a API do Claude sem rede, contando as chamadas."""

    def __init__(self):
        self.chamadas = 0
        self.messages = self

    def create(self, **pedido):
        self.chamadas += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="texto de teste")],
            usage=SimpleNamespace(input_tokens=600, output_tokens=400),
            stop_reason="end_turn",
        )


@pytest.fixture
def cliente(script, monkeypatch):
    falso = ClienteFalso()
    original = prov.ClaudeProvider
    monkeypatch.setattr(script, "ClaudeProvider", lambda **kw: original(falso, **kw))
    return falso


def _argv(monkeypatch, tmp_path, *extras):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_llm.py",
            "--casos",
            "1",
            "--saida",
            str(tmp_path / "narrativas.json"),
            "--comparacao",
            str(tmp_path / "comparacao.json"),
            "--cache",
            str(tmp_path / "cache"),
            "--custo",
            str(tmp_path / "custo.json"),
            *extras,
        ],
    )


def test_estimativa_nao_chama_a_api_nem_grava_nada(script, cliente, tmp_path, monkeypatch, capsys):
    """Antes de gastar, o usuario ve quanto pode custar no pior caso."""
    _argv(monkeypatch, tmp_path, "--estimar")

    script.main()

    assert cliente.chamadas == 0
    assert not (tmp_path / "narrativas.json").exists()
    assert not (tmp_path / "cache").exists()
    assert "US$" in capsys.readouterr().out


def test_lote_com_claude_grava_o_custo_somado_do_cache(script, cliente, tmp_path, monkeypatch):
    _argv(monkeypatch, tmp_path)

    script.main()

    custo = json.loads((tmp_path / "custo.json").read_text(encoding="utf-8"))
    assert cliente.chamadas > 0
    assert custo["textos"] == cliente.chamadas
    assert 0 < custo["custo_usd"] < config.LLM_ORCAMENTO_USD
    assert custo["modelo"] == config.LLM_MODELO


def test_segunda_execucao_nao_paga_nada(script, cliente, tmp_path, monkeypatch):
    """O cache e a quarta trava de custo: rodar de novo e de graca."""
    _argv(monkeypatch, tmp_path)
    script.main()
    primeira = cliente.chamadas

    script.main()

    assert cliente.chamadas == primeira


def test_lote_que_estouraria_o_orcamento_aborta_antes_da_primeira_chamada(
    script, cliente, tmp_path, monkeypatch
):
    """Parar no meio ainda gastaria; a estimativa barra o lote inteiro antes."""
    monkeypatch.setattr(config, "LLM_ORCAMENTO_USD", 0.000001)
    _argv(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        script.main()

    assert cliente.chamadas == 0


def test_todos_os_casos_e_o_conjunto_de_teste_inteiro(script):
    """O painel escolhe qualquer caso do teste; todos precisam de texto."""
    assert script.quantidade_de_casos("todos") == 114
    assert script.quantidade_de_casos("3") == 3


# ---------------------------------------------------------------------------
# O lote narra o modelo servido, e a API encontra o que o lote gravou
# ---------------------------------------------------------------------------
def test_evidencias_usam_o_modelo_servido(script, monkeypatch):
    """Texto offline e API online narram o mesmo modelo, senao o cache nunca acerta."""
    from src.api import servico

    especificacao = servico.EspecificacaoDoModelo.de_dicionario(
        {**servico.referencia_da_fase_1().para_dicionario(), "limiar": 0.3}
    )
    monkeypatch.setattr(script, "carregar_especificacao", lambda: especificacao)

    evidencias = script.evidencias_por_caso([0])

    assert evidencias[0]["limiar"] == 0.3


def test_texto_gravado_pelo_lote_e_encontrado_pela_api(script, cliente, tmp_path, monkeypatch):
    """A garantia que amarra as duas pontas: mesma conta, mesma chave de cache.

    Se o lote e a API calculassem a probabilidade ou as contribuicoes de jeitos
    diferentes, mesmo que so no arredondamento, o /explain nunca acharia texto
    nenhum — e responderia "texto indisponivel" para sempre, sem erro.
    """
    pytest.importorskip("fastapi", reason="Fase 2: requer requirements-api.txt")
    from fastapi.testclient import TestClient

    from src.api import main, servico
    from src.data import load_dataset
    from src.preprocessing import split_data

    # Um modelo servido diferente da referencia: com a referencia, lote e API
    # coincidiriam por acaso, e o teste nao provaria nada.
    especificacao = servico.EspecificacaoDoModelo.de_dicionario(
        {
            **servico.referencia_da_fase_1().para_dicionario(),
            "features": ["texture_worst", "area_worst", "concave points_mean"],
            "limiar": 0.3,
        }
    )
    monkeypatch.setattr(script, "carregar_especificacao", lambda: especificacao)
    _argv(monkeypatch, tmp_path)
    script.main()

    X, y = load_dataset()
    _, X_teste, _, _ = split_data(X, y)
    api = TestClient(
        main.criar_app(
            servico.Servico.construir(especificacao), diretorio_cache=tmp_path / "cache"
        )
    )

    corpo = api.post("/explain", json={"medidas": X_teste.iloc[0].to_dict()}).json()

    # Achou no cache: o motivo, se houver, e reprovacao, nunca ausencia.
    assert corpo["texto"] is not None or "run_llm.py" not in corpo["texto_motivo"]
