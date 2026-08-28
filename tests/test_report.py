"""Testes da geracao do relatorio HTML."""

import json
import re

import pytest

from src import config
from src.report import build_report

METRICAS_MINIMAS = {
    "dataset": {
        "nome": "Base de teste",
        "amostras": 569,
        "features": 30,
        "benignos": 357,
        "malignos": 212,
        "treino": 455,
        "teste": 114,
    },
    "modelo_escolhido": "Regressão Logística",
    "metrica_de_selecao": "f1",
    "validacao_cruzada": [
        {"modelo": "Regressão Logística", "accuracy": 0.97, "precision": 0.98, "recall": 0.95, "f1": 0.96, "roc_auc": 0.99},
        {"modelo": "KNN", "accuracy": 0.96, "precision": 0.98, "recall": 0.91, "f1": 0.95, "roc_auc": 0.98},
    ],
    "hiperparametros": [],
    "teste": [
        {
            "modelo": "Regressão Logística",
            "accuracy": 0.96,
            "precision": 0.97,
            "recall": 0.92,
            "f1": 0.95,
            "roc_auc": 0.99,
            "verdadeiros_negativos": 71,
            "falsos_positivos": 1,
            "falsos_negativos": 3,
            "verdadeiros_positivos": 39,
        }
    ],
    "limiares": [
        {"limiar": 0.30, "recall": 0.97, "precisao": 0.97, "f1": 0.97, "falsos_negativos": 1, "falsos_positivos": 1},
        {"limiar": 0.50, "recall": 0.92, "precisao": 0.97, "f1": 0.95, "falsos_negativos": 3, "falsos_positivos": 1},
    ],
    "explicabilidade": {
        "coeficientes": [{"feature": "texture_worst", "coeficiente": 1.43, "razao_de_chances": 4.2}],
        "permutacao": [{"feature": "texture_worst", "queda_media": 0.054, "desvio": 0.021}],
        "shap_global": [{"feature": "texture_worst", "shap_medio": 1.25}],
    },
}

METRICAS_VISAO = {
    "dataset": {
        "nome": "Base de imagens de teste",
        "classes": ["benign", "malignant", "normal"],
        "imagens": 780,
        "classe_positiva": "malignant",
        "por_classe": [
            {"classe": "benign", "imagens": 437, "proporcao": 0.5603},
            {"classe": "malignant", "imagens": 210, "proporcao": 0.2692},
            {"classe": "normal", "imagens": 133, "proporcao": 0.1705},
        ],
    },
    "duplicatas": {
        "imagens": 780,
        "grupos": 668,
        "grupos_com_repeticao": 95,
        "imagens_redundantes": 112,
        "grupos_com_rotulos_contraditorios": 7,
    },
    "particoes": [
        {"conjunto": "treino", "imagens": 556, "benign": 311, "malignant": 150, "normal": 95, "proporcao_positiva": 0.2698},
    ],
    "modelo_escolhido": "MobileNetV2 (ajuste fino)",
    "metrica_de_selecao": "recall_maligno",
    "treino": [
        {
            "modelo": "MobileNetV2 (ajuste fino)",
            "parametros": 2261827,
            "epocas_treinadas": 12,
            "melhor_epoca": 6,
            "val_accuracy": 0.8661,
            "val_loss": 0.4234,
            "segundos": 78.4,
        }
    ],
    "teste": [
        {
            "modelo": "MobileNetV2 (ajuste fino)",
            "accuracy": 0.8393,
            "f1": 0.8214,
            "recall_maligno": 0.9,
            "precisao_maligno": 0.7714,
            "malignos_no_teste": 30,
            "malignos_nao_detectados": 3,
        }
    ],
    "figuras": [
        {
            "arquivo": "25_gradcam_acertos.png",
            "titulo": "Grad-CAM — casos detectados",
            "leitura": "o calor se concentra sobre a lesão",
        }
    ],
}


@pytest.fixture
def pagina(tmp_path):
    """Relatório sem a entrega de imagens, para exercitar o estado pendente.

    O dicionário vazio é deliberado: `None` faria o gerador ler o
    `metrics_vision.json` do projeto, e o teste deixaria de ser isolado.
    """
    return build_report(
        METRICAS_MINIMAS, tmp_path / "index.html", metrics_vision={}
    ).read_text(encoding="utf-8")


def test_relatorio_traz_as_tres_abas(pagina):
    for identificador in ("wisconsin", "imagens", "comparacao"):
        assert f'id="painel-{identificador}"' in pagina
        assert f'id="aba-{identificador}"' in pagina

    assert pagina.count('role="tab"') == 3
    assert pagina.count('role="tabpanel"') == 3
    # Apenas a primeira aba nasce selecionada. O espaço antes do atributo evita
    # contar o seletor de mesmo nome que existe na folha de estilo.
    assert pagina.count(' aria-selected="true"') == 1


def test_aba_de_imagens_fica_pendente_sem_metricas(pagina):
    assert 'class="pendente"' in pagina


def test_relatorio_usa_os_numeros_recebidos(pagina):
    assert "Regressão Logística" in pagina
    assert "569" in pagina and "455" in pagina and "114" in pagina
    # Números formatados no padrão brasileiro.
    assert "0,920" in pagina


def test_cada_aba_tem_indice_com_uma_entrada_por_secao(pagina):
    secoes = re.findall(r'<section id="(wisconsin-\d+)"', pagina)
    elos = re.findall(r'<a href="#(wisconsin-\d+)"', pagina)

    assert secoes, "a aba tabular deve ter seções"
    assert secoes == elos, "o índice lateral deve espelhar as seções, na mesma ordem"


def test_tabelas_de_dados_ficam_recolhidas_sob_as_figuras(pagina):
    assert '<details class="dados">' in pagina
    assert "<summary>" in pagina


def test_relatorio_referencia_apenas_figuras_existentes(pagina):
    referencias = re.findall(r'<img src="\.\./figures/([^"]+)"', pagina)

    assert referencias, "o relatório deve exibir figuras"
    for arquivo in referencias:
        assert (config.FIGURES_DIR / arquivo).exists(), f"figura ausente: {arquivo}"


def test_relatorio_e_autocontido(pagina):
    """Sem CDN, fonte remota ou script externo: precisa abrir por duplo clique."""
    assert "<style>" in pagina and "<script>" in pagina
    assert "http://" not in pagina
    assert 'src="http' not in pagina
    assert "@import" not in pagina


def test_relatorio_tem_recursos_de_acessibilidade(pagina):
    assert 'class="pular"' in pagina, "link de pular para o conteúdo"
    assert 'role="tablist"' in pagina
    assert 'aria-controls="painel-wisconsin"' in pagina
    assert 'lang="pt-BR"' in pagina
    assert "prefers-reduced-motion" in pagina


def test_relatorio_tem_alternador_de_tema(pagina):
    assert 'class="tema"' in pagina
    # Dois estados, cada um com seu ícone em SVG (nunca emoji).
    for estado in ("claro", "escuro"):
        assert f'data-icone="{estado}"' in pagina
    assert 'data-icone="sistema"' not in pagina, "o estado sistema foi removido"

    # O tema escolhido pelo leitor precisa vencer a preferência do sistema.
    assert ':root[data-tema="escuro"]' in pagina
    assert ":root:not([data-tema])" in pagina

    # Script no head aplica o tema antes da primeira pintura, e sem escolha
    # salva ele segue a preferência do sistema — não força claro.
    inicio_do_corpo = pagina.index("<body>")
    cabecalho = pagina[:inicio_do_corpo]
    assert "localStorage.getItem('tema')" in cabecalho
    assert "prefers-color-scheme: dark" in cabecalho


def test_alternador_de_tema_fica_no_topo_direito_da_pagina(pagina):
    # Dentro do envelope e antes da capa, mas fora da barra de abas.
    assert pagina.index('class="envelope"') < pagina.index('class="tema"')
    assert pagina.index('class="tema"') < pagina.index('class="capa"')
    barra = pagina[pagina.index('class="barra-abas"') :]
    assert 'class="tema"' not in barra[: barra.index("</div>")]

    # Ancorado no envelope, não na janela: não acompanha a rolagem.
    assert "position: relative; max-width: var(--largura)" in pagina
    regra = pagina[pagina.index(".tema {") : pagina.index(".tema:hover")]
    assert "position: absolute" in regra
    assert "position: fixed" not in regra


def test_troca_de_aba_sobe_ate_a_barra_de_abas(pagina):
    # A âncora existe para medir a posição real da barra, que é sticky.
    assert 'id="ancora-abas"' in pagina
    assert pagina.index('id="ancora-abas"') < pagina.index('class="barra-abas"')

    # Sobe até o início da aba, nunca até o topo do documento.
    assert "window.scrollTo({ top: 0" not in pagina
    assert "Math.min(topoDasAbas(), maximo)" in pagina


def test_relatorio_mostra_o_codigo_que_executa_cada_etapa(pagina):
    assert 'class="feito"' in pagina
    assert "Como foi feito" in pagina

    # Os trechos saem do arquivo-fonte, não de texto copiado para o relatório.
    assert "src/preprocessing.py · build_preprocessor() · build_pipeline() · split_data()" in pagina
    assert "src/evaluation.py · threshold_analysis()" in pagina
    assert "SimpleImputer" in pagina and "stratify=y" in pagina

    # Docstring fora: a prosa da seção já explica, repetir dobraria o trecho.
    assert "Encadeia o pre-processamento" not in pagina

    # O realce é feito na geração, sem biblioteca externa nem rede.
    assert '<span class="k">' in pagina
    assert "pre.codigo .s" in pagina


def test_relatorio_tem_o_painel_de_decisoes_com_contrafactual(pagina):
    assert 'class="decisoes"' in pagina
    assert "Decisões e o contrafactual" in pagina

    # Cada decisão traz o caminho oposto: é o que a justifica.
    assert pagina.count("Se fosse ao contrário") >= 6
    assert "Por quê" in pagina
    assert "vazamento" in pagina or "carregariam informação do teste" in pagina


def test_legenda_da_figura_ocupa_a_largura_toda(pagina):
    # Título e leitura em elementos separados, para o grid de duas colunas.
    assert 'class="titulo-figura"' in pagina
    assert 'class="leitura"' in pagina

    regra = pagina[pagina.index("figcaption {") : pagina.index("figcaption .titulo-figura")]
    assert "display: grid" in regra
    assert "max-width: none" in regra, "a legenda não pode ter medida própria"

    # A coluna de conteúdo é que define a medida: texto, figura e tabela
    # terminam na mesma margem.
    assert "max-width: var(--medida)" in pagina
    assert "p { max-width: none;" in pagina


def test_aba_de_imagens_e_montada_a_partir_das_metricas(tmp_path):
    pagina = build_report(
        METRICAS_MINIMAS, tmp_path / "index.html", metrics_vision=METRICAS_VISAO
    ).read_text(encoding="utf-8")

    assert 'class="pendente"' not in pagina
    assert "MobileNetV2" in pagina
    assert "780" in pagina and "malignant" in pagina
    # A auditoria de duplicatas precisa aparecer: é o que sustenta as métricas.
    assert "112" in pagina and "duplicatas" in pagina.lower()
    # A comparação entre as duas entregas usa o recall de cada uma.
    assert "0,900" in pagina


def test_relatorio_le_metrics_json_por_padrao(tmp_path):
    metricas = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
    destino = build_report(metricas, tmp_path / "index.html")

    assert destino.exists()
    assert metricas["modelo_escolhido"] in destino.read_text(encoding="utf-8")
