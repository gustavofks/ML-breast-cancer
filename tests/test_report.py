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
    "explicabilidade": {"coeficientes": [{"feature": "texture_worst", "coeficiente": 1.43, "razao_de_chances": 4.2}]},
}


@pytest.fixture
def pagina(tmp_path):
    destino = build_report(METRICAS_MINIMAS, tmp_path / "index.html")
    return destino.read_text(encoding="utf-8")


def test_relatorio_traz_as_duas_abas(pagina):
    assert 'id="painel-wisconsin"' in pagina
    assert 'id="painel-extra"' in pagina
    # A aba do extra nasce oculta e sinalizada como pendente.
    assert 'id="painel-extra" role="tabpanel" hidden' in pagina
    assert "pendente" in pagina


def test_relatorio_usa_os_numeros_recebidos(pagina):
    assert "Regressão Logística" in pagina
    assert "569" in pagina and "455" in pagina and "114" in pagina
    # Numeros formatados no padrao brasileiro.
    assert "0,920" in pagina


def test_relatorio_referencia_apenas_figuras_existentes(pagina):
    referencias = re.findall(r'<img src="\.\./figures/([^"]+)"', pagina)

    assert referencias, "o relatório deve exibir figuras"
    for arquivo in referencias:
        assert (config.FIGURES_DIR / arquivo).exists(), f"figura ausente: {arquivo}"


def test_relatorio_e_autocontido(pagina):
    """Sem CDN, fonte remota ou script externo: precisa abrir por duplo clique."""
    assert "<style>" in pagina and "<script>" in pagina
    assert "http://" not in pagina
    assert "src=\"http" not in pagina
    assert "@import" not in pagina


def test_aba_extra_e_montada_a_partir_das_metricas_de_imagem(tmp_path):
    """Quando `metrics_vision` existe, a aba deixa de ser um aviso de pendência."""
    metricas_visao = {
        "dataset": {
            "classes": ["benign", "malignant"],
            "imagens": 780,
            "classe_positiva": "malignant",
            "por_classe": [
                {"classe": "benign", "imagens": 437, "proporcao": 0.5603},
                {"classe": "malignant", "imagens": 343, "proporcao": 0.4397},
            ],
        },
        "modelo_escolhido": "MobileNetV2 (transferência)",
        "teste": [
            {"modelo": "MobileNetV2 (transferência)", "accuracy": 0.88, "precision": 0.86, "recall": 0.84, "f1": 0.85}
        ],
        "figuras": [],
    }

    pagina = build_report(
        METRICAS_MINIMAS, tmp_path / "index.html", metrics_vision=metricas_visao
    ).read_text(encoding="utf-8")

    assert '<div class="pendente">' not in pagina
    assert "MobileNetV2" in pagina
    assert "780" in pagina and "malignant" in pagina


def test_relatorio_le_metrics_json_por_padrao(tmp_path):
    metricas = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
    destino = build_report(metricas, tmp_path / "index.html")

    assert destino.exists()
    assert metricas["modelo_escolhido"] in destino.read_text(encoding="utf-8")
