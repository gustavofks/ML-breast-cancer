"""Figuras do algoritmo genetico, no tema visual da Fase 1.

As figuras vao para `results/genetic/`, e nao para `results/figures/`, porque a
secao 4.8 do design mantem os artefatos do algoritmo genetico juntos. O tema e o
mesmo da Fase 1 (`src.plotting.apply_style`), para que o relatorio nao pareca
costurado de duas fontes.

As funcoes devolvem a figura em vez de grava-la. E isso que permite testar o
conteudo — numero de linhas, rotulos, escala — em vez de apenas verificar que um
arquivo nasceu, que e justamente o teste que a Fase 1 aprendeu a nao confiar
quando o Grad-CAM gerava imagem plausivel e errada.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src import config
from src.genetic.artifacts import custo_clinico, frequencia_de_features
from src.plotting import COLOR_BENIGNO, COLOR_MALIGNO, INK_MUTED, apply_style


def salvar(figura: plt.Figure, nome: str, destino: Path | None = None) -> Path:
    """Grava a figura em `results/genetic/` e devolve o caminho."""
    pasta = destino or config.GENETIC_DIR
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome
    figura.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(figura)
    return caminho


def figura_convergencia(payloads: list[dict[str, Any]]) -> plt.Figure:
    """Melhor fitness por geracao, uma linha por execucao.

    A curva e a diferenca entre afirmar que houve convergencia e mostrar que
    houve. Sem elitismo a linha cai, e e isso que o experimento E4 mede.
    """
    apply_style()
    figura, eixo = plt.subplots(figsize=(7.6, 4.6))

    for payload in payloads:
        historico = payload["historico"]
        geracoes = [registro["geracao"] for registro in historico]
        melhores = [registro["melhor_fitness"] for registro in historico]
        rotulo = f"{payload['experimento']} | semente {payload['semente']}"
        eixo.plot(geracoes, melhores, linewidth=1.2, label=rotulo)

    eixo.set_xlabel("Geração")
    eixo.set_ylabel("Melhor fitness da geração")
    eixo.set_title("Convergência por execução")
    eixo.legend(fontsize=7, loc="lower right")
    eixo.spines[["top", "right"]].set_visible(False)
    return figura


def figura_fronteira(payloads: list[dict[str, Any]]) -> plt.Figure:
    """Custo clinico contra numero de features, um ponto por execucao.

    E a figura que responde a pergunta clinica do projeto: existe subconjunto
    pequeno que preserva o recall? Cada ponto e uma solucao encontrada, e a
    fronteira inferior esquerda e o que o patologista gostaria de ver.
    """
    apply_style()
    figura, eixo = plt.subplots(figsize=(6.8, 4.8))

    features = [p["avaliacao"]["n_features"] for p in payloads]
    custos = [custo_clinico(p) for p in payloads]
    recalls = [p["avaliacao"]["recall"] for p in payloads]

    dispersao = eixo.scatter(
        features,
        custos,
        c=recalls,
        cmap="viridis",
        s=48,
        edgecolor=INK_MUTED,
        linewidth=0.4,
    )
    barra = figura.colorbar(dispersao, ax=eixo)
    barra.set_label("Recall (maligno)")

    eixo.set_xlabel("Número de features ativas")
    eixo.set_ylabel("Custo clínico por amostra")
    eixo.set_title("Fronteira entre custo clínico e número de medidas")
    eixo.spines[["top", "right"]].set_visible(False)
    return figura


def figura_frequencia(
    payloads: list[dict[str, Any]],
    shap_global: list[dict[str, Any]],
    top: int = 15,
) -> plt.Figure:
    """Frequencia de selecao de cada medida, destacando as do topo do SHAP.

    A Fase 1 encerrou a explicabilidade com tres tecnicas convergindo. Esta
    figura e o quarto caminho: barras em laranja sao medidas que tambem estao no
    topo do ranking SHAP, e coincidencia repetida entre metodos independentes e
    evidencia, nao sorte.
    """
    apply_style()
    linhas = list(reversed(frequencia_de_features(payloads, shap_global)[:top]))

    figura, eixo = plt.subplots(figsize=(7.2, max(3.2, 0.32 * len(linhas) + 1)))

    nomes = [linha["feature"] for linha in linhas]
    valores = [linha["frequencia"] for linha in linhas]
    # Laranja quando a medida esta entre as dez primeiras do SHAP: e a
    # convergencia entre os dois metodos que a secao 4.9 procura.
    cores = [
        COLOR_MALIGNO
        if linha["posicao_shap"] is not None and linha["posicao_shap"] <= 10
        else COLOR_BENIGNO
        for linha in linhas
    ]

    eixo.barh(nomes, valores, color=cores)
    eixo.set_xlabel("Fração das execuções que selecionaram a medida")
    eixo.set_xlim(0, 1.0)
    eixo.set_title("Frequência de seleção, destacando o topo do ranking SHAP")
    eixo.grid(axis="y", visible=False)
    eixo.spines[["top", "right"]].set_visible(False)
    return figura
