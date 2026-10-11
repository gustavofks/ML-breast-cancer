"""Painel Gradio: escolher um caso, ver a previsao, a explicacao e o texto.

O painel **consome a API por HTTP** e nao reimplementa nada: previsao, limiar,
contribuicoes e texto chegam todos de `/explain`. E o que a secao 7 do design
exige, e o que garante que a tela mostra exatamente o que a API publicada
responde.

A logica fica em funcoes puras, que recebem o cliente HTTP como argumento. O
Gradio so monta a tela, em `criar_painel`, e e importado ali dentro: os testes da
logica rodam sem ele, contra a API real, pelo `TestClient` da FastAPI, que e um
cliente httpx como o de producao.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.data import load_dataset
from src.plotting import COLOR_BENIGNO, COLOR_MALIGNO, apply_style
from src.preprocessing import split_data

# Aviso permanente da secao 7. Fica fixo na tela, e nao dentro do texto gerado:
# o texto pode faltar, o aviso nao.
AVISO = (
    "**Sistema de apoio à decisão.** A previsão e a explicação ajudam a leitura do "
    "caso, mas o diagnóstico final é sempre do médico responsável."
)

# Quantas contribuicoes o grafico mostra. Dez cabem legiveis numa tela, e as
# demais pesam pouco no caso tipico.
TOP_CONTRIBUICOES = 10


def casos_de_teste() -> pd.DataFrame:
    """O conjunto de teste, com o diagnostico registrado na base.

    Painel de demonstracao sobre casos que o modelo nunca viu. O diagnostico
    registrado vai junto para que o avaliador veja acerto **e** erro, inclusive
    falso negativo, em vez de uma vitrine que so mostra o que da certo.
    """
    X, y = load_dataset()
    _, X_teste, _, y_teste = split_data(X, y)
    casos = X_teste.reset_index(drop=True).copy()
    casos["diagnostico_registrado"] = [config.CLASS_NAMES[int(v)] for v in y_teste]
    return casos


def rotulo_do_caso(casos: pd.DataFrame, indice: int) -> str:
    return f"Caso {indice} — registrado como {casos['diagnostico_registrado'].iloc[indice]}"


def figura_de_contribuicoes(contribuicoes: list[dict[str, Any]]) -> plt.Figure:
    """As maiores contribuicoes do caso, com a mais importante no topo.

    Laranja empurra para maligno, azul para benigno: as mesmas cores da Fase 1,
    para que o grafico do painel e os do relatorio se leiam da mesma forma.
    """
    apply_style()
    maiores = sorted(contribuicoes, key=lambda c: abs(c["contribuicao"]), reverse=True)[
        :TOP_CONTRIBUICOES
    ]
    # barh desenha de baixo para cima: invertida, a maior fica no topo.
    maiores = list(reversed(maiores))

    figura, eixo = plt.subplots(figsize=(7.2, 0.42 * len(maiores) + 1.2))
    eixo.barh(
        [c["feature"] for c in maiores],
        [c["contribuicao"] for c in maiores],
        color=[COLOR_MALIGNO if c["contribuicao"] > 0 else COLOR_BENIGNO for c in maiores],
    )
    eixo.axvline(0, color="#6b6b68", linewidth=0.8)
    eixo.set_xlabel("Contribuição para a previsão (positiva empurra para maligno)")
    eixo.set_title("Medidas que mais pesaram neste caso")
    eixo.grid(axis="y", visible=False)
    eixo.spines[["top", "right"]].set_visible(False)
    figura.tight_layout()
    return figura


def _porcentagem(valor: float) -> str:
    """12,3% e nao 12.3%: o painel e lido em portugues."""
    return f"{valor:.1%}".replace(".", ",")


def analisar(api, casos: pd.DataFrame, indice: int):
    """Resumo, grafico e texto de um caso, a partir de `/explain`.

    `api` e qualquer cliente com `.post`, no padrao do httpx. Falha de rede vira
    mensagem na tela, e nao excecao: a tela quebrada no meio de uma
    demonstracao e pior que uma frase dizendo que a API esta fora. So erros de
    transporte sao tratados assim; erro de programacao continua levantando.
    """
    medidas = casos.drop(columns="diagnostico_registrado").iloc[indice].to_dict()
    try:
        resposta = api.post("/explain", json={"medidas": medidas})
    except (httpx.HTTPError, OSError) as erro:
        return f"**API indisponível:** {erro}", None, ""
    if resposta.status_code != 200:
        return f"**A API recusou o caso** ({resposta.status_code}): {resposta.text}", None, ""

    corpo = resposta.json()
    linhas = [
        f"### {rotulo_do_caso(casos, indice)}",
        f"**Previsão do modelo:** {corpo['classe']}",
        f"**Probabilidade de malignidade:** {_porcentagem(corpo['probabilidade'])} "
        f"(limiar de decisão: {_porcentagem(corpo['limiar'])})",
    ]
    if corpo["alertas"]:
        fora = ", ".join(
            f"{a['feature']} ({str(round(a['desvios'], 1)).replace('.', ',')} desvios padrão)"
            for a in corpo["alertas"]
        )
        linhas.append(
            f"**Atenção:** medidas fora da distribuição do treino — {fora}. "
            "Confira a coleta antes de confiar na previsão."
        )

    if corpo["texto"]:
        texto = corpo["texto"]
    else:
        texto = f"_Texto em linguagem natural indisponível: {corpo['texto_motivo']}._"

    figura = figura_de_contribuicoes(corpo["contribuicoes"])
    # Fechada no pyplot, mas ainda desenhavel: o Gradio a serializa com savefig.
    # Sem isso, cada caso analisado deixaria uma figura aberta para sempre.
    plt.close(figura)
    return "\n\n".join(linhas), figura, texto
