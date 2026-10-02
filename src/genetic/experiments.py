"""Configuracoes exigidas pelo enunciado, e os bracos que as tornam legiveis.

Sao quatro experimentos, e nao tres, porque o quarto responde uma pergunta que
os outros nao respondem: quanto o elitismo contribui de fato. E4 e identico a E1
em tudo menos no elitismo, entao a diferenca observada tem uma causa so.
"""

from __future__ import annotations

from itertools import product

from src import config
from src.genetic.chromosome import SEGMENTOS
from src.genetic.engine import ConfiguracaoAG
from src.genetic.fitness import CustoClinico

SEMENTES = config.GENETIC_SEEDS

FAMILIAS = ("Regressão Logística", "KNN", "Random Forest")

EXPERIMENTOS: tuple[ConfiguracaoAG, ...] = (
    ConfiguracaoAG(
        nome="E1 base",
        populacao=40,
        geracoes=30,
        taxa_mutacao=0.10,
        taxa_cruzamento=0.80,
        torneio=3,
        elitismo=2,
    ),
    # Populacao grande, mutacao alta, pressao baixa: diversidade evita o otimo local?
    ConfiguracaoAG(
        nome="E2 exploração",
        populacao=80,
        geracoes=30,
        taxa_mutacao=0.25,
        taxa_cruzamento=0.60,
        torneio=2,
        elitismo=2,
    ),
    # Populacao pequena, mutacao baixa, pressao alta: converge mais rapido?
    ConfiguracaoAG(
        nome="E3 explotação",
        populacao=20,
        geracoes=60,
        taxa_mutacao=0.05,
        taxa_cruzamento=0.90,
        torneio=5,
        elitismo=2,
    ),
    ConfiguracaoAG(
        nome="E4 sem elitismo",
        populacao=40,
        geracoes=30,
        taxa_mutacao=0.10,
        taxa_cruzamento=0.80,
        torneio=3,
        elitismo=0,
    ),
)

# Bracos de ablacao da secao 4.6. Afirmar que "o AG achou solucao melhor" nao
# prova nada se o ganho vier inteiro do limiar; cada braco isola um segmento.
BRACOS: dict[str, frozenset[str]] = {
    "AG, só limiar": frozenset({"limiar"}),
    "AG, só features": frozenset({"mascara"}),
    "AG, só hiperparâmetros": frozenset({"hiperparametros"}),
    "AG, completo": SEGMENTOS,
}

# Os bracos rodam apenas sobre a regressao logistica, o modelo escolhido na
# Fase 1: a pergunta e sobre a contribuicao de cada segmento, nao sobre a
# familia. As quatro configuracoes e que rodam sobre as tres familias.
FAMILIA_DOS_BRACOS = "Regressão Logística"

# Varredura de sensibilidade da secao 4.5.1. Analise separada, que nao entra na
# contagem de experimentos do enunciado: a razao de custo e decisao de politica
# clinica, e a varredura entrega a curva que torna essa decisao informada.
VARREDURA_CUSTO: tuple[CustoClinico, ...] = tuple(
    CustoClinico(
        falso_negativo=float(razao) * config.CUSTO_FALSO_POSITIVO,
        falso_positivo=config.CUSTO_FALSO_POSITIVO,
        parcimonia=alfa,
    )
    for razao, alfa in product((5, 10, 20), (0.0, 0.05, 0.15))
)

CONFIGURACAO_DA_VARREDURA = EXPERIMENTOS[0]
