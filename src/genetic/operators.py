"""Selecao, cruzamento e mutacao, com um operador por tipo de gene.

O cruzamento da mascara e **uniforme**, e nao de ponto unico: as 30 medidas nao
tem relacao de vizinhanca entre si, entao cortar o vetor numa posicao criaria
vies posicional sem nenhuma justificativa no dominio.

A mutacao de gene real com `log=True` perturba o expoente, nao o valor. Somar
ruido gaussiano direto em `C` moveria 0,001 e 100 na mesma magnitude absoluta, o
que na pratica congelaria a ponta baixa do espaco.
"""

from __future__ import annotations

import numpy as np

from src.genetic.chromosome import (
    ESPACOS,
    SEGMENTOS,
    Gene,
    GeneInteiro,
    GeneReal,
    Individuo,
    reparar,
    sortear_gene,
)

# Desvio da perturbacao gaussiana, em fracao da amplitude do gene. Pequeno o
# bastante para refinar uma solucao boa, grande o bastante para sair de um
# plato local.
DESVIO_MUTACAO = 0.15


def torneio(
    populacao: list[Individuo],
    fitnesses: list[float],
    k: int,
    rng: np.random.Generator,
) -> Individuo:
    """Escolhe o melhor entre `k` sorteados.

    `k` e o botao de pressao seletiva: k=1 e sorteio uniforme, k grande faz a
    populacao convergir cedo para o melhor individuo. E o parametro que os
    experimentos E2 e E3 variam em direcoes opostas.

    O sorteio e **sem reposicao**. Com reposicao, um torneio do tamanho da
    populacao ainda poderia deixar o melhor individuo de fora, e a pressao
    seletiva de `k` deixaria de ter significado exato.
    """
    indices = rng.choice(len(populacao), size=min(k, len(populacao)), replace=False)
    melhor = max(indices, key=lambda indice: fitnesses[indice])
    return populacao[int(melhor)]


def cruzar(
    pai: Individuo,
    mae: Individuo,
    taxa: float,
    rng: np.random.Generator,
    *,
    segmentos: frozenset[str] = SEGMENTOS,
) -> tuple[Individuo, Individuo]:
    """Uniforme na mascara, aritmetico nos reais, troca nos demais."""
    if rng.random() >= taxa:
        return pai, mae

    mascara_a, mascara_b = pai.mascara, mae.mascara
    if "mascara" in segmentos:
        sorteio = rng.random(len(pai.mascara)) < 0.5
        mascara_a = tuple(
            int(m if troca else p)
            for p, m, troca in zip(pai.mascara, mae.mascara, sorteio)
        )
        mascara_b = tuple(
            int(p if troca else m)
            for p, m, troca in zip(pai.mascara, mae.mascara, sorteio)
        )

    hiper_a, hiper_b = dict(pai.hiperparametros), dict(mae.hiperparametros)
    if "hiperparametros" in segmentos:
        for gene in ESPACOS[pai.familia]:
            valor_pai, valor_mae = hiper_a[gene.nome], hiper_b[gene.nome]
            if valor_pai == valor_mae:
                # Nada a combinar. A guarda importa: sem ela, a media ponderada
                # de dois valores iguais pode devolver o ultimo bit diferente, e
                # o cruzamento deixaria de preservar o material genetico do pai.
                continue
            if isinstance(gene, GeneReal):
                peso = float(rng.random())
                hiper_a[gene.nome] = peso * valor_pai + (1 - peso) * valor_mae
                hiper_b[gene.nome] = (1 - peso) * valor_pai + peso * valor_mae
            elif rng.random() < 0.5:
                hiper_a[gene.nome], hiper_b[gene.nome] = valor_mae, valor_pai

    limiar_a, limiar_b = pai.limiar, mae.limiar
    if "limiar" in segmentos and pai.limiar != mae.limiar:
        peso = float(rng.random())
        limiar_a = peso * pai.limiar + (1 - peso) * mae.limiar
        limiar_b = (1 - peso) * pai.limiar + peso * mae.limiar

    filho_a = reparar(Individuo(pai.familia, mascara_a, hiper_a, limiar_a), rng)
    filho_b = reparar(Individuo(pai.familia, mascara_b, hiper_b, limiar_b), rng)
    return filho_a, filho_b


def mutar(
    individuo: Individuo,
    taxa: float,
    rng: np.random.Generator,
    *,
    segmentos: frozenset[str] = SEGMENTOS,
) -> Individuo:
    """Mutacao por gene: cada gene sorteia a propria chance contra `taxa`."""
    mascara = individuo.mascara
    if "mascara" in segmentos:
        sorteio = rng.random(len(individuo.mascara)) < taxa
        mascara = tuple(
            int(not bit) if muda else int(bit)
            for bit, muda in zip(individuo.mascara, sorteio)
        )

    hiperparametros = dict(individuo.hiperparametros)
    if "hiperparametros" in segmentos:
        for gene in ESPACOS[individuo.familia]:
            if rng.random() >= taxa:
                continue
            hiperparametros[gene.nome] = _mutar_gene(gene, hiperparametros[gene.nome], rng)

    limiar = individuo.limiar
    if "limiar" in segmentos and rng.random() < taxa:
        limiar = individuo.limiar + float(rng.normal(0.0, DESVIO_MUTACAO))

    return reparar(Individuo(individuo.familia, mascara, hiperparametros, limiar), rng)


def _mutar_gene(gene: Gene, valor, rng: np.random.Generator):
    """Perturbacao proporcional a amplitude do gene; o reparo corta o excesso."""
    if isinstance(gene, GeneReal):
        if gene.log:
            expoente = np.log10(max(float(valor), gene.minimo))
            amplitude = np.log10(gene.maximo) - np.log10(gene.minimo)
            return float(10 ** (expoente + rng.normal(0.0, DESVIO_MUTACAO * amplitude)))
        amplitude = gene.maximo - gene.minimo
        return float(valor + rng.normal(0.0, DESVIO_MUTACAO * amplitude))
    if isinstance(gene, GeneInteiro):
        amplitude = gene.maximo - gene.minimo
        return int(round(float(valor) + rng.normal(0.0, DESVIO_MUTACAO * amplitude)))
    # Categorico nao tem vizinhanca definida: mutacao e novo sorteio.
    return sortear_gene(gene, rng)
