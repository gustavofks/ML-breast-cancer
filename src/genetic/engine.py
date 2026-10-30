"""Laco evolutivo: geracoes, elitismo, historico e memorizacao do fitness.

O motor nao conhece a base. Ele recebe `(X, y)` — sempre o conjunto de treino —
e devolve o historico completo da execucao, para que o relatorio possa mostrar a
curva de convergencia em vez de afirmar que houve convergencia.

A memorizacao existe por economia real: uma avaliacao custa uma validacao
cruzada de 5 folds, e o elitismo garante que os melhores individuos reaparecam
identicos geracao apos geracao.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config
from src.genetic import operators as ops
from src.genetic.chromosome import SEGMENTOS, Individuo, chave, sortear
from src.genetic.chromosome import para_dicionario as individuo_para_dicionario
from src.genetic.fitness import Avaliacao, CustoClinico, avaliar


@dataclass(frozen=True)
class ConfiguracaoAG:
    """Parametros do algoritmo genetico. A funcao fitness fica de proposito fora.

    Os experimentos da secao 4.5 do design variam **somente** estes campos: e
    isso que torna a comparacao entre eles interpretavel.
    """

    nome: str
    populacao: int
    geracoes: int
    taxa_mutacao: float
    taxa_cruzamento: float
    torneio: int
    elitismo: int = 2

    def __post_init__(self) -> None:
        """Rejeita configuracao invalida na construcao, nao no meio do laco.

        O caso que motiva a validacao e `elitismo >= populacao`: a elite ocuparia
        a geracao inteira, nenhum filho seria gerado, e o laco repetiria a mesma
        populacao sem erro nenhum — um algoritmo genetico que nao evolui e um
        relatorio afirmando convergencia que nunca houve. Os demais limites estao
        aqui pelo mesmo motivo: falhar com mensagem clara custa menos que
        depurar um estouro obscuro trinta geracoes adiante.
        """
        if self.populacao < 2:
            raise ValueError(
                f"populacao deve ser pelo menos 2 para haver dois pais, "
                f"recebido {self.populacao}."
            )
        if self.geracoes < 1:
            raise ValueError(f"geracoes deve ser pelo menos 1, recebido {self.geracoes}.")
        if not 0 <= self.elitismo < self.populacao:
            raise ValueError(
                f"elitismo deve ficar entre 0 e populacao - 1 "
                f"({self.populacao - 1}), recebido {self.elitismo}."
            )
        if not 1 <= self.torneio <= self.populacao:
            raise ValueError(
                f"torneio deve ficar entre 1 e a populacao ({self.populacao}), "
                f"recebido {self.torneio}."
            )
        for nome_taxa, taxa in (
            ("taxa_mutacao", self.taxa_mutacao),
            ("taxa_cruzamento", self.taxa_cruzamento),
        ):
            if not 0.0 <= taxa <= 1.0:
                raise ValueError(f"{nome_taxa} deve ficar em [0, 1], recebido {taxa}.")


@dataclass(frozen=True)
class RegistroGeracao:
    """Uma linha da curva de convergencia."""

    geracao: int
    melhor_fitness: float
    fitness_medio: float
    desvio_fitness: float
    diversidade: float


@dataclass
class Resultado:
    """Tudo que uma execucao produz, pronto para virar JSON e figura."""

    configuracao: ConfiguracaoAG
    familia: str
    semente: int
    segmentos: frozenset[str]
    historico: list[RegistroGeracao]
    melhor: Individuo
    melhor_avaliacao: Avaliacao
    melhores: list[tuple[Individuo, Avaliacao]]
    avaliacoes_totais: int


def diversidade(populacao: list[Individuo]) -> float:
    """Fracao esperada de genes de mascara que diferem entre dois individuos.

    Medida direta, sem sortear pares: para cada gene, a chance de dois
    individuos discordarem e `2p(1-p)`. Vale 0 em populacao uniforme e 0,5 no
    maximo, e e o numero que expoe a perda de diversidade do experimento E3.
    """
    mascaras = np.array([individuo.mascara for individuo in populacao], dtype=float)
    proporcoes = mascaras.mean(axis=0)
    return float((2 * proporcoes * (1 - proporcoes)).mean())


def evoluir(
    configuracao: ConfiguracaoAG,
    familia: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    semente: int = config.SEED,
    custo: CustoClinico | None = None,
    segmentos: frozenset[str] = SEGMENTOS,
    cv_seed: int = config.SEED,
    n_jobs: int = 1,
) -> Resultado:
    """Executa uma busca completa e devolve o historico e os cinco melhores.

    Os cinco melhores existem para a verificacao de estabilidade da secao 4.7 do
    design: eles sao reavaliados depois com outra semente de validacao cruzada, e
    queda acentuada denuncia sobreajuste as particoes.

    `n_jobs` avalia os individuos novos de cada geracao em paralelo, um por
    processo. O resultado e identico ao sequencial: a avaliacao depende so do
    individuo e da semente da validacao cruzada, nunca do gerador da busca, e a
    memoria e preenchida na mesma ordem de primeira aparicao.
    """
    custo = custo or CustoClinico()
    rng = np.random.default_rng(semente)
    memoria: dict[tuple, Avaliacao] = {}
    individuos_vistos: dict[tuple, Individuo] = {}

    def pontuar_populacao(populacao: list[Individuo]) -> list[Avaliacao]:
        # Individuos ainda nao avaliados, na ordem de primeira aparicao e sem
        # repeticao. A ordem importa: o ranking dos cinco melhores desempata por
        # ela, e a versao paralela tem de inserir na memoria como a sequencial.
        novos: dict[tuple, Individuo] = {}
        for individuo in populacao:
            identidade = chave(individuo)
            if identidade not in memoria and identidade not in novos:
                novos[identidade] = individuo

        if n_jobs == 1:
            avaliacoes = [
                avaliar(individuo, X, y, custo=custo, cv_seed=cv_seed)
                for individuo in novos.values()
            ]
        else:
            avaliacoes = Parallel(n_jobs=n_jobs)(
                delayed(avaliar)(individuo, X, y, custo=custo, cv_seed=cv_seed)
                for individuo in novos.values()
            )

        for (identidade, individuo), avaliacao in zip(novos.items(), avaliacoes, strict=True):
            memoria[identidade] = avaliacao
            individuos_vistos[identidade] = individuo
        return [memoria[chave(individuo)] for individuo in populacao]

    populacao = [
        sortear(familia, rng, segmentos=segmentos) for _ in range(configuracao.populacao)
    ]
    historico: list[RegistroGeracao] = []
    melhor_global: tuple[Individuo, Avaliacao] | None = None

    for geracao in range(configuracao.geracoes):
        avaliacoes = pontuar_populacao(populacao)
        fitnesses = [avaliacao.fitness for avaliacao in avaliacoes]

        indice_melhor = int(np.argmax(fitnesses))
        if melhor_global is None or fitnesses[indice_melhor] > melhor_global[1].fitness:
            melhor_global = (populacao[indice_melhor], avaliacoes[indice_melhor])

        historico.append(
            RegistroGeracao(
                geracao=geracao,
                # Melhor **desta** geracao, e nao o melhor acumulado. E essa
                # escolha que torna a monotonicidade da serie uma afirmacao
                # sobre o elitismo: com elitismo ela nunca cai, e sem elitismo a
                # queda aparece no historico em vez de ficar escondida atras de
                # um maximo acumulado.
                melhor_fitness=round(float(max(fitnesses)), 6),
                fitness_medio=round(float(np.mean(fitnesses)), 6),
                desvio_fitness=round(float(np.std(fitnesses)), 6),
                diversidade=round(diversidade(populacao), 6),
            )
        )

        if geracao == configuracao.geracoes - 1:
            break

        ordenados = sorted(zip(populacao, fitnesses), key=lambda par: par[1], reverse=True)
        proxima = [individuo for individuo, _ in ordenados[: configuracao.elitismo]]

        while len(proxima) < configuracao.populacao:
            pai = ops.torneio(populacao, fitnesses, configuracao.torneio, rng)
            mae = ops.torneio(populacao, fitnesses, configuracao.torneio, rng)
            filho_a, filho_b = ops.cruzar(
                pai, mae, configuracao.taxa_cruzamento, rng, segmentos=segmentos
            )
            for filho in (filho_a, filho_b):
                if len(proxima) < configuracao.populacao:
                    proxima.append(
                        ops.mutar(filho, configuracao.taxa_mutacao, rng, segmentos=segmentos)
                    )

        populacao = proxima

    ranking = sorted(memoria.items(), key=lambda par: par[1].fitness, reverse=True)
    melhores = [
        (individuos_vistos[identidade], avaliacao) for identidade, avaliacao in ranking[:5]
    ]

    return Resultado(
        configuracao=configuracao,
        familia=familia,
        semente=semente,
        segmentos=segmentos,
        historico=historico,
        melhor=melhor_global[0],
        melhor_avaliacao=melhor_global[1],
        melhores=melhores,
        avaliacoes_totais=len(memoria),
    )


def para_dicionario(resultado: Resultado, colunas: list[str]) -> dict[str, Any]:
    """Execucao inteira em forma serializavel, para `results/genetic/`."""
    return {
        "experimento": resultado.configuracao.nome,
        "familia": resultado.familia,
        "semente": resultado.semente,
        "segmentos": sorted(resultado.segmentos),
        "configuracao": {
            "populacao": resultado.configuracao.populacao,
            "geracoes": resultado.configuracao.geracoes,
            "taxa_mutacao": resultado.configuracao.taxa_mutacao,
            "taxa_cruzamento": resultado.configuracao.taxa_cruzamento,
            "torneio": resultado.configuracao.torneio,
            "elitismo": resultado.configuracao.elitismo,
        },
        "avaliacoes_totais": resultado.avaliacoes_totais,
        "historico": [vars(registro) for registro in resultado.historico],
        "melhor": individuo_para_dicionario(resultado.melhor, colunas),
        "avaliacao": vars(resultado.melhor_avaliacao),
        "melhores": [
            {
                "individuo": individuo_para_dicionario(individuo, colunas),
                "avaliacao": vars(avaliacao),
            }
            for individuo, avaliacao in resultado.melhores
        ],
    }
