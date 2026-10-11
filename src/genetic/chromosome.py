"""Codificacao, sorteio, reparo e decodificacao do individuo.

O cromossomo e **tipado em tres segmentos** em vez de um vetor de bits
homogeneo: mascara de features, hiperparametros da familia e limiar de decisao.
A escolha e deliberada — cada tipo de gene exige mutacao e cruzamento
diferentes, e forcar tudo em binario tornaria a decodificacao obscura sem
nenhum ganho de busca nesta base.

Genes que poderiam ser reais continuos (`max_depth`, `max_features`) sao
declarados categoricos com valores discretos para que cada gene tenha um
operador so. Gene de tipo misto exigiria operador condicional, complexidade sem
retorno aqui.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import build_pipeline

LIMIAR_MIN = config.LIMIAR_MIN
LIMIAR_MAX = config.LIMIAR_MAX

# Limiar do braco de comparacao que nao otimiza o limiar: o padrao do sklearn.
LIMIAR_PADRAO = 0.5

N_FEATURES = config.EXPECTED_N_FEATURES

SEGMENTOS = frozenset({"mascara", "hiperparametros", "limiar"})


# ---------------------------------------------------------------------------
# Tipos de gene
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GeneReal:
    """Gene continuo. Com `log=True` a amostragem e uniforme em escala log.

    `C` da regressao logistica varia por cinco ordens de grandeza; sortear
    uniformemente na escala linear concentraria quase toda a populacao acima de
    10, e a regiao interessante nunca seria visitada.
    """

    nome: str
    minimo: float
    maximo: float
    log: bool = False


@dataclass(frozen=True)
class GeneInteiro:
    """Gene inteiro. `passo=2` a partir de `minimo` impar mantem so os impares."""

    nome: str
    minimo: int
    maximo: int
    passo: int = 1


@dataclass(frozen=True)
class GeneCategorico:
    """Gene de dominio finito, com ou sem ordem."""

    nome: str
    valores: tuple[Any, ...]


Gene = GeneReal | GeneInteiro | GeneCategorico


# ---------------------------------------------------------------------------
# Espacos de busca por familia
# ---------------------------------------------------------------------------
ESPACOS: dict[str, tuple[Gene, ...]] = {
    "Regressão Logística": (
        GeneReal("C", 1e-3, 1e2, log=True),
        GeneCategorico("class_weight", (None, "balanced")),
    ),
    "KNN": (
        # Vizinhanca impar evita empate na votacao de um problema binario.
        GeneInteiro("n_neighbors", 1, 31, passo=2),
        GeneCategorico("weights", ("uniform", "distance")),
        GeneCategorico("p", (1, 2)),
    ),
    "Random Forest": (
        # Teto de 300 arvores, e nao 500, por custo medido: uma avaliacao de
        # fitness do Random Forest custa 2,54 s contra 0,056 s da regressao
        # logistica, 45 vezes mais. Com 500 os quatro experimentos sobre as tres
        # familias passariam de 12 horas; com 300, caem para cerca de 8. E a
        # mitigacao que a secao 12 do design prescreveu para esse risco.
        #
        # O teto para exatamente em 300 porque e o valor da Fase 1, declarado em
        # PADROES. Como `reparar` corta todo gene para dentro do dominio, um teto
        # menor substituiria o padrao da Fase 1 em silencio, e o braco de
        # comparacao deixaria de reproduzir o modelo original.
        GeneInteiro("n_estimators", 50, 300),
        GeneCategorico("max_depth", (None, 2, 4, 6, 8, 10, 14, 20)),
        GeneInteiro("min_samples_leaf", 1, 10),
        GeneCategorico("max_features", ("sqrt", "log2", 0.3, 0.5, 0.7, 1.0)),
    ),
}

# Valores dos bracos de comparacao que nao otimizam hiperparametro: exatamente
# os defaults usados por `src.models.get_models` na Fase 1.
PADROES: dict[str, dict[str, Any]] = {
    "Regressão Logística": {"C": 1.0, "class_weight": None},
    "KNN": {"n_neighbors": 5, "weights": "uniform", "p": 2},
    "Random Forest": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
}

CONSTRUTORES: dict[str, Callable[[dict[str, Any]], BaseEstimator]] = {
    "Regressão Logística": lambda p: LogisticRegression(
        max_iter=5000, random_state=config.SEED, **p
    ),
    "KNN": lambda p: KNeighborsClassifier(**p),
    # n_jobs=1: o paralelismo fica na populacao (`engine.evoluir`), nao dentro de
    # cada floresta. Com 455 amostras, distribuir arvores entre threads custa
    # mais que treina-las (1,65 s contra 1,18 s por avaliacao, medido), e o
    # resultado e identico: com `random_state` fixo, n_jobs nao muda a floresta.
    "Random Forest": lambda p: RandomForestClassifier(
        random_state=config.SEED, n_jobs=1, **p
    ),
}


@dataclass(frozen=True)
class Individuo:
    """Uma solucao candidata: quais medidas usar, como treinar, onde cortar."""

    familia: str
    mascara: tuple[int, ...]
    hiperparametros: dict[str, Any]
    limiar: float

    def substituir(self, **campos: Any) -> "Individuo":
        """Copia com campos trocados.

        O individuo e imutavel para que a populacao anterior nunca seja alterada
        por engano dentro de um operador.
        """
        return replace(self, **campos)


# ---------------------------------------------------------------------------
# Sorteio
# ---------------------------------------------------------------------------
def sortear_gene(gene: Gene, rng: np.random.Generator) -> Any:
    """Valor aleatorio dentro do dominio declarado do gene."""
    if isinstance(gene, GeneReal):
        if gene.log:
            expoente = rng.uniform(np.log10(gene.minimo), np.log10(gene.maximo))
            return float(10.0**expoente)
        return float(rng.uniform(gene.minimo, gene.maximo))
    if isinstance(gene, GeneInteiro):
        candidatos = list(range(gene.minimo, gene.maximo + 1, gene.passo))
        return int(candidatos[int(rng.integers(len(candidatos)))])
    return gene.valores[int(rng.integers(len(gene.valores)))]


def sortear(
    familia: str,
    rng: np.random.Generator,
    *,
    segmentos: frozenset[str] = SEGMENTOS,
) -> Individuo:
    """Individuo aleatorio. Segmento fora de `segmentos` nasce no valor padrao.

    E assim que os bracos de ablacao da secao 4.6 do design sao construidos: o
    braco "AG so limiar" recebe `segmentos={"limiar"}` e todo o resto fica igual
    ao modelo original da Fase 1.
    """
    if "mascara" in segmentos:
        mascara = tuple(int(bit) for bit in rng.integers(0, 2, size=N_FEATURES))
    else:
        mascara = (1,) * N_FEATURES

    if "hiperparametros" in segmentos:
        hiperparametros = {gene.nome: sortear_gene(gene, rng) for gene in ESPACOS[familia]}
    else:
        hiperparametros = dict(PADROES[familia])

    if "limiar" in segmentos:
        limiar = float(rng.uniform(LIMIAR_MIN, LIMIAR_MAX))
    else:
        limiar = LIMIAR_PADRAO

    return reparar(Individuo(familia, mascara, hiperparametros, limiar), rng)


# ---------------------------------------------------------------------------
# Reparo
# ---------------------------------------------------------------------------
def reparar(individuo: Individuo, rng: np.random.Generator) -> Individuo:
    """Devolve um individuo valido, seja qual for a entrada.

    Roda depois de todo operador, para que nenhum individuo invalido chegue ao
    fitness. Nenhum gene de penalidade (`l1`/`l2`) entrou no espaco justamente
    para nao criar par penalidade-solver incompativel: quando o espaco nao
    permite a combinacao invalida, o reparo nao precisa saber dela.
    """
    mascara = tuple(1 if bit else 0 for bit in individuo.mascara)
    if sum(mascara) == 0:
        # Mascara zerada nao tem matriz de entrada. Ativa uma medida sorteada,
        # e nao uma fixa, para nao enviesar a populacao reparada.
        indice = int(rng.integers(N_FEATURES))
        mascara = tuple(1 if i == indice else 0 for i in range(N_FEATURES))

    hiperparametros = {
        gene.nome: _reparar_gene(gene, individuo.hiperparametros.get(gene.nome))
        for gene in ESPACOS[individuo.familia]
    }

    limiar = float(np.clip(individuo.limiar, LIMIAR_MIN, LIMIAR_MAX))

    return Individuo(individuo.familia, mascara, hiperparametros, limiar)


def _reparar_gene(gene: Gene, valor: Any) -> Any:
    """Corta o valor para dentro do dominio; valor irrecuperavel vira o primeiro."""
    if isinstance(gene, GeneReal):
        if valor is None:
            return gene.minimo
        return float(np.clip(float(valor), gene.minimo, gene.maximo))
    if isinstance(gene, GeneInteiro):
        if valor is None:
            return gene.minimo
        cortado = int(np.clip(int(round(float(valor))), gene.minimo, gene.maximo))
        # Reancora no passo, para que `n_neighbors` continue impar.
        cortado -= (cortado - gene.minimo) % gene.passo
        return int(np.clip(cortado, gene.minimo, gene.maximo))
    return valor if valor in gene.valores else gene.valores[0]


# ---------------------------------------------------------------------------
# Decodificacao e serializacao
# ---------------------------------------------------------------------------
def decodificar(individuo: Individuo, colunas: list[str]) -> tuple[Pipeline, list[str]]:
    """Traduz o individuo no pipeline da Fase 1 e na lista de medidas ativas.

    O pipeline sai de `build_pipeline`, e nao de um construtor proprio: e o que
    garante que o modelo otimizado e o modelo original sejam treinados pelo
    mesmo codigo, com a padronizacao reajustada dentro de cada fold.
    """
    selecionadas = [coluna for coluna, ativo in zip(colunas, individuo.mascara) if ativo]
    estimador = CONSTRUTORES[individuo.familia](dict(individuo.hiperparametros))
    return build_pipeline(estimador), selecionadas


def chave(individuo: Individuo) -> tuple:
    """Identidade hashavel do individuo, usada para memorizar o fitness.

    Reavaliar o mesmo individuo custa uma validacao cruzada inteira, e o
    elitismo garante que os melhores reaparecam geracao apos geracao.
    """
    hiperparametros = tuple(
        sorted((nome, repr(valor)) for nome, valor in individuo.hiperparametros.items())
    )
    return (individuo.familia, individuo.mascara, hiperparametros, round(individuo.limiar, 6))


def para_dicionario(individuo: Individuo, colunas: list[str]) -> dict[str, Any]:
    """Forma legivel para JSON: o relatorio cita features por nome, nao por indice."""
    selecionadas = [coluna for coluna, ativo in zip(colunas, individuo.mascara) if ativo]
    return {
        "familia": individuo.familia,
        "n_features": len(selecionadas),
        "features": selecionadas,
        "mascara": list(individuo.mascara),
        "hiperparametros": {
            nome: (round(valor, 6) if isinstance(valor, float) else valor)
            for nome, valor in individuo.hiperparametros.items()
        },
        "limiar": round(individuo.limiar, 4),
    }
