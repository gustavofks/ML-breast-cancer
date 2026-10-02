"""Executa as buscas do algoritmo genetico e grava o historico em results/genetic/.

Uso, a partir da raiz do projeto:

    python scripts/run_genetic.py --smoke        # verificacao rapida, minutos
    python scripts/run_genetic.py                # experimentos completos

O script separa treino e teste uma unica vez e entrega **somente o treino** ao
pacote `src.genetic`. O conjunto de teste e descartado no proprio
desempacotamento: nao existe variavel para acessar por engano. A avaliacao final
dos bracos contra o teste intocado e etapa propria, posterior.

E idempotente: rodar de novo com a mesma semente regenera o mesmo arquivo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.genetic import engine, experiments  # noqa: E402
from src.genetic.chromosome import SEGMENTOS  # noqa: E402
from src.preprocessing import split_data  # noqa: E402

CONFIGURACAO_SMOKE = engine.ConfiguracaoAG(
    nome="smoke",
    populacao=6,
    geracoes=3,
    taxa_mutacao=0.2,
    taxa_cruzamento=0.8,
    torneio=2,
    elitismo=2,
)


def montar_execucoes(
    smoke: bool,
) -> list[tuple[engine.ConfiguracaoAG, str, int, frozenset[str]]]:
    """Lista de (configuracao, familia, semente, segmentos) a executar.

    No modo completo sao os quatro experimentos sobre as tres familias com tres
    sementes, mais os bracos de ablacao sobre a regressao logistica.
    """
    if smoke:
        return [(CONFIGURACAO_SMOKE, "Regressão Logística", config.SEED, SEGMENTOS)]

    execucoes = [
        (configuracao, familia, semente, SEGMENTOS)
        for configuracao in experiments.EXPERIMENTOS
        for familia in experiments.FAMILIAS
        for semente in experiments.SEMENTES
    ]
    execucoes += [
        (experiments.EXPERIMENTOS[0], experiments.FAMILIA_DOS_BRACOS, semente, segmentos)
        for nome, segmentos in experiments.BRACOS.items()
        if nome != "AG, completo"
        for semente in experiments.SEMENTES
    ]
    return execucoes


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca por algoritmo genetico.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="execucao minima, para verificar o encanamento sem esperar horas",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help="arquivo JSON de saida (padrao: results/genetic/execucoes.json)",
    )
    argumentos = parser.parse_args()

    X, y = load_dataset()
    # O teste e descartado aqui e nao volta a aparecer neste arquivo.
    X_train, _, y_train, _ = split_data(X, y)
    colunas = list(X_train.columns)

    execucoes = montar_execucoes(argumentos.smoke)
    print(f"{len(execucoes)} execucoes, {len(y_train)} amostras de treino")

    payloads = []
    for indice, (configuracao, familia, semente, segmentos) in enumerate(execucoes, start=1):
        rotulo = f"{configuracao.nome} | {familia} | semente {semente} | {sorted(segmentos)}"
        print(f"{indice}/{len(execucoes)}  {rotulo}")
        inicio = time.perf_counter()

        resultado = engine.evoluir(
            configuracao,
            familia,
            X_train,
            y_train,
            semente=semente,
            segmentos=segmentos,
        )

        payloads.append(engine.para_dicionario(resultado, colunas))

        # O tempo decorrido vai para o log, nunca para o artefato: cronometro de
        # parede muda a cada execucao e quebraria a garantia de que rodar duas
        # vezes com a mesma semente produz o mesmo arquivo. O numero importa —
        # a secao 12 do design registra a lentidao do Random Forest como risco —
        # mas o lugar dele e a saida do terminal.
        segundos = round(time.perf_counter() - inicio, 2)
        avaliacao = resultado.melhor_avaliacao
        print(
            f"      fitness {avaliacao.fitness:.4f} | recall {avaliacao.recall:.4f} | "
            f"FN {avaliacao.falsos_negativos} | features {avaliacao.n_features} | "
            f"limiar {resultado.melhor.limiar:.3f} | {segundos}s"
        )

    destino = argumentos.saida or (config.GENETIC_DIR / "execucoes.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(payloads, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Gravado em {destino}")


if __name__ == "__main__":
    main()
