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
from src.genetic.fitness import CustoClinico, reavaliar_lote  # noqa: E402
from src.preprocessing import split_data  # noqa: E402

# Deslocamento da semente usada na verificacao de estabilidade. Fixo, e nao
# sorteado, para que o artefato continue reprodutivel entre execucoes.
DESLOCAMENTO_DA_ESTABILIDADE = 100

Execucao = tuple[engine.ConfiguracaoAG, str, int, frozenset[str], CustoClinico]

CONFIGURACAO_SMOKE = engine.ConfiguracaoAG(
    nome="smoke",
    populacao=6,
    geracoes=3,
    taxa_mutacao=0.2,
    taxa_cruzamento=0.8,
    torneio=2,
    elitismo=2,
)


def montar_execucoes(smoke: bool) -> list[Execucao]:
    """Execucoes dos experimentos e dos bracos, na politica de custo padrao.

    Os quatro experimentos variam **apenas** parametros do algoritmo genetico; a
    funcao fitness fica fixa. E isso que torna a comparacao entre eles
    interpretavel, e por isso o custo e o mesmo em todas estas execucoes.

    No modo completo sao os quatro experimentos sobre as tres familias com tres
    sementes, mais os bracos de ablacao sobre a regressao logistica.
    """
    padrao = CustoClinico()

    if smoke:
        return [(CONFIGURACAO_SMOKE, "Regressão Logística", config.SEED, SEGMENTOS, padrao)]

    execucoes = [
        (configuracao, familia, semente, SEGMENTOS, padrao)
        for configuracao in experiments.EXPERIMENTOS
        for familia in experiments.FAMILIAS
        for semente in experiments.SEMENTES
    ]
    execucoes += [
        (
            experiments.EXPERIMENTOS[0],
            experiments.FAMILIA_DOS_BRACOS,
            semente,
            segmentos,
            padrao,
        )
        for nome, segmentos in experiments.BRACOS.items()
        if nome != "AG, completo"
        for semente in experiments.SEMENTES
    ]
    return execucoes


def montar_varredura(smoke: bool = False) -> list[Execucao]:
    """Varredura da secao 4.5.1: E1 na regressao logistica, nove politicas de custo.

    Analise separada, que nao entra na contagem de experimentos do enunciado. A
    configuracao do algoritmo fica fixa justamente para que a unica coisa que
    varie seja a politica clinica — a razao de custo e o peso de parcimonia sao
    decisao da instituicao de saude, e a varredura entrega a curva que torna essa
    decisao informada.
    """
    configuracao = CONFIGURACAO_SMOKE if smoke else experiments.CONFIGURACAO_DA_VARREDURA
    return [
        (configuracao, experiments.FAMILIA_DOS_BRACOS, semente, SEGMENTOS, custo)
        for custo in experiments.VARREDURA_CUSTO
        for semente in experiments.SEMENTES
    ]


def destino_padrao(sweep: bool) -> Path:
    """Arquivo de saida quando `--saida` nao e passado.

    Cada busca tem o seu. Com destino unico, rodar a varredura depois dos
    experimentos sobrescreveria horas de execucao sem erro: o arquivo continuaria
    valido, so que com outras execucoes dentro.
    """
    return config.GENETIC_DIR / ("varredura.json" if sweep else "execucoes.json")


# Campos de `ConfiguracaoAG` que mudam o resultado. O nome fica de fora daqui
# porque o payload o grava em `experimento`, e nao dentro de `configuracao`.
CAMPOS_DA_CONFIGURACAO = (
    "populacao",
    "geracoes",
    "taxa_mutacao",
    "taxa_cruzamento",
    "torneio",
    "elitismo",
)


def identidade_do_payload(payload: dict) -> str:
    """O que distingue uma execucao gravada de outra, para decidir se ja foi feita.

    Inclui a configuracao inteira, e nao so o nome do experimento: se a populacao
    mudou entre duas rodadas, o resultado gravado responde a outra pergunta, e
    reaproveita-lo misturaria configuracoes sem erro nenhum.
    """
    return json.dumps(
        {
            "configuracao": {
                campo: payload["configuracao"][campo] for campo in CAMPOS_DA_CONFIGURACAO
            },
            "experimento": payload["experimento"],
            "familia": payload["familia"],
            "semente": payload["semente"],
            "segmentos": sorted(payload["segmentos"]),
            "custo": payload["custo"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def identidade(
    configuracao, familia: str, semente: int, segmentos, custo: CustoClinico
) -> str:
    """A mesma identidade, para uma execucao ainda por fazer.

    Monta os campos no formato do payload e delega, para que as duas pontas nao
    possam divergir.
    """
    return identidade_do_payload(
        {
            "configuracao": {
                campo: getattr(configuracao, campo) for campo in CAMPOS_DA_CONFIGURACAO
            },
            "experimento": configuracao.nome,
            "familia": familia,
            "semente": semente,
            "segmentos": list(segmentos),
            "custo": {
                "falso_negativo": custo.falso_negativo,
                "falso_positivo": custo.falso_positivo,
                "parcimonia": custo.parcimonia,
            },
        }
    )


def _gravar(payloads: list[dict], destino: Path) -> None:
    """Grava o artefato inteiro, sobrescrevendo o anterior."""
    destino.write_text(json.dumps(payloads, indent=2, ensure_ascii=False), encoding="utf-8")


def nome_do_braco(segmentos: frozenset[str]) -> str:
    """Nome legivel do braco, para a tabela comparativa do relatorio."""
    for nome, candidato in experiments.BRACOS.items():
        if candidato == segmentos:
            return nome
    return "AG, completo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca por algoritmo genetico.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="execucao minima, para verificar o encanamento sem esperar horas",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="roda a varredura de sensibilidade da funcao fitness, nao os experimentos",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help=(
            "arquivo JSON de saida (padrao: results/genetic/execucoes.json, "
            "ou varredura.json com --sweep)"
        ),
    )
    parser.add_argument(
        "--retomar",
        action="store_true",
        help="reaproveita as execucoes ja gravadas na saida e roda so as que faltam",
    )
    argumentos = parser.parse_args()

    X, y = load_dataset()
    # O teste e descartado aqui e nao volta a aparecer neste arquivo.
    X_train, _, y_train, _ = split_data(X, y)
    colunas = list(X_train.columns)

    if argumentos.sweep:
        execucoes = montar_varredura(argumentos.smoke)
    else:
        execucoes = montar_execucoes(argumentos.smoke)
    print(f"{len(execucoes)} execucoes, {len(y_train)} amostras de treino")

    destino = argumentos.saida or destino_padrao(argumentos.sweep)
    destino.parent.mkdir(parents=True, exist_ok=True)

    # Sem `--retomar`, um artefato existente nao e sobrescrito. Recomecar de
    # lista vazia regravaria o arquivo na primeira execucao concluida, e as horas
    # ja gravadas sumiriam sem erro nenhum.
    ja_feitas: dict[str, dict] = {}
    if destino.exists() and destino.stat().st_size > 0:
        if not argumentos.retomar:
            sys.exit(
                f"{destino} ja existe. Use --retomar para aproveitar as execucoes "
                "gravadas, ou apague o arquivo para recomecar do zero."
            )
        ja_feitas = {
            identidade_do_payload(payload): payload
            for payload in json.loads(destino.read_text(encoding="utf-8"))
        }

    payloads = []
    for indice, (configuracao, familia, semente, segmentos, custo) in enumerate(
        execucoes, start=1
    ):
        rotulo = f"{configuracao.nome} | {familia} | semente {semente} | {sorted(segmentos)}"
        print(f"{indice}/{len(execucoes)}  {rotulo}")

        # Cada execucao nasce da propria semente, sem estado compartilhado com as
        # outras: reaproveitar a gravada da o mesmo resultado que refaze-la. A
        # ordem do arquivo segue a lista de execucoes, e nao a ordem de
        # conclusao, para que o artefato retomado seja identico ao continuo.
        gravada = ja_feitas.get(identidade(configuracao, familia, semente, segmentos, custo))
        if gravada is not None:
            payloads.append(gravada)
            print("      ja gravada, reaproveitada")
            continue

        inicio = time.perf_counter()

        resultado = engine.evoluir(
            configuracao,
            familia,
            X_train,
            y_train,
            semente=semente,
            custo=custo,
            segmentos=segmentos,
        )

        payload = engine.para_dicionario(resultado, colunas)
        payload["braco"] = nome_do_braco(segmentos)
        payload["custo"] = {
            "falso_negativo": custo.falso_negativo,
            "falso_positivo": custo.falso_positivo,
            "parcimonia": custo.parcimonia,
        }
        # Verificacao de estabilidade da secao 4.7: os cinco melhores sao
        # reavaliados sob outra particao. Queda acentuada denuncia que a busca
        # aprendeu as dobras, e nao o problema.
        payload["estabilidade"] = [
            vars(item)
            for item in reavaliar_lote(
                resultado.melhores,
                X_train,
                y_train,
                custo=custo,
                cv_seed=config.SEED + DESLOCAMENTO_DA_ESTABILIDADE,
            )
        ]
        payloads.append(payload)
        # Regravado a cada execucao concluida. A execucao completa leva horas, e
        # gravar so no fim transformaria qualquer interrupcao em perda total.
        _gravar(payloads, destino)

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

    print(f"Gravado em {destino}")


if __name__ == "__main__":
    main()
