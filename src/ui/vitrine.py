"""Vitrine da Fase 2: processo, resultados e o servico, numa pagina so.

A mesma pagina roda em dois lugares. **Servida pelo container**, em `/`, ela
detecta a API pelo `/health` e chama as rotas de verdade: o console mostra a
resposta, o log e as metricas ao vivo. **Publicada no Space estatico**, onde nada
executa no servidor, ela usa o que foi congelado aqui: as respostas de `/explain`
para os 114 casos, a gravacao de uma sessao do console e os parametros do
simulador, que refaz a conta do modelo no navegador.

Nenhum numero e escrito a mao. Processo e resultados vem dos artefatos de
`results/`; casos, console e simulador vem da propria API, consultada em
processo. A API de montagem e uma segunda instancia, com logger proprio e fluxo
descartado: as chamadas da montagem nao aparecem no monitoramento da API servida.

Os ativos da pagina (molde, estilo, script e o Chart.js) ficam em
`src/ui/vitrine/` e sao embutidos no HTML: a pagina nao carrega nada de fora.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from src import config
from src.api import registro
from src.data import load_dataset
from src.preprocessing import split_data
from src.ui import estatico, simulador
from src.ui.app import AVISO

BRACO_COMPLETO = "AG, completo"

# Desvios padrao do cenario de desvio no console. Acima do limiar da sentinela
# (5) com folga, para o alerta aparecer sem ambiguidade na demonstracao.
DESVIOS_DO_CENARIO = 8


def _ler(caminho: Path) -> Any | None:
    """Artefato ausente vira `None`: a secao correspondente diz que falta, nao quebra."""
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Processo e resultados do algoritmo genetico
# ---------------------------------------------------------------------------
def convergencia(execucoes: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    """Melhor fitness por geracao, media das sementes, por experimento e familia.

    So o braco completo: os bracos de ablacao buscam num espaco menor e
    misturados na mesma curva pareceriam convergir "pior" por construcao.
    """
    grupos: dict[tuple[str, str], list[list[float]]] = {}
    for execucao in execucoes:
        if execucao["braco"] != BRACO_COMPLETO:
            continue
        curva = [g["melhor_fitness"] for g in execucao["historico"]]
        grupos.setdefault((execucao["experimento"], execucao["familia"]), []).append(curva)

    saida: dict[str, dict[str, list[float]]] = {}
    for (experimento, familia), curvas in sorted(grupos.items()):
        geracoes = min(len(c) for c in curvas)
        saida.setdefault(experimento, {})[familia] = [
            round(mean(c[g] for c in curvas), 6) for g in range(geracoes)
        ]
    return saida


def configuracoes(execucoes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Parametros do algoritmo por experimento, para a jornada citar o que mudou."""
    return {e["experimento"]: e["configuracao"] for e in execucoes if "configuracao" in e}


def processo(resultados: Path) -> dict[str, Any] | None:
    genetico = resultados / "genetic"
    resumo = _ler(genetico / "summary.json")
    execucoes = _ler(genetico / "execucoes.json")
    if resumo is None or execucoes is None:
        return None
    return {
        "execucoes": resumo["execucoes"],
        "pesos": next(e["custo"] for e in execucoes if e["braco"] == BRACO_COMPLETO),
        "configuracoes": configuracoes(execucoes),
        "convergencia": convergencia(execucoes),
        "bracos": resumo["bracos"],
        "features": resumo["features"],
        "varredura": resumo["varredura"],
        "teste": _ler(genetico / "comparacao_final.json"),
        "regra": _ler(genetico / "regra_do_modelo_servido.json"),
    }


# ---------------------------------------------------------------------------
# Qualidade dos textos
# ---------------------------------------------------------------------------
def textos(resultados: Path) -> dict[str, Any] | None:
    """Aprovacao por verificacao, versoes de prompt, custo e os textos de gestao."""
    llm = resultados / "llm"
    narrativas = _ler(llm / "narrativas.json")
    if narrativas is None:
        return None

    verificacoes: dict[str, int] = {}
    for narrativa in narrativas:
        for nome, resultado in narrativa["qualidade"]["verificacoes"].items():
            verificacoes[nome] = verificacoes.get(nome, 0) + int(resultado["aprovado"])

    por_produto: dict[str, int] = {}
    for narrativa in narrativas:
        por_produto[narrativa["identificador"]] = por_produto.get(narrativa["identificador"], 0) + 1

    return {
        "total": len(narrativas),
        "aprovados": sum(1 for n in narrativas if n["qualidade"]["aprovado"]),
        "verificacoes": verificacoes,
        "por_produto": por_produto,
        "prompts": _ler(llm / "comparacao_de_prompts.json"),
        "custo": _ler(llm / "custo.json"),
        "gestao": [
            {"identificador": n["identificador"], "texto": n["texto"]}
            for n in narrativas
            if not n["identificador"].startswith("explicacao_paciente")
        ],
    }


# ---------------------------------------------------------------------------
# Simulador e cenarios do console
# ---------------------------------------------------------------------------
def faixas(features: list[str]) -> dict[str, dict[str, float]]:
    """Minimo, maximo e media por classe de cada medida, **no treino**.

    Os controles do simulador e os corpos do console nunca olham o teste.
    """
    X, y = load_dataset()
    X_treino, _, y_treino, _ = split_data(X, y)
    saida = {}
    for nome in features:
        coluna = X_treino[nome]
        saida[nome] = {
            "minimo": float(coluna.min()),
            "maximo": float(coluna.max()),
            "media": float(coluna.mean()),
            "desvio": float(coluna.std(ddof=0)),
            "media_benigno": float(coluna[y_treino == 0].mean()),
            "media_maligno": float(coluna[y_treino == 1].mean()),
        }
    return saida


def cenarios(faixas_: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """As chamadas do console: uma por rota, mais desvio e entrada invalida."""
    nomes = list(faixas_)
    benigno = {n: round(f["media_benigno"], 4) for n, f in faixas_.items()}
    maligno = {n: round(f["media_maligno"], 4) for n, f in faixas_.items()}
    # A medida de maior escala relativa sofre o desvio: e onde um erro de unidade
    # na coleta (mm2 por cm2, por exemplo) apareceria de verdade.
    alvo = max(nomes, key=lambda n: faixas_[n]["desvio"] / max(abs(faixas_[n]["media"]), 1e-9))
    desvio = {
        **maligno,
        alvo: round(faixas_[alvo]["media"] + DESVIOS_DO_CENARIO * faixas_[alvo]["desvio"], 4),
    }
    faltante = {n: v for n, v in benigno.items() if n != nomes[0]}

    return [
        {"id": "health", "titulo": "Saúde do serviço", "metodo": "GET", "rota": "/health",
         "descricao": "Confirma que o serviço está de pé e o modelo carregou."},
        {"id": "model-info", "titulo": "Modelo servido", "metodo": "GET", "rota": "/model-info",
         "descricao": "Família, medidas, limiar, origem e commit do modelo em produção."},
        {"id": "predict-benigno", "titulo": "Previsão: perfil benigno", "metodo": "POST",
         "rota": "/predict", "corpo": {"medidas": benigno},
         "descricao": "Médias do treino para tumores benignos."},
        {"id": "predict-maligno", "titulo": "Previsão: perfil maligno", "metodo": "POST",
         "rota": "/predict", "corpo": {"medidas": maligno},
         "descricao": "Médias do treino para tumores malignos."},
        {"id": "explain-maligno", "titulo": "Explicação: perfil maligno", "metodo": "POST",
         "rota": "/explain", "corpo": {"medidas": maligno},
         "descricao": "A previsão mais o peso de cada medida no resultado."},
        {"id": "predict-desvio", "titulo": "Medida fora da distribuição", "metodo": "POST",
         "rota": "/predict", "corpo": {"medidas": desvio},
         "descricao": f"{alvo} a {DESVIOS_DO_CENARIO} desvios padrão da média do treino: "
                      "a previsão sai, e a sentinela registra o alerta."},
        {"id": "predict-faltante", "titulo": "Medida faltante", "metodo": "POST",
         "rota": "/predict", "corpo": {"medidas": faltante},
         "descricao": f"Sem {nomes[0]}: a API recusa com 422 e diz qual medida falta."},
        {"id": "metrics", "titulo": "Métricas", "metodo": "GET", "rota": "/metrics",
         "descricao": "Contadores, latências e alertas no formato do Prometheus."},
    ]


def gravar_console(api, logger, cenarios_: list[dict[str, Any]]) -> dict[str, Any]:
    """Executa os cenarios contra a API de montagem e guarda o que ela respondeu.

    E o que a versao estatica mostra no lugar das chamadas ao vivo: respostas,
    log e metricas reais, de uma sessao gravada, declarada como tal na tela.
    """
    respostas = {}
    for numero, cenario in enumerate(cenarios_, start=1):
        identificador = f"gravacao-{numero}"
        inicio = time.perf_counter()
        resposta = api.request(
            cenario["metodo"],
            cenario["rota"],
            json=cenario.get("corpo"),
            headers={"X-Request-ID": identificador},
        )
        tipo = resposta.headers.get("content-type", "")
        respostas[cenario["id"]] = {
            "status": resposta.status_code,
            "latencia_ms": round((time.perf_counter() - inicio) * 1000, 2),
            "id": resposta.headers.get("X-Request-ID", identificador),
            "corpo": resposta.json() if "json" in tipo else resposta.text,
        }
    return {
        "respostas": respostas,
        "eventos": registro.memoria(logger).depois(0),
        "metricas": api.get("/metrics").text,
    }


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------
def montar_dados(
    servico,
    *,
    resultados: Path = config.RESULTS_DIR,
    diretorio_cache: Path | None = None,
) -> dict[str, Any]:
    """Tudo o que a pagina mostra, consultando uma API propria, em processo."""
    from fastapi.testclient import TestClient

    from src.api.main import criar_app

    logger = registro.configurar("api.vitrine", io.StringIO())
    api = TestClient(
        criar_app(
            servico,
            diretorio_cache=diretorio_cache or resultados / "llm" / "cache",
            logger=logger,
        )
    )

    casos = estatico.dados_dos_casos(api)
    faixas_ = faixas(casos["modelo"]["features"])
    cenarios_ = cenarios(faixas_)
    # O console e gravado antes que qualquer outra chamada polua o log e as
    # metricas; os 114 /explain dos casos foram feitos acima, entao a memoria e
    # zerada com um logger novo.
    logger = registro.configurar("api.vitrine.console", io.StringIO())
    console_api = TestClient(
        criar_app(
            servico,
            diretorio_cache=diretorio_cache or resultados / "llm" / "cache",
            logger=logger,
        )
    )

    return {
        "modelo": casos["modelo"],
        "casos": casos["casos"],
        "simulador": {"parametros": simulador.parametros(servico), "faixas": faixas_},
        "console": {"cenarios": cenarios_, "gravacao": gravar_console(console_api, logger, cenarios_)},
        "processo": processo(resultados),
        "textos": textos(resultados),
    }


def _ativo(nome: str) -> str:
    return (config.VITRINE_DIR / nome).read_text(encoding="utf-8")


def pagina(dados: dict[str, Any]) -> str:
    """A pagina inteira, com estilo, Chart.js, script e dados embutidos."""
    return (
        _ativo("modelo.html")
        .replace("__ESTILO__", _ativo("estilo.css"))
        .replace("__CHARTJS__", _ativo("vendor/chart.umd.min.js"))
        .replace("__LOGICA__", _ativo("logica.js"))
        .replace("__VITRINE__", _ativo("vitrine.js"))
        .replace("__AVISO__", AVISO.replace("**", ""))
        .replace("__DADOS__", estatico._json_seguro(dados))
    )
