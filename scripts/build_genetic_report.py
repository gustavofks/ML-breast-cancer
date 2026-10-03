"""Gera `summary.json` e as figuras do algoritmo genetico a partir das execucoes.

Uso, a partir da raiz do projeto:

    python scripts/build_genetic_report.py

Le `results/genetic/execucoes.json` e o ranking SHAP de `results/metrics.json`, e
grava o consolidado e as tres figuras em `results/genetic/`. Se a varredura de
sensibilidade ja tiver rodado (`varredura.json`), a tabela dela entra no
consolidado e ganha a figura 14.

Existe como script, e nao como um trecho solto de linha de comando, pela mesma
regra que a Fase 1 aplica aos relatorios: nenhum numero e digitado a mao, e o que
produz os numeros precisa ser executavel de novo e testavel. E idempotente —
rodar duas vezes regenera os mesmos arquivos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # backend sem janela: o script roda em terminal

from src import config  # noqa: E402
from src.genetic import artifacts, plots  # noqa: E402

# Numeracao continua a das figuras da Fase 1, que vao ate a 10.
FIGURAS = {
    "convergencia": "11_convergencia.png",
    "fronteira": "12_fronteira_custo_features.png",
    "frequencia": "13_frequencia_features.png",
}

# Fora de FIGURAS porque so existe quando a varredura rodou.
FIGURA_VARREDURA = "14_varredura_custo.png"


def ler_shap(metricas: Path) -> list[dict]:
    """Ranking SHAP global da Fase 1, se existir.

    A ausencia nao e erro: o cruzamento da secao 4.9 e um extra sobre a
    agregacao, e travar a geracao inteira por causa dele seria desproporcional.
    """
    if not metricas.exists():
        return []
    payload = json.loads(metricas.read_text(encoding="utf-8"))
    return payload.get("explicabilidade", {}).get("shap_global", [])


def ler_varredura(caminho: Path | None) -> list[dict]:
    """Execucoes da varredura de sensibilidade, se ela ja tiver rodado.

    A varredura e analise separada, que roda depois e independente dos
    experimentos. Ausencia aqui nao e erro: e o estado normal antes de ela rodar.
    """
    if caminho is None or not caminho.exists():
        return []
    return json.loads(caminho.read_text(encoding="utf-8"))


def construir(
    execucoes: Path, metricas: Path, destino: Path, varredura: Path | None = None
) -> dict:
    """Monta o resumo, grava `summary.json` e as figuras, e devolve o resumo.

    As execucoes da varredura ficam numa chave propria e nunca entram nas tabelas
    dos experimentos: rodam sob outras politicas de custo, e misturadas
    distorceriam a contagem e a comparacao entre bracos.
    """
    payloads = json.loads(execucoes.read_text(encoding="utf-8"))
    shap_global = ler_shap(metricas)
    payloads_varredura = ler_varredura(varredura)

    resumo = artifacts.montar_summary(payloads, shap_global)
    # Lista vazia explicita quando a varredura nao rodou, para que o relatorio
    # saiba que ela falta em vez de achar que o campo foi esquecido.
    resumo["varredura"] = artifacts.tabela_da_varredura(payloads_varredura)

    destino.mkdir(parents=True, exist_ok=True)
    (destino / "summary.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plots.salvar(plots.figura_convergencia(payloads), FIGURAS["convergencia"], destino)
    plots.salvar(plots.figura_fronteira(payloads), FIGURAS["fronteira"], destino)
    plots.salvar(
        plots.figura_frequencia(payloads, shap_global), FIGURAS["frequencia"], destino
    )
    # Sem varredura, nenhuma figura: uma figura em branco seria lida como resultado.
    if payloads_varredura:
        plots.salvar(plots.figura_varredura(payloads_varredura), FIGURA_VARREDURA, destino)

    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description="Artefatos do algoritmo genetico.")
    parser.add_argument("--execucoes", type=Path, default=None)
    parser.add_argument("--metricas", type=Path, default=None)
    parser.add_argument("--destino", type=Path, default=None)
    parser.add_argument("--varredura", type=Path, default=None)
    argumentos = parser.parse_args()

    resumo = construir(
        argumentos.execucoes or (config.GENETIC_DIR / "execucoes.json"),
        argumentos.metricas or config.METRICS_FILE,
        argumentos.destino or config.GENETIC_DIR,
        argumentos.varredura or (config.GENETIC_DIR / "varredura.json"),
    )

    print(f"{resumo['execucoes']} execucoes agregadas")
    print()
    print(f"{'braco':<24} {'custo':>9} {'recall':>9} {'FN':>6} {'FP':>6} {'feat':>6}")
    for linha in resumo["bracos"]:
        print(
            f"{linha['braco']:<24} {linha['custo_medio']:>9.4f} "
            f"{linha['recall_medio']:>9.4f} {linha['falsos_negativos_medio']:>6.1f} "
            f"{linha['falsos_positivos_medio']:>6.1f} {linha['features_medio']:>6.1f}"
        )

    if resumo["varredura"]:
        print()
        print(f"{'razao':>6} {'alfa':>6} {'limiar':>8} {'feat':>6} {'FN':>6} {'FP':>6}")
        for linha in resumo["varredura"]:
            print(
                f"{linha['razao_custo']:>6g} {linha['parcimonia']:>6g} "
                f"{linha['limiar_medio']:>8.3f} {linha['features_medio']:>6.1f} "
                f"{linha['falsos_negativos_medio']:>6.1f} {linha['falsos_positivos_medio']:>6.1f}"
            )


if __name__ == "__main__":
    main()
