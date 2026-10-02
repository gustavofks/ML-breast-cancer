"""Gera os textos em linguagem natural e grava `results/llm/`.

Uso, a partir da raiz do projeto:

    python scripts/run_llm.py --fake     # verifica o encanamento, sem Ollama
    python scripts/run_llm.py            # geracao real, exige Ollama no ar

O modo `--fake` existe porque a camada inteira fala com um `Protocol`: da para
exercitar prompts, contratos de saida e verificadores de qualidade antes de ter
qualquer modelo local instalado.

A geracao real roda **uma vez**, na maquina de desenvolvimento, e o que vai para
o repositorio e o cache em `results/llm/cache/`. O container publicado le o cache
e nao precisa de modelo, chave de API ou internet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src import config, explain, models  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.llm import narrator, prompts  # noqa: E402
from src.llm.provider import CachedProvider, FakeProvider, OllamaProvider  # noqa: E402
from src.preprocessing import split_data  # noqa: E402

def resposta_falsa(template, contexto: dict) -> str:
    """Texto que respeita o contrato do template que o pediu.

    O modo de verificacao serve para exercitar o encanamento, nao para simular um
    modelo ruim. Se a resposta falsa fosse generica, tudo reprovaria sempre e o
    `--fake` deixaria de distinguir encanamento quebrado de texto ruim. Os
    verificadores sao testados contra defeitos na suite deles, que e onde essa
    garantia pertence.
    """
    features = [item["feature"] for item in contexto.get("maiores_contribuicoes", [])]
    citacao = ", ".join(features) if features else "nenhuma medida específica"

    corpo = {
        template.secoes[0]: "Texto de verificação, gerado sem modelo de linguagem.",
        template.secoes[1]: f"Medidas consideradas: {citacao}.",
        template.secoes[-1]: (
            "Esta leitura é apoio à decisão e não substitui a avaliação do "
            "médico responsável."
        ),
    }
    blocos = [f"## {secao}\n{texto}" for secao, texto in corpo.items()]
    return "\n\n".join(blocos)


def selecionar_casos(quantidade: int) -> list[int]:
    """Indices dos casos explicados, deterministicos.

    Sortear a cada execucao tornaria o artefato irreprodutivel, e o relatorio
    cita casos especificos. Os primeiros indices do conjunto de teste bastam: a
    escolha nao e analitica, e a estratificacao do split ja garante variedade.
    """
    return list(range(quantidade))


def evidencias_por_caso(indices: list[int]) -> list[dict]:
    """Probabilidade, limiar e contribuicoes SHAP de cada caso pedido.

    Usa o pipeline e o explicador da Fase 1, em vez de recalcular: a explicacao
    que o texto narra tem de ser a mesma que o grafico mostra, senao o paragrafo
    e a figura contariam historias diferentes sobre o mesmo caso.
    """
    X, y = load_dataset()
    X_train, X_avaliacao, y_train, _ = split_data(X, y)

    modelo = models.get_models()["Regressão Logística"].fit(X_train, y_train)
    probabilidades = modelo.predict_proba(X_avaliacao)[:, 1]

    explicacao = explain.shap_explanation(modelo, X_train, X_avaliacao)
    colunas = list(X_avaliacao.columns)

    evidencias = []
    for indice in indices:
        valores = np.asarray(explicacao[indice].values).ravel()
        evidencias.append(
            narrator.contexto_do_paciente(
                probabilidade=float(probabilidades[indice]),
                limiar=0.5,
                contribuicoes=list(zip(colunas, (float(v) for v in valores))),
            )
        )
    return evidencias


def _narrar(template, contexto: dict, provider):
    """Narra com o provedor dado, ou com um falso que respeita o contrato."""
    usado = provider or FakeProvider([resposta_falsa(template, contexto)])
    return narrator.narrar(template, contexto, usado)


def montar_narrativas(provider, casos: int) -> tuple[list, list]:
    """Gera o lote inteiro e a comparacao entre versoes de prompt."""
    narrativas = []

    contextos = evidencias_por_caso(selecionar_casos(casos))
    template_paciente = prompts.TEMPLATES["explicacao_paciente_v2"]
    for contexto in contextos:
        narrativas.append(_narrar(template_paciente, contexto, provider))

    metricas = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
    narrativas.append(
        _narrar(
            prompts.TEMPLATES["resumo_gestao_v1"],
            narrator.contexto_de_gestao(metricas),
            provider,
        )
    )

    # A nota de mudanca depende do summary do algoritmo genetico. Enquanto os
    # experimentos nao tiverem rodado, o produto simplesmente nao entra no lote,
    # em vez de entrar com numero inventado.
    resumo_ag = config.GENETIC_DIR / "summary.json"
    if resumo_ag.exists():
        narrativas.append(
            _narrar(
                prompts.TEMPLATES["nota_de_mudanca_v1"],
                narrator.contexto_de_mudanca(
                    json.loads(resumo_ag.read_text(encoding="utf-8"))
                ),
                provider,
            )
        )

    # Comparacao entre versoes do mesmo prompt, sobre o primeiro caso. E o que
    # transforma "aplicamos prompt engineering" em afirmacao medida.
    comparacao = []
    if contextos:
        for template in prompts.versoes_de("explicacao_paciente"):
            narrativa = _narrar(template, contextos[0], provider)
            comparacao.append(
                {
                    "identificador": narrativa.identificador,
                    "versao": narrativa.versao,
                    "aprovado": narrativa.qualidade.aprovado,
                    "verificacoes": {
                        nome: v.aprovado
                        for nome, v in narrativa.qualidade.verificacoes.items()
                    },
                }
            )

    return narrativas, comparacao


def main() -> None:
    parser = argparse.ArgumentParser(description="Geracao dos textos da Fase 2.")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="usa um provedor deterministico, para verificar sem Ollama instalado",
    )
    parser.add_argument("--casos", type=int, default=3)
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--comparacao", type=Path, default=None)
    argumentos = parser.parse_args()

    provider = None if argumentos.fake else CachedProvider(OllamaProvider())

    narrativas, comparacao = montar_narrativas(provider, argumentos.casos)

    destino = argumentos.saida or (config.LLM_DIR / "narrativas.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps([n.para_dicionario() for n in narrativas], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    destino_comparacao = argumentos.comparacao or (
        config.LLM_DIR / "comparacao_de_prompts.json"
    )
    destino_comparacao.parent.mkdir(parents=True, exist_ok=True)
    destino_comparacao.write_text(
        json.dumps(comparacao, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    aprovadas = sum(1 for n in narrativas if n.qualidade.aprovado)
    print(f"{len(narrativas)} textos gerados, {aprovadas} aprovados nas quatro verificacoes")
    for narrativa in narrativas:
        reprovou = [
            nome
            for nome, v in narrativa.qualidade.verificacoes.items()
            if not v.aprovado
        ]
        estado = "ok" if not reprovou else f"reprovou em {', '.join(reprovou)}"
        print(f"  {narrativa.identificador:<26} {estado}")
    print(f"Gravado em {destino}")


if __name__ == "__main__":
    main()
