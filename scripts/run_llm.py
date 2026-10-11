"""Gera os textos em linguagem natural e grava `results/llm/`.

Uso, a partir da raiz do projeto:

    python scripts/run_llm.py --fake                 # encanamento, sem modelo nenhum
    python scripts/run_llm.py --casos todos --estimar  # quanto pode custar, sem gastar
    python scripts/run_llm.py --casos todos          # geracao real com o Claude

O modo `--fake` existe porque a camada inteira fala com um `Protocol`: da para
exercitar prompts, contratos de saida e verificadores de qualidade sem modelo.

A geracao real roda **uma vez**, na maquina de desenvolvimento, com a chave em
`ANTHROPIC_API_KEY`, e o que vai para o repositorio e o cache em
`results/llm/cache/`. O container publicado le o cache e nao precisa de modelo,
chave de API ou internet.

Antes de qualquer chamada paga, o lote inteiro e estimado pelo pior caso. Se a
estimativa passar do orcamento de `config.LLM_ORCAMENTO_USD`, o script para sem
gastar nada: parar no meio ainda teria gasto.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.api import explicacao  # noqa: E402
from src.api.servico import Servico, carregar_especificacao  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.llm import narrator, prompts  # noqa: E402
from src.llm.provider import (  # noqa: E402
    CachedProvider,
    CacheVazioError,
    ClaudeProvider,
    FakeProvider,
    OllamaProvider,
    custo_do_cache,
    pior_caso_usd,
)
from src.preprocessing import split_data  # noqa: E402


class Estimador:
    """Provedor que nao gera nada: soma o pior caso do que o cache ainda nao tem.

    Consulta o cache pela mesma chave que a geracao real usaria, entao texto ja
    gerado entra como custo zero — e e: o `CachedProvider` nunca paga de novo.
    Nao grava nada.
    """

    def __init__(self, diretorio: Path, modelo: str) -> None:
        self._cache = CachedProvider(None, diretorio, modelo=modelo)
        self.chamadas = 0
        self.pior_caso_usd = 0.0

    def generate(self, prompt: str, *, system: str, seed: int) -> str:
        try:
            return self._cache.generate(prompt, system=system, seed=seed)
        except CacheVazioError:
            self.chamadas += 1
            self.pior_caso_usd += pior_caso_usd(prompt, system)
            return ""


def quantidade_de_casos(valor: str) -> int:
    """Numero de casos pedido, ou o conjunto de teste inteiro com `todos`.

    O painel deixa escolher qualquer caso do teste, e caso sem texto aparece como
    "texto indisponivel". Com `todos`, nenhum fica de fora.
    """
    if valor == "todos":
        X, y = load_dataset()
        _, X_teste, _, _ = split_data(X, y)
        return len(X_teste)
    return int(valor)

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
    """Probabilidade, limiar e contribuicoes de cada caso, pelo modelo servido.

    Mesma especificacao, mesma previsao e mesma funcao de contribuicoes que a API
    usa. O cache gravado aqui so e encontrado la se as duas pontas fizerem a
    mesma conta, ate o arredondamento: qualquer divergencia mudaria o prompt, e
    com ele a chave, e o /explain responderia "texto indisponivel" para sempre,
    sem erro nenhum.
    """
    modelo = Servico.construir(carregar_especificacao())
    X, y = load_dataset()
    _, X_avaliacao, _, _ = split_data(X, y)

    evidencias = []
    for indice in indices:
        caso = X_avaliacao.iloc[[indice]]
        previsao = modelo.prever(caso.iloc[0].to_dict())
        evidencias.append(
            narrator.contexto_do_paciente(
                probabilidade=previsao.probabilidade,
                limiar=previsao.limiar,
                contribuicoes=explicacao.contribuicoes(modelo, caso),
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
        help="usa um provedor deterministico, para verificar sem modelo nenhum",
    )
    parser.add_argument(
        "--provedor",
        choices=("claude", "ollama"),
        default="claude",
        help="claude (padrao, pago e com orcamento) ou ollama (local e gratuito)",
    )
    parser.add_argument(
        "--estimar",
        action="store_true",
        help="so mostra o pior caso de custo; nao chama a API nem grava nada",
    )
    parser.add_argument("--casos", default="3", help="numero de casos, ou `todos`")
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--comparacao", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=config.LLM_CACHE_DIR)
    parser.add_argument("--custo", type=Path, default=None)
    argumentos = parser.parse_args()

    casos = quantidade_de_casos(argumentos.casos)
    claude = argumentos.provedor == "claude"
    modelo = config.LLM_MODELO if claude else config.OLLAMA_MODELO

    provider = None
    if not argumentos.fake:
        estimador = Estimador(argumentos.cache, modelo)
        montar_narrativas(estimador, casos)
        print(
            f"{estimador.chamadas} textos a gerar com {modelo}; pior caso "
            f"US$ {estimador.pior_caso_usd:.4f} (orcamento US$ {config.LLM_ORCAMENTO_USD:.2f})"
        )
        if argumentos.estimar:
            return
        # O lote inteiro e barrado antes da primeira chamada. Parar no meio, pela
        # trava do provedor, ainda teria gasto o que veio antes.
        if claude and estimador.pior_caso_usd > config.LLM_ORCAMENTO_USD:
            sys.exit(
                "O pior caso passa do orcamento. Reduza --casos ou ajuste "
                "LLM_ORCAMENTO_USD em src/config.py; nada foi gasto."
            )
        interno = (
            ClaudeProvider(orcamento_usd=config.LLM_ORCAMENTO_USD) if claude else OllamaProvider()
        )
        provider = CachedProvider(interno, argumentos.cache, modelo=modelo)

    narrativas, comparacao = montar_narrativas(provider, casos)

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

    if provider is not None and claude:
        # O custo gravado sai do cache, e nao do contador desta execucao: rodar
        # de novo nao gasta nada, e o arquivo continua dizendo quanto o lote
        # custou. E o numero que o relatorio cita.
        custo = {
            "modelo": modelo,
            "orcamento_usd": config.LLM_ORCAMENTO_USD,
            **custo_do_cache(argumentos.cache),
        }
        destino_custo = argumentos.custo or (config.LLM_DIR / "custo.json")
        destino_custo.write_text(json.dumps(custo, indent=2), encoding="utf-8")
        print(
            f"Gasto desta execucao: US$ {interno.gasto_usd:.4f}. "
            f"Custo total do cache: US$ {custo['custo_usd']:.4f} em {custo['textos']} textos."
        )


if __name__ == "__main__":
    main()
