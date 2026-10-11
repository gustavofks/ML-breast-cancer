"""Provedores de texto, atras de um `Protocol`.

A camada inteira depende do `Protocol`, e nao de um cliente concreto. E isso que
permite a suite rodar sem modelo de linguagem instalado, e que torna a troca do
modelo local uma mudanca de configuracao.

O `CachedProvider` e o que viabiliza a publicacao em nuvem: o container le o cache
versionado e nao precisa de modelo, chave de API ou internet. A fronteira entre a
geracao, que roda na maquina de desenvolvimento, e o consumo, que roda em
qualquer lugar, e a garantia de reprodutibilidade desta camada.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from src import config


class CacheVazioError(RuntimeError):
    """Resposta ausente do cache sem provedor disponivel para gera-la."""


class OrcamentoEsgotado(RuntimeError):
    """A proxima chamada poderia passar do orcamento; ela nao foi feita."""


class GeracaoIncompleta(RuntimeError):
    """O modelo parou antes de terminar, ou recusou. O texto nao vai para o cache."""


@runtime_checkable
class Provider(Protocol):
    """Contrato minimo de geracao de texto."""

    def generate(self, prompt: str, *, system: str, seed: int) -> str:
        """Devolve o texto gerado para o par (system, prompt)."""
        ...


class FakeProvider:
    """Provedor deterministico, sem modelo nenhum por tras.

    Vive no codigo de producao, e nao apenas nos testes, porque
    `scripts/run_llm.py` tambem o usa em modo de verificacao: da para exercitar o
    lote inteiro, os prompts e os verificadores de qualidade sem Ollama
    instalado. Quando as respostas acabam, a ultima se repete, para que o lote
    nao quebre por causa do tamanho da lista.
    """

    def __init__(self, respostas: list[str]) -> None:
        self._respostas = list(respostas)
        self.chamadas = 0

    def generate(self, prompt: str, *, system: str, seed: int) -> str:
        indice = min(self.chamadas, len(self._respostas) - 1)
        self.chamadas += 1
        return self._respostas[indice]


class OllamaProvider:
    """Fala HTTP com o servidor local do Ollama, com temperatura zero.

    O `import requests` fica dentro do metodo de proposito: o resto da camada e a
    suite inteira funcionam sem a dependencia instalada, e so quem de fato chama
    o modelo precisa dela.
    """

    def __init__(
        self,
        modelo: str = config.OLLAMA_MODELO,
        url: str = config.LLM_URL,
        timeout: int = config.LLM_TIMEOUT,
    ) -> None:
        self.modelo = modelo
        self.url = url
        self.timeout = timeout

    def generate(self, prompt: str, *, system: str, seed: int) -> str:
        import requests

        resposta = requests.post(
            self.url,
            json={
                "model": self.modelo,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": config.LLM_TEMPERATURA, "seed": seed},
            },
            timeout=self.timeout,
        )
        resposta.raise_for_status()
        return resposta.json()["response"].strip()


def pior_caso_usd(
    prompt: str,
    system: str,
    *,
    max_tokens: int = config.CLAUDE_MAX_TOKENS,
    preco_entrada_usd: float = config.CLAUDE_PRECO_ENTRADA_USD,
    preco_saida_usd: float = config.CLAUDE_PRECO_SAIDA_USD,
) -> float:
    """Custo maximo de uma chamada, estimado por cima.

    Um token a cada dois caracteres superestima de proposito: portugues fica
    perto de tres caracteres por token neste tokenizador. A saida entra pelo teto
    inteiro, que ja inclui o pensamento. Funcao pura, sem cliente: a estimativa
    do lote roda sem chave de API.
    """
    tokens_entrada = len(prompt + system) // 2 + 1
    return (tokens_entrada * preco_entrada_usd + max_tokens * preco_saida_usd) / 1_000_000


class ClaudeProvider:
    """Fala com a API do Claude, com o custo travado por construcao.

    Tres travas vivem aqui. Esforco baixo, porque o pensamento adaptativo do
    Haiku 5.5 e cobrado como saida. Teto de tokens por chamada. E um orcamento
    total conferido **antes** de cada chamada: se o pior caso da proxima nao
    couber no que sobra, ela nao sai. A quarta trava e o `CachedProvider`, que
    nunca paga duas vezes pelo mesmo texto.

    Nao envia `temperature`: o Haiku 5.5 recusa parametro de amostragem com 400.
    Tambem nao ha semente na API. A reprodutibilidade vem do cache versionado — o
    texto publicado e o que ficou gravado, como o `metrics.json` da Fase 1.

    O `import anthropic` fica no construtor, como o `requests` do Ollama: a suite
    roda sem o SDK, injetando um cliente falso.
    """

    def __init__(
        self,
        cliente=None,
        *,
        modelo: str = config.LLM_MODELO,
        esforco: str = config.CLAUDE_ESFORCO,
        max_tokens: int = config.CLAUDE_MAX_TOKENS,
        orcamento_usd: float = config.LLM_ORCAMENTO_USD,
        preco_entrada_usd: float = config.CLAUDE_PRECO_ENTRADA_USD,
        preco_saida_usd: float = config.CLAUDE_PRECO_SAIDA_USD,
    ) -> None:
        if cliente is None:
            import anthropic

            cliente = anthropic.Anthropic()
        self._cliente = cliente
        self.modelo = modelo
        self.esforco = esforco
        self.max_tokens = max_tokens
        self.orcamento_usd = orcamento_usd
        self._preco_entrada = preco_entrada_usd
        self._preco_saida = preco_saida_usd
        self.gasto_usd = 0.0
        self.ultimo_uso: dict | None = None

    def custo(self, tokens_entrada: int, tokens_saida: int) -> float:
        return (
            tokens_entrada * self._preco_entrada + tokens_saida * self._preco_saida
        ) / 1_000_000

    def pior_caso(self, prompt: str, system: str) -> float:
        return pior_caso_usd(
            prompt,
            system,
            max_tokens=self.max_tokens,
            preco_entrada_usd=self._preco_entrada,
            preco_saida_usd=self._preco_saida,
        )

    def generate(self, prompt: str, *, system: str, seed: int) -> str:
        restante = self.orcamento_usd - self.gasto_usd
        if self.pior_caso(prompt, system) > restante:
            raise OrcamentoEsgotado(
                f"gasto US$ {self.gasto_usd:.4f} de US$ {self.orcamento_usd:.2f}; "
                "a proxima chamada poderia passar do teto e nao foi feita"
            )

        resposta = self._cliente.messages.create(
            model=self.modelo,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.esforco},
        )

        # Cobrado mesmo quando o texto nao serve: o gasto e registrado antes de
        # qualquer verificacao.
        uso = resposta.usage
        custo = self.custo(uso.input_tokens, uso.output_tokens)
        self.gasto_usd += custo
        self.ultimo_uso = {
            "modelo": self.modelo,
            "tokens_entrada": uso.input_tokens,
            "tokens_saida": uso.output_tokens,
            "custo_usd": round(custo, 8),
        }

        # Texto cortado ou recusado nao pode chegar ao cache: seria servido para
        # sempre, sem que ninguem soubesse que esta incompleto.
        if resposta.stop_reason in ("max_tokens", "refusal"):
            raise GeracaoIncompleta(f"geracao interrompida: {resposta.stop_reason}")
        texto = "".join(b.text for b in resposta.content if b.type == "text").strip()
        if not texto:
            raise GeracaoIncompleta("resposta sem texto")
        return texto


def custo_do_cache(diretorio: Path = config.LLM_CACHE_DIR) -> dict:
    """Custo total do cache versionado, somado das entradas.

    O custo vive em cada entrada, gravado no momento da geracao, entao o total
    sai do artefato, e nao de um contador que zera a cada execucao. E o numero
    que o relatorio cita.
    """
    total = {"textos": 0, "tokens_entrada": 0, "tokens_saida": 0, "custo_usd": 0.0}
    for caminho in sorted(Path(diretorio).glob("*.json")):
        uso = json.loads(caminho.read_text(encoding="utf-8")).get("uso")
        total["textos"] += 1
        if uso:
            total["tokens_entrada"] += uso["tokens_entrada"]
            total["tokens_saida"] += uso["tokens_saida"]
            total["custo_usd"] += uso["custo_usd"]
    total["custo_usd"] = round(total["custo_usd"], 6)
    return total


class CachedProvider:
    """Embrulha qualquer provedor e persiste as respostas em disco.

    A chave e o hash de modelo, prompt de sistema, prompt do usuario e parametros
    de geracao, **tomados em conjunto**. Mudou o prompt, muda a chave, e a
    resposta antiga nao e reaproveitada por engano — o que seria um defeito
    silencioso: o relatorio citaria a versao nova do prompt exibindo o texto
    produzido pela versao velha.
    """

    def __init__(
        self,
        interno: Provider | None,
        diretorio: Path = config.LLM_CACHE_DIR,
        *,
        modelo: str = config.LLM_MODELO,
    ) -> None:
        self._interno = interno
        self._diretorio = Path(diretorio)
        self._modelo = modelo

    def chave(self, prompt: str, *, system: str, seed: int) -> str:
        """Identidade da geracao, em hexadecimal."""
        material = json.dumps(
            {
                "modelo": self._modelo,
                "system": system,
                "prompt": prompt,
                "seed": seed,
                "temperatura": config.LLM_TEMPERATURA,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    def generate(self, prompt: str, *, system: str, seed: int) -> str:
        chave = self.chave(prompt, system=system, seed=seed)
        caminho = self._diretorio / f"{chave}.json"

        if caminho.exists():
            return json.loads(caminho.read_text(encoding="utf-8"))["resposta"]

        if self._interno is None:
            raise CacheVazioError(
                f"Resposta ausente do cache ({chave}) e nenhum provedor disponivel. "
                "Rode `python scripts/run_llm.py` na maquina de desenvolvimento, "
                "com a chave da API do Claude no ambiente, para gerar e versionar "
                "o cache."
            )

        resposta = self._interno.generate(prompt, system=system, seed=seed)
        self._diretorio.mkdir(parents=True, exist_ok=True)
        caminho.write_text(
            json.dumps(
                {
                    "modelo": self._modelo,
                    "system": system,
                    "prompt": prompt,
                    "resposta": resposta,
                    # Uso e custo, quando o provedor os informa. E o que permite
                    # somar o custo do lote a partir do proprio artefato.
                    "uso": getattr(self._interno, "ultimo_uso", None),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return resposta
