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
        modelo: str = config.LLM_MODELO,
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
                "com o Ollama no ar, para gerar e versionar o cache."
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
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return resposta
