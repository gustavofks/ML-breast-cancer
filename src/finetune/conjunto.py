"""Exemplos de treino, a regra de origem e a separacao sem vazamento.

Duas regras do spec viram verificacao em codigo, e nao convencao. Origem: todo
exemplo declara de onde veio, e so o professor e as fontes publicas sao aceitos
(clausula D.4 dos termos da Anthropic). Sistema: exemplo de treino nao tem
mensagem de sistema, porque ela seria texto escrito fora do professor; o
comportamento e aprendido pelos exemplos, e a instrucao de sistema existe so na
inferencia.

Vazamento: pergunta de treino quase igual a uma de teste sai do treino. Sem
isso, o teste mediria memoria da pergunta, e nao capacidade de responder.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src import config
from src.clinica.curadoria import quase_iguais

ORIGENS_PERMITIDAS = frozenset({config.FASE3_ORIGEM_PROFESSOR, "fonte:inca"})


@dataclass(frozen=True)
class Exemplo:
    tipo: str
    protocolo: str | None
    origem: str
    mensagens: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.origem not in ORIGENS_PERMITIDAS:
            raise ValueError(f"Origem nao permitida no treino: {self.origem}")
        if any(m["role"] == "system" for m in self.mensagens):
            raise ValueError("Exemplo de treino nao pode ter mensagem de sistema.")

    @property
    def pergunta(self) -> str:
        conteudo = self.mensagens[0]["content"]
        return conteudo.rsplit("PERGUNTA: ", 1)[-1].strip()


def mensagem_do_usuario(pergunta: str, bloco_trechos: str | None = None) -> str:
    if bloco_trechos is None:
        return f"PERGUNTA: {pergunta}"
    return f"TRECHOS:\n{bloco_trechos}\n\nPERGUNTA: {pergunta}"


def dividir(candidatos_treino: list[Exemplo], teste: list[Exemplo], *,
            ineditos: set[str], semente: int) -> dict:
    """Treino e validacao a partir dos candidatos; o teste chega pronto.

    Candidato de protocolo inedito vai para o teste, nunca para o treino.
    """
    teste = list(teste) + [e for e in candidatos_treino if e.protocolo in ineditos]
    restantes = [e for e in candidatos_treino if e.protocolo not in ineditos]
    limpos = [e for e in restantes
              if not any(quase_iguais(e.pergunta, t.pergunta) for t in teste)]

    embaralhados = list(limpos)
    random.Random(semente).shuffle(embaralhados)
    n_validacao = round(len(embaralhados) * config.FASE3_FRACAO_VALIDACAO)
    return {
        "treino": embaralhados[n_validacao:],
        "validacao": embaralhados[:n_validacao],
        "teste": teste,
        "vazamentos_removidos": len(restantes) - len(limpos),
    }


def salvar_jsonl(exemplos: list[Exemplo], caminho: Path) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="\n") as saida:
        for exemplo in exemplos:
            saida.write(json.dumps(asdict(exemplo), ensure_ascii=False, sort_keys=True) + "\n")


def carregar_jsonl(caminho: Path) -> list[Exemplo]:
    linhas = Path(caminho).read_text(encoding="utf-8").splitlines()
    return [Exemplo(**json.loads(linha)) for linha in linhas if linha.strip()]
