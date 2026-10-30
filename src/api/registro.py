"""Log estruturado em JSON, uma linha por evento.

Uma linha por evento, em JSON, porque e o que qualquer agregador de log le sem
configuracao. As entradas da API sao medidas morfologicas anonimas, e cada linha
**declara** isso no campo `dado_identificavel`: a secao 6.2 do design pede que a
ausencia seja afirmada, nao presumida.

O modulo se chama `registro`, e nao `logging` como na secao 3.1 do design: um
modulo `logging` dentro do pacote sombreia o da biblioteca padrao para quem o
importar de forma relativa, e o nome nao ganha nada com esse risco.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from typing import IO, Any

from src import config


class FormatoJson(logging.Formatter):
    """Serializa os campos do evento, mais nivel e momento em UTC."""

    def format(self, record: logging.LogRecord) -> str:
        campos = dict(getattr(record, "campos", {}))
        campos["nivel"] = record.levelname
        campos["momento"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        return json.dumps(campos, ensure_ascii=False, sort_keys=True)


class Memoria(logging.Handler):
    """Os eventos mais recentes, em memoria, numerados em ordem de chegada.

    Existe para a vitrine mostrar o log na tela, sem acesso ao stdout do
    container. A capacidade e fixa: uma memoria sem limite cresceria a cada
    requisicao. A sequencia nunca reinicia, entao quem le pede so o que veio
    depois do ultimo numero que viu, mesmo que eventos antigos ja tenham saido.

    O evento guardado e o mesmo JSON da linha do fluxo, relido: a tela e o
    agregador de log nunca divergem.
    """

    def __init__(self, capacidade: int) -> None:
        super().__init__()
        self.setFormatter(FormatoJson())
        self._eventos: deque[dict[str, Any]] = deque(maxlen=capacidade)
        self._seq = 0
        self._trava = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        evento = json.loads(self.format(record))
        with self._trava:
            self._seq += 1
            self._eventos.append({"seq": self._seq, **evento})

    def depois(self, seq: int) -> list[dict[str, Any]]:
        with self._trava:
            return [evento for evento in self._eventos if evento["seq"] > seq]


def configurar(
    nome: str = "api",
    fluxo: IO[str] | None = None,
    *,
    capacidade: int = config.API_EVENTOS_EM_MEMORIA,
) -> logging.Logger:
    """Logger com um handler JSON no fluxo e uma memoria dos eventos recentes.

    Idempotente: cada `criar_app` chama isto, e um handler a mais duplicaria
    cada linha do log.
    """
    logger = logging.getLogger(nome)
    logger.handlers.clear()
    handler = logging.StreamHandler(fluxo or sys.stdout)
    handler.setFormatter(FormatoJson())
    logger.addHandler(handler)
    logger.addHandler(Memoria(capacidade))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def memoria(logger: logging.Logger) -> Memoria:
    """A memoria de eventos que `configurar` prendeu ao logger."""
    return next(h for h in logger.handlers if isinstance(h, Memoria))


def _emitir(logger: logging.Logger, evento: str, nivel: int, campos: dict[str, Any]) -> None:
    logger.log(
        nivel,
        evento,
        extra={"campos": {**campos, "evento": evento, "dado_identificavel": False}},
    )


def requisicao(logger: logging.Logger, **campos: Any) -> None:
    """Uma requisicao atendida: id, rota, metodo, status, latencia, resultado."""
    _emitir(logger, "requisicao", logging.INFO, campos)


def desvio(logger: logging.Logger, **campos: Any) -> None:
    """Uma medida fora da distribuicao do treino, em nivel de aviso."""
    _emitir(logger, "desvio", logging.WARNING, campos)
