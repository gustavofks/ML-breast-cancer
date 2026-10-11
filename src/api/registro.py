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
from datetime import datetime, timezone
from typing import IO, Any


class FormatoJson(logging.Formatter):
    """Serializa os campos do evento, mais nivel e momento em UTC."""

    def format(self, record: logging.LogRecord) -> str:
        campos = dict(getattr(record, "campos", {}))
        campos["nivel"] = record.levelname
        campos["momento"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        return json.dumps(campos, ensure_ascii=False, sort_keys=True)


def configurar(nome: str = "api", fluxo: IO[str] | None = None) -> logging.Logger:
    """Logger com um unico handler JSON.

    Idempotente: cada `criar_app` chama isto, e um handler a mais duplicaria
    cada linha do log.
    """
    logger = logging.getLogger(nome)
    logger.handlers.clear()
    handler = logging.StreamHandler(fluxo or sys.stdout)
    handler.setFormatter(FormatoJson())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


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
