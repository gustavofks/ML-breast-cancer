"""Registro de auditoria do assistente: um evento JSON por linha, por execucao.

Cada passo do fluxo e cada pergunta livre deixam um evento: execucao, no,
horario e campos resumidos (ids de trechos, contagens, resultado das guardas,
decisao do medico). O texto do prontuario nao entra, so o pseudonimo da
paciente, e isso e verificado na gravacao: o registro pode ser lido pela API e
pela vitrine sem expor mais do que o banco anonimizado ja expoe.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

CAMPOS_PROIBIDOS = frozenset({"contexto", "contexto_paciente", "historico", "evolucoes"})
_PSEUDONIMO = re.compile(r"^PACIENTE-\d{4}$")


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Auditoria:
    def __init__(self, caminho: Path, *, relogio: Callable[[], str] = _agora) -> None:
        self.caminho = Path(caminho)
        self._relogio = relogio
        self._trava = threading.Lock()

    def registrar(self, execucao: str, no: str, **campos) -> dict:
        proibidos = CAMPOS_PROIBIDOS & set(campos)
        if proibidos:
            raise ValueError(f"texto do prontuario na auditoria: {sorted(proibidos)}")
        if "paciente" in campos and not _PSEUDONIMO.match(str(campos["paciente"])):
            raise ValueError("paciente na auditoria so por pseudonimo")
        evento = {"execucao": execucao, "no": no, "horario": self._relogio(), **campos}
        with self._trava:
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
            with self.caminho.open("a", encoding="utf-8", newline="\n") as arquivo:
                arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
        return evento

    def ler(self, execucao: str) -> list[dict]:
        if not self.caminho.exists():
            return []
        eventos = (json.loads(linha) for linha in self.caminho.read_text(encoding="utf-8").splitlines()
                   if linha.strip())
        return [e for e in eventos if e["execucao"] == execucao]
