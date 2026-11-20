"""Banco de prontuarios da clinica ficticia: SQLite e consultas fixas.

O assistente nunca escreve SQL. Ele escolhe uma destas funcoes, e o codigo faz a
consulta, sempre parametrizada e sempre filtrada por uma paciente. Assim o modelo
nao le outra paciente nem monta consulta a partir de texto injetado no prontuario.
A conexao entregue ao assistente e aberta so para leitura, e quem garante isso e
o proprio SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ESQUEMA = """
CREATE TABLE pacientes (paciente TEXT PRIMARY KEY, idade INTEGER, historico TEXT, origem TEXT);
CREATE TABLE exames (
    id INTEGER PRIMARY KEY, paciente TEXT REFERENCES pacientes(paciente), exame TEXT,
    solicitado_em TEXT, resultado_em TEXT, resultado TEXT, medidas TEXT, origem TEXT
);
CREATE TABLE evolucoes (
    id INTEGER PRIMARY KEY, paciente TEXT REFERENCES pacientes(paciente),
    data TEXT, texto TEXT, origem TEXT
);
"""


def criar(caminho: Path, registros: list[dict]) -> None:
    """Recria o banco do zero a partir dos registros anonimizados."""
    caminho = Path(caminho)
    caminho.unlink(missing_ok=True)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(caminho)
    with con:
        con.executescript(ESQUEMA)
        for r in registros:
            if r["tipo"] == "paciente":
                con.execute("INSERT INTO pacientes VALUES (?, ?, ?, ?)",
                            (r["paciente"], r["idade"], r["historico"], r["origem"]))
            elif r["tipo"] == "exame":
                con.execute(
                    "INSERT INTO exames (paciente, exame, solicitado_em, resultado_em, resultado,"
                    " medidas, origem) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (r["paciente"], r["exame"], r["solicitado_em"], r["resultado_em"],
                     r["resultado"], json.dumps(r["medidas"]) if r["medidas"] else None,
                     r["origem"]),
                )
            elif r["tipo"] == "evolucao":
                con.execute("INSERT INTO evolucoes (paciente, data, texto, origem) VALUES (?, ?, ?, ?)",
                            (r["paciente"], r["data"], r["texto"], r["origem"]))
    con.close()


def conectar(caminho: Path) -> sqlite3.Connection:
    """Conexao so de leitura, compartilhavel entre threads.

    A API atende cada rota numa thread do pool e o LangGraph roda os nos em
    threads proprias, entao a conexao aberta na montagem e usada em outra
    thread. Sem escrita, compartilhar e seguro: o modulo sqlite3 serializa as
    chamadas, e nada muda o banco por esta conexao.
    """
    con = sqlite3.connect(f"file:{Path(caminho).as_posix()}?mode=ro", uri=True,
                          check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _exame(linha: sqlite3.Row) -> dict:
    exame = dict(linha)
    exame["medidas"] = json.loads(exame["medidas"]) if exame["medidas"] else None
    return exame


def dados_paciente(con: sqlite3.Connection, paciente: str) -> dict | None:
    linha = con.execute("SELECT * FROM pacientes WHERE paciente = ?", (paciente,)).fetchone()
    return dict(linha) if linha else None


def exames(con: sqlite3.Connection, paciente: str) -> list[dict]:
    linhas = con.execute(
        "SELECT * FROM exames WHERE paciente = ? ORDER BY solicitado_em", (paciente,)
    ).fetchall()
    return [_exame(linha) for linha in linhas]


def evolucoes(con: sqlite3.Connection, paciente: str) -> list[dict]:
    linhas = con.execute(
        "SELECT * FROM evolucoes WHERE paciente = ? ORDER BY data", (paciente,)
    ).fetchall()
    return [dict(linha) for linha in linhas]


def pendencias(con: sqlite3.Connection, paciente: str, *, hoje: str, prazo_dias: int) -> list[dict]:
    limite = (date.fromisoformat(hoje) - timedelta(days=prazo_dias)).isoformat()
    linhas = con.execute(
        "SELECT * FROM exames WHERE paciente = ? AND resultado IS NULL AND solicitado_em < ?"
        " ORDER BY solicitado_em",
        (paciente, limite),
    ).fetchall()
    return [_exame(linha) for linha in linhas]
