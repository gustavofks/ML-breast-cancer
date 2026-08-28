"""Extracao e realce dos trechos de codigo exibidos no relatorio HTML.

O relatorio explica o que foi feito mostrando o codigo que faz. Se esses
trechos fossem copiados a mao para dentro de `src/report.py`, envelheceriam na
primeira refatoracao e a pagina passaria a descrever um pipeline que nao existe
mais. Aqui eles sao lidos do arquivo-fonte a cada geracao, pela mesma razao que
nenhuma metrica e escrita a mao: a pagina e um retrato do projeto, nao uma
copia dele.

A docstring da funcao e removida do recorte. Ela ja foi traduzida na prosa da
secao; repeti-la dentro do bloco de codigo dobraria o tamanho do trecho sem
acrescentar informacao.

O realce usa apenas a biblioteca padrao (`tokenize`), porque a pagina precisa
abrir por duplo clique, sem servidor, sem rede e sem dependencia extra.
"""

from __future__ import annotations

import ast
import io
import keyword
import textwrap
import tokenize
from functools import lru_cache
from html import escape
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Palavras que sao nomes comuns em Python mas nao palavras reservadas. Realcar
# `self` e os tipos embutidos deixa a leitura mais proxima da do editor.
EMBUTIDOS = frozenset(
    {
        "self",
        "None",
        "True",
        "False",
        "int",
        "str",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "len",
        "range",
        "sorted",
        "enumerate",
        "zip",
        "print",
        "isinstance",
        "round",
        "max",
        "min",
        "sum",
        "next",
        "super",
        "type",
    }
)


class TrechoNaoEncontrado(LookupError):
    """Nome pedido nao existe no modulo — erra na geracao, nao na pagina."""


@lru_cache(maxsize=None)
def _arquivo(modulo: str) -> tuple[str, ast.Module]:
    caminho = RAIZ / modulo
    fonte = caminho.read_text(encoding="utf-8")
    return fonte, ast.parse(fonte)


def _sem_docstring(no: ast.AST, linhas: list[str], base: int) -> list[str]:
    """Remove as linhas da docstring do recorte, se houver uma."""
    corpo = getattr(no, "body", None)
    if not corpo:
        return linhas

    primeiro = corpo[0]
    e_docstring = (
        isinstance(primeiro, ast.Expr)
        and isinstance(primeiro.value, ast.Constant)
        and isinstance(primeiro.value.value, str)
    )
    if not e_docstring:
        return linhas

    inicio = primeiro.lineno - base
    fim = primeiro.end_lineno - base
    restante = linhas[:inicio] + linhas[fim + 1 :]

    # A linha em branco que separava a docstring do corpo perde a funcao.
    while len(restante) > 1 and not restante[1].strip():
        del restante[1]
    return restante


def fonte(modulo: str, *nomes: str) -> str:
    """Devolve o codigo das funcoes pedidas, sem docstring e sem indentacao.

    Args:
        modulo: caminho relativo a raiz do repositorio, como `src/models.py`.
        nomes: nomes de funcoes ou classes definidas no modulo, na ordem em que
            devem aparecer no bloco.

    Raises:
        TrechoNaoEncontrado: se algum nome nao existir no modulo. Falhar aqui e
            proposital: um trecho renomeado quebra a geracao do relatorio em vez
            de sumir silenciosamente da pagina.
    """
    texto, arvore = _arquivo(modulo)
    todas = texto.splitlines()

    definicoes = {
        no.name: no
        for no in arvore.body
        if isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    blocos = []
    for nome in nomes:
        no = definicoes.get(nome)
        if no is None:
            raise TrechoNaoEncontrado(f"{modulo} nao define {nome!r}")

        recorte = todas[no.lineno - 1 : no.end_lineno]
        recorte = _sem_docstring(no, recorte, no.lineno)
        while recorte and not recorte[-1].strip():
            recorte.pop()
        blocos.append(textwrap.dedent("\n".join(recorte)))

    return "\n\n\n".join(blocos)


def destacar(codigo: str) -> str:
    """Marca palavras reservadas, textos, numeros e comentarios com `<span>`.

    Percorre os tokens e reemite o codigo preservando o espaco original entre
    eles, para que a indentacao continue exata. Se a tokenizacao falhar — codigo
    incompleto, por exemplo — devolve o codigo apenas escapado: a pagina perde a
    cor, nunca o conteudo.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(codigo).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return escape(codigo)

    linhas = codigo.splitlines(keepends=True)
    saida: list[str] = []
    linha_atual, coluna_atual = 1, 0
    apos_def = False

    def _vao(ate_linha: int, ate_coluna: int) -> None:
        """Reemite o texto cru entre o fim do token anterior e o proximo."""
        nonlocal linha_atual, coluna_atual
        while (linha_atual, coluna_atual) < (ate_linha, ate_coluna):
            if linha_atual > len(linhas):
                return
            linha = linhas[linha_atual - 1]
            if linha_atual < ate_linha:
                saida.append(escape(linha[coluna_atual:]))
                linha_atual += 1
                coluna_atual = 0
            else:
                saida.append(escape(linha[coluna_atual:ate_coluna]))
                coluna_atual = ate_coluna

    for tipo, texto, (linha_inicio, coluna_inicio), (linha_fim, coluna_fim), _ in tokens:
        if tipo in (tokenize.ENCODING, tokenize.ENDMARKER):
            continue

        _vao(linha_inicio, coluna_inicio)

        classe = ""
        if tipo == tokenize.COMMENT:
            classe = "c"
        elif tipo == tokenize.STRING:
            classe = "s"
        elif tipo == tokenize.NUMBER:
            classe = "n"
        elif tipo == tokenize.NAME:
            if apos_def:
                classe = "d"
            elif keyword.iskeyword(texto):
                classe = "k"
            elif texto in EMBUTIDOS:
                classe = "b"

        apos_def = tipo == tokenize.NAME and texto in ("def", "class")

        marcado = escape(texto)
        saida.append(f'<span class="{classe}">{marcado}</span>' if classe else marcado)
        linha_atual, coluna_atual = linha_fim, coluna_fim

    return "".join(saida)


def bloco(modulo: str, *nomes: str) -> str:
    """Codigo das funcoes pedidas, ja escapado e realcado para o HTML."""
    return destacar(fonte(modulo, *nomes))
