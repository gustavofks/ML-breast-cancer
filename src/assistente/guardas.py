"""Guardas: a verificacao da resposta que nao depende do modelo.

Camada final da defesa (spec, secao 8). O comportamento vem do treino, e a
guarda confere o resultado em codigo: dose, verbo de prescricao, diagnostico
definitivo, citacao ausente, citacao de trecho que nao foi recuperado e frase
citada sem apoio no trecho. Resposta reprovada nao chega ao medico como
sugestao: a cadeia a retem, e o fluxo da semana 5 a transforma em alerta.

Negacao conta por frase: "nao posso prescrever" e recusa correta, e a primeira
curadoria barrou 15 recusas boas por olhar so o verbo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src import config
from src.clinica import curadoria
from src.corpus.protocolos import CITACAO, Trecho, citacoes_em

_MEDICAMENTO = (r"(medicamentos?|rem[eé]dios?|comprimidos?|doses?|antibi[oó]tico|analg[eé]sico|"
                r"dipirona|paracetamol|ibuprofeno|amoxicilina|tamoxifeno|anti-?inflamat[oó]rio)")
_PRESCRICAO = re.compile(
    r"\bprescrevo\b|\bvou prescrever\b"
    rf"|\b(tome|tomar|administre|administrar|aplique|inicie|iniciar|use|usar)\b.{{0,40}}\b{_MEDICAMENTO}"
    r"|\b(recomendo|indico|sugiro)\b.{0,20}\b(o uso d[eo]|tomar|usar|iniciar)\b",
    re.I)
_DIAGNOSTICO = re.compile(
    r"\b(voc[eê]|a paciente|ela)\s+(tem|possui|est[aá] com)\s+(um\s+)?(c[aâ]ncer|carcinoma|tumor maligno)\b"
    r"|\b[eé]\s+(um\s+|uma\s+)?(carcinoma|tumor maligno|les[aã]o maligna)\b",
    re.I)
_NEGACAO = re.compile(r"\b(n[aã]o|nunca|nem)\b", re.I)
# Recusa ou "nao encontrei": resposta sem afirmacao clinica, que nao precisa de citacao.
_SEM_AFIRMACAO = re.compile(
    r"\bn[aã]o (posso|consigo|encontrei|tenho|h[aá]|consta|abordam?|tratam?|mencionam?|cont[eé]m|trazem?)\b",
    re.I)
_FRASE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Verificacao:
    motivos: tuple[str, ...]

    @property
    def aprovado(self) -> bool:
        return not self.motivos


def _frases(texto: str) -> list[str]:
    """Frases do texto; citacao solta depois do ponto ("... especial. [PROT-10 v1 §4]")
    volta para a frase anterior, que e a que ela apoia."""
    frases: list[str] = []
    for parte in (f.strip() for f in _FRASE.split(texto)):
        if not parte:
            continue
        if frases and not CITACAO.sub("", parte).strip(" .;"):
            frases[-1] = f"{frases[-1]} {parte}"
        else:
            frases.append(parte)
    return frases


def _afirmativo(padrao: re.Pattern, frase: str) -> bool:
    achado = padrao.search(frase)
    return bool(achado) and not _NEGACAO.search(frase[:achado.start()])


def verificar(texto: str, recuperados: list[Trecho], *, cortada: bool = False) -> Verificacao:
    motivos: list[str] = []
    if cortada:
        motivos.append("resposta cortada no limite de tokens")
    if curadoria.tem_posologia(texto):
        motivos.append("menciona dose ou posologia")
    frases = _frases(texto)
    if any(_afirmativo(_PRESCRICAO, f) for f in frases):
        motivos.append("orienta uso de medicamento")
    if any(_afirmativo(_DIAGNOSTICO, f) for f in frases):
        motivos.append("afirma diagnóstico definitivo")

    por_id = {t.id: t for t in recuperados}
    citadas = citacoes_em(texto)
    for id_citado in citadas:
        if id_citado not in por_id:
            motivos.append(f"cita trecho não recuperado: {id_citado}")
    if recuperados and not citadas and not _SEM_AFIRMACAO.search(texto):
        motivos.append("afirmação sem citação de trecho")
    for frase in frases:
        ids = [i for i in citacoes_em(frase) if i in por_id]
        if ids:
            fonte = "\n".join(por_id[i].texto for i in ids)
            if curadoria.suporte_lexical(CITACAO.sub(" ", frase), fonte) < config.FASE3_SUPORTE_MINIMO:
                motivos.append(f"frase sem apoio no trecho citado: {frase[:80]}")
    return Verificacao(tuple(motivos))
