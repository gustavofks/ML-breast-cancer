"""Fontes publicas dos protocolos e o mapa tema -> paginas.

Os PDFs do INCA ficam fora do Git, porque o repositorio do INCA declara
direitos reservados; aqui ficam so a URL, o hash e as paginas usadas. Os
protocolos do hospital ficticio sao reescritos pelo modelo professor a partir
dessas paginas, nunca copiados.

O protocolo interno (PROT-13) nao tem fonte externa: seus numeros vem da
configuracao, os mesmos que o banco usa para calcular pendencias. Sao
parametros de funcionamento do sistema, nao conteudo clinico, e o relatorio
declara isso.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from src import config


@dataclass(frozen=True)
class Fonte:
    titulo: str
    url: str
    arquivo: str


@dataclass(frozen=True)
class Tema:
    id: str
    titulo: str
    fonte: str  # chave de FONTES ou "interno"
    paginas: tuple[tuple[int, int], ...]
    papel: str  # "treino" ou "inedito"


FONTES = {
    "diretrizes_2015": Fonte(
        "Diretrizes para a detecção precoce do câncer de mama no Brasil (INCA, 2015)",
        "https://ninho.inca.gov.br/jspui/bitstream/123456789/10486/1/Diretrizes%20para%20detec%c3%a7%c3%a3o%20precoce%20do%20c%c3%a2ncer%20de%20mama%20-relat%c3%b3rio%20de%20recomenda%c3%a7%c3%a3o%20-%202015.pdf",
        "diretrizes_2015.pdf",
    ),
    "parametros": Fonte(
        "Parâmetros técnicos para programação de ações de detecção precoce do câncer de mama (INCA)",
        "https://ninho.inca.gov.br/jspui/bitstream/123456789/12746/1/Par%c3%a2metros%20t%c3%a9cnicos%20para%20programa%c3%a7%c3%a3o%20de%20a%c3%a7%c3%b5es%20de%20detec%c3%a7%c3%a3o%20precoce%20do%20c%c3%a2ncer%20de%20mama..pdf",
        "parametros.pdf",
    ),
}

TEMAS = (
    Tema("PROT-01", "Rastreamento mamográfico: faixa etária e periodicidade", "diretrizes_2015", ((62, 74),), "treino"),
    Tema("PROT-02", "Riscos do rastreamento: falso-positivo, radiação e sobrediagnóstico", "diretrizes_2015", ((56, 61),), "treino"),
    Tema("PROT-03", "Autoexame das mamas", "diretrizes_2015", ((75, 80),), "treino"),
    Tema("PROT-04", "Exame clínico das mamas", "diretrizes_2015", ((81, 86),), "treino"),
    Tema("PROT-05", "Ressonância magnética no rastreamento", "diretrizes_2015", ((87, 89),), "treino"),
    Tema("PROT-06", "Ultrassonografia no rastreamento", "diretrizes_2015", ((90, 93),), "treino"),
    Tema("PROT-07", "Termografia e tomossíntese", "diretrizes_2015", ((93, 97),), "inedito"),
    Tema("PROT-08", "Conscientização e diagnóstico precoce", "diretrizes_2015", ((107, 112),), "treino"),
    Tema("PROT-09", "Sinais e sintomas suspeitos e encaminhamento", "diretrizes_2015", ((112, 121),), "treino"),
    Tema("PROT-10", "Confirmação diagnóstica em um único serviço", "diretrizes_2015", ((122, 126),), "treino"),
    Tema("PROT-11", "Conduta por categoria BI-RADS", "parametros", ((10, 13), (20, 20)), "treino"),
    Tema("PROT-12", "Lesões palpáveis e não palpáveis: PAAF e punção por agulha grossa", "parametros", ((23, 27),), "treino"),
    Tema("PROT-13", "Gestão de pendências de exames", "interno", (), "treino"),
)

_CABECALHOS = (
    re.compile(r"^\s*\d*\s*Instituto Nacional de C[aâ]ncer.*$", re.M),
    re.compile(r"^\s*Par[aâ]metros T[eé]cnicos para Programa[cç][aã]o.*$", re.M),
    re.compile(r"^\s*Instrumentos de Gest[aã]o\s*$", re.M),
    re.compile(r"^\s*\d{1,3}\s*$", re.M),  # numero de pagina sozinho na linha
)


def limpar(texto: str) -> str:
    for padrao in _CABECALHOS:
        texto = padrao.sub("", texto)
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)  # hifenizacao de fim de linha
    texto = re.sub(r"\n{2,}", "\n", texto)
    return texto.strip()


def extrair(tema: Tema, diretorio: Path = config.FONTES_DIR) -> str:
    """Texto limpo das paginas do tema. pypdf so existe no WSL."""
    from pypdf import PdfReader

    leitor = PdfReader(Path(diretorio) / FONTES[tema.fonte].arquivo)
    partes = []
    for inicio, fim in tema.paginas:
        for numero in range(inicio, fim + 1):
            partes.append(leitor.pages[numero - 1].extract_text() or "")
    return limpar("\n".join(partes))


def sha256(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def fatos_internos(versao: int) -> str:
    """Fatos do PROT-13, em forma de lista, a partir da configuracao."""
    prazo = config.CLINICA_PRAZO_PENDENCIA_DIAS if versao == 1 else config.CLINICA_PRAZO_PENDENCIA_DIAS_V2
    return "\n".join([
        f"- Exame solicitado sem resultado ha mais de {prazo} dias e considerado pendente.",
        "- Toda pendencia gera alerta para a equipe medica responsavel pela paciente.",
        "- O alerta informa o exame, a data da solicitacao e ha quantos dias esta pendente.",
        "- PAAF com laudo citopatologico pendente e classificada pelo modelo de triagem antes do retorno.",
        "- Caso perto do limiar do modelo de triagem gera alerta de cautela, mesmo sem pendencia vencida.",
        "- O resultado do modelo de triagem nao substitui o laudo; a decisao e sempre do medico.",
    ])
