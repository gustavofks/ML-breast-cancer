"""Pedidos ao modelo professor e leitura do que ele devolve.

O professor escreve todo o texto que entra no treino (spec 4.2). Os pedidos
daqui sao instrucoes, nao conteudo: nao aparecem em nenhum exemplo. O pedido de
protocolo proibe acrescentar fatos porque, na semana 1, o professor acrescentou
um limite de crescimento que nao estava na fonte; a curadoria confere de novo.
"""

from __future__ import annotations

import json
import re

from src import config

_HOSPITAL = config.CLINICA_NOME


def _usuario(texto: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": texto}]


def pedido_protocolo(titulo: str, fonte: str) -> list[dict[str, str]]:
    return _usuario(
        f"Você escreve protocolos internos do {_HOSPITAL}, um hospital fictício.\n"
        f"Escreva o protocolo \"{titulo}\" usando SOMENTE os fatos do TEXTO DE ORIGEM abaixo.\n"
        "Regras:\n"
        "- Não acrescente número, prazo, porcentagem, idade, exemplo, exame ou recomendação "
        "que não esteja no texto de origem.\n"
        "- Não cite medicamento nem dose.\n"
        "- Linguagem técnica, em português do Brasil.\n"
        "- Formato exato, em markdown:\n"
        f"# {titulo}\n## Objetivo\n(um parágrafo)\n## Público\n(um parágrafo)\n"
        "## Conduta\n(itens numerados)\n## Observações\n(itens numerados, se houver)\n\n"
        f"TEXTO DE ORIGEM:\n{fonte}"
    )


def pedido_verificacao(documento: str, fonte: str) -> list[dict[str, str]]:
    return _usuario(
        "Compare o DOCUMENTO com o TEXTO DE ORIGEM. Liste as frases do DOCUMENTO que afirmam "
        "algo que NÃO está no TEXTO DE ORIGEM. Responda só com uma lista JSON no formato "
        '[{"frase": "..."}]. Se todas as frases tiverem base, responda [].\n\n'
        f"TEXTO DE ORIGEM:\n{fonte}\n\nDOCUMENTO:\n{documento}"
    )


def pedido_perguntas(bloco_trechos: str, *, n: int, ids: list[str]) -> list[dict[str, str]]:
    citacoes = ", ".join(f"[{i}]" for i in ids)
    return _usuario(
        f"Abaixo estão trechos de protocolos do {_HOSPITAL}. Escreva {n} perguntas diferentes "
        "que um médico faria sobre eles, cada uma com a resposta.\n"
        "Regras da resposta:\n"
        "- Responda só com o que está nos trechos.\n"
        f"- Depois de cada afirmação, cite o trecho usado, exatamente neste formato: {citacoes}.\n"
        "- Termine lembrando que a decisão é do médico responsável.\n"
        "- Não prescreva medicamento nem dose.\n"
        'Responda só com uma lista JSON: [{"pergunta": "...", "resposta": "..."}].\n\n'
        f"TRECHOS:\n{bloco_trechos}"
    )


def pedido_recusas(n: int) -> list[dict[str, str]]:
    return _usuario(
        f"Escreva {n} pedidos diferentes que um profissional poderia fazer a um assistente "
        f"virtual do {_HOSPITAL} e que o assistente deve recusar: prescrever medicamento ou dose, "
        "dar diagnóstico definitivo, dispensar avaliação médica, alterar prontuário. Varie o tom "
        "e o contexto. Para cada pedido, escreva a recusa do assistente: educada, curta, explicando "
        "que a decisão e a prescrição cabem ao médico, e oferecendo o que ele pode fazer "
        "(consultar protocolos, resumir o prontuário). "
        'Responda só com uma lista JSON: [{"pergunta": "...", "resposta": "..."}].'
    )


def pedido_fora_do_escopo(n: int) -> list[dict[str, str]]:
    return _usuario(
        f"Escreva {n} perguntas fora do assunto de saúde (esporte, culinária, programação, "
        f"política...) que alguém poderia fazer ao assistente clínico do {_HOSPITAL}, cada uma "
        "com a resposta do assistente: recusa curta e educada, dizendo que ele só apoia a equipe "
        "em temas clínicos do hospital. "
        'Responda só com uma lista JSON: [{"pergunta": "...", "resposta": "..."}].'
    )


def pedido_gerais(n: int) -> list[dict[str, str]]:
    return _usuario(
        f"Escreva {n} perguntas variadas de conhecimento geral de saúde (fisiologia, prevenção, "
        "exames de rotina, hábitos), sem relação com câncer de mama, cada uma com uma resposta "
        "correta de duas a quatro frases completas, em português do Brasil, sem prescrever "
        "medicamento nem dose. "
        'Responda só com uma lista JSON: [{"pergunta": "...", "resposta": "..."}].'
    )


def pedido_nao_encontrado(pergunta: str, bloco_trechos: str) -> list[dict[str, str]]:
    return _usuario(
        f"Um assistente do {_HOSPITAL} recebeu a PERGUNTA e os TRECHOS abaixo. Os trechos não "
        "respondem à pergunta. Escreva a resposta do assistente: diga que não encontrou a "
        "informação nos protocolos recebidos, não tente responder de memória e sugira consultar "
        "o médico responsável ou o protocolo adequado. Responda só com o texto da resposta.\n\n"
        f"PERGUNTA: {pergunta}\n\nTRECHOS:\n{bloco_trechos}"
    )


def ler_lista_json(texto: str, campos: tuple[str, ...] = ()) -> list[dict]:
    """Primeira lista JSON do texto, mesmo dentro de cerca de markdown.

    JSON quebrado vira lista vazia: o pedido e refeito ou descartado pelo
    orquestrador, e um item mal formado nunca entra no conjunto.
    """
    inicio, fim = texto.find("["), texto.rfind("]")
    if inicio < 0 or fim <= inicio:
        return []
    try:
        dados = json.loads(texto[inicio:fim + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(dados, list):
        return []
    return [item for item in dados
            if isinstance(item, dict) and all(isinstance(item.get(c), str) and item[c].strip()
                                              for c in campos)]


def frases(texto: str) -> list[str]:
    return [f.strip() for f in re.split(r"(?<=[.!?])\s+|\n", texto) if f.strip()]
