"""Templates de prompt, com identificador e versao.

Cada template carrega nome, versao e identificador para que o relatorio possa
citar exatamente qual prompt produziu qual texto. A explicacao por paciente
existe em duas versoes, e a v1 e deliberadamente mais fraca — sem exemplo
resolvido e com contrato de saida frouxo. Medir as duas nas mesmas verificacoes
automaticas e o que transforma "aplicamos prompt engineering" de afirmacao em
resultado.

Tecnicas aplicadas, cada uma visivel no texto dos templates:

- papel e fronteira no prompt de sistema: apoio a decisao, nunca diagnostico
- evidencia entregue como JSON estruturado, e nao como prosa, porque numero em
  prosa convida o modelo a reescrever e numero em JSON convida a copiar
- um exemplo resolvido, para fixar formato e tom (so na v2)
- contrato de saida explicito, com secoes nomeadas e limite de tamanho
- instrucao de nao inventar numeros, usando apenas os valores fornecidos
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src import config

# Repetido em todo prompt de sistema. A fronteira clinica nao e preambulo
# decorativo: e o contrato que o verificador de seguranca confere no texto
# gerado, e que impede a saida de se passar por laudo.
FRONTEIRA = (
    "Você é um assistente de apoio à decisão clínica. Você explica a saída de um "
    "modelo estatístico para profissionais de saúde. Você não emite diagnóstico, "
    "não confirma nem descarta doença, e deixa explícito que sua leitura não "
    "substitui a avaliação do médico responsável."
)

LIMITE = config.LLM_LIMITE_DE_PALAVRAS


@dataclass(frozen=True)
class Template:
    """Um prompt versionado, com o contrato de saida que ele promete."""

    nome: str
    versao: int
    sistema: str
    usuario: str
    secoes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def identificador(self) -> str:
        """Chave estavel, do jeito que o relatorio e o artefato a citam."""
        return f"{self.nome}_v{self.versao}"

    def renderizar(self, evidencia: dict[str, Any]) -> str:
        """Injeta a evidencia como JSON no corpo do prompt.

        Usa `format_map` sobre um dicionario sem default: chave ausente levanta
        `KeyError` em vez de deixar um buraco no texto. Template que renderiza
        com buraco produz saida plausivel e errada, que e o defeito mais caro
        desta camada.
        """
        if not evidencia:
            raise KeyError("evidencia vazia: o prompt nao pode ser renderizado sem dados")

        return self.usuario.format_map(
            {"evidencia": json.dumps(evidencia, indent=2, ensure_ascii=False)}
        )


_CONTRATO_PACIENTE = ("Leitura do caso", "Medidas que mais pesaram", "Ressalva")
_CONTRATO_GESTAO = ("Panorama", "O que merece atenção", "Ressalva")
_CONTRATO_MUDANCA = ("O que mudou", "O que isso significa na prática", "Ressalva")

_EXEMPLO_RESOLVIDO = """
Exemplo de resposta bem formada, para um caso diferente deste:

## Leitura do caso
A probabilidade estimada de malignidade é 0,88, acima do limiar de 0,30 adotado
neste serviço, então o caso é sinalizado para revisão prioritária.

## Medidas que mais pesaram
As três medidas de maior contribuição foram texture_worst, area_worst e
concave points_mean, todas acima da faixa típica dos casos benignos desta base.

## Ressalva
Esta leitura é apoio à decisão e não substitui a avaliação do médico responsável.
"""

TEMPLATES: dict[str, Template] = {}


def _registrar(template: Template) -> Template:
    TEMPLATES[template.identificador] = template
    return template


_registrar(
    Template(
        nome="explicacao_paciente",
        versao=1,
        sistema=FRONTEIRA,
        usuario=(
            "Explique a previsão do modelo para este caso.\n\n"
            "Evidência:\n{evidencia}\n\n"
            "Use apenas os valores fornecidos acima; não calcule nem estime "
            "nenhum número que não esteja na evidência.\n"
            "Organize a resposta nas seções: Leitura do caso, Medidas que mais "
            "pesaram, Ressalva."
        ),
        secoes=_CONTRATO_PACIENTE,
    )
)

_registrar(
    Template(
        nome="explicacao_paciente",
        versao=2,
        sistema=FRONTEIRA,
        usuario=(
            "Explique a previsão do modelo para este caso, para um médico que vai "
            "ler o texto ao lado do gráfico de contribuições.\n\n"
            "Evidência:\n{evidencia}\n\n"
            "Regras:\n"
            "- Use apenas os valores fornecidos acima. Não calcule, não arredonde "
            "e não estime nenhum número ausente da evidência.\n"
            "- Cite nominalmente as três medidas de maior contribuição.\n"
            "- Escreva no máximo "
            f"{LIMITE} palavras.\n"
            "- Use exatamente estas seções, nesta ordem, como títulos de nível 2: "
            "Leitura do caso, Medidas que mais pesaram, Ressalva.\n"
            f"{_EXEMPLO_RESOLVIDO}"
        ),
        secoes=_CONTRATO_PACIENTE,
    )
)

_registrar(
    Template(
        nome="resumo_gestao",
        versao=1,
        sistema=FRONTEIRA,
        usuario=(
            "Transforme as métricas abaixo em insights acionáveis para a gestão "
            "clínica do serviço.\n\n"
            "Evidência:\n{evidencia}\n\n"
            "Regras:\n"
            "- Use apenas os valores fornecidos acima; nenhum número novo.\n"
            "- Fale em termos de carga de trabalho e risco, não de estatística.\n"
            "- Escreva no máximo "
            f"{LIMITE} palavras.\n"
            "- Use exatamente estas seções, nesta ordem, como títulos de nível 2: "
            "Panorama, O que merece atenção, Ressalva."
        ),
        secoes=_CONTRATO_GESTAO,
    )
)

_registrar(
    Template(
        nome="nota_de_mudanca",
        versao=1,
        sistema=FRONTEIRA,
        usuario=(
            "Explique, em linguagem simples, o que a otimização por algoritmo "
            "genético alterou em relação ao modelo anterior.\n\n"
            "Evidência:\n{evidencia}\n\n"
            "Regras:\n"
            "- Use apenas os valores fornecidos acima; nenhum número novo.\n"
            "- Se a evidência mostrar que o ganho não se confirmou, diga isso com "
            "clareza em vez de suavizar.\n"
            "- Escreva no máximo "
            f"{LIMITE} palavras.\n"
            "- Use exatamente estas seções, nesta ordem, como títulos de nível 2: "
            "O que mudou, O que isso significa na prática, Ressalva."
        ),
        secoes=_CONTRATO_MUDANCA,
    )
)


def versoes_de(nome: str) -> list[Template]:
    """Todas as versoes de um prompt, da mais antiga para a mais nova."""
    return sorted(
        (t for t in TEMPLATES.values() if t.nome == nome), key=lambda t: t.versao
    )
