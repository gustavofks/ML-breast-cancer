"""Hospital 3D: apoio visual da apresentacao do Tech Challenge.

Um predio com robos funcionarios, em que cada sala e uma etapa real do projeto.
Como a vitrine, e uma pagina so, com tudo embutido: roda no Space estatico sem
servidor, sem chave e sem custo, e nada pode cair no meio da apresentacao.

**Nenhum numero e escrito a mao.** As falas dos robos sao moldes preenchidos com
`results/*.json`, e cada resposta traz o arquivo de onde veio, o mesmo habito que
a Fase 3 exige do assistente. Os casos e o log vem da pagina montada da vitrine,
que consultou a API de verdade: refazer essa consulta aqui exigiria o pandas, e
este modulo usa so a biblioteca padrao para montar tambem na maquina de
desenvolvimento, onde o Smart App Control bloqueia o pandas.

O texto dos robos fica fora de qualquer conjunto de treino da Fase 3.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from src import config
from src.ui import filmes
from src.ui.formato import num, pct, por_nome

# Mesmo texto de `src/ui/app.py`, que nao pode ser importado aqui: ele traz o pandas.
AVISO = (
    "Sistema de apoio à decisão. A previsão e a explicação ajudam a leitura do caso, "
    "mas o diagnóstico final é sempre do médico responsável."
)

BRACO_COMPLETO = "AG, completo"
BRACO_REFERENCIA = "Original"

# Ordem da visita guiada: a historia do projeto, que tambem e a ordem no corredor.
ORDEM_DA_VISITA = (
    "recepcao", "dados", "modelos", "imagem", "evolucao", "laudos", "central", "obras",
)

FONTE_METRICAS = "results/metrics.json"
FONTE_VISAO = "results/metrics_vision.json"
FONTE_COMPARACAO = "results/genetic/comparacao_final.json"
FONTE_RESUMO = "results/genetic/summary.json"
FONTE_SERVIDO = "results/genetic/modelo_servido.json"
FONTE_NARRATIVAS = "results/llm/narrativas.json"
FONTE_CUSTO = "results/llm/custo.json"
FONTE_ENUNCIADO = "docs/referencias/tc3_instrucoes.pdf"
FONTE_CONFIG = "src/config.py"


def _ler(caminho: Path) -> Any:
    return json.loads(caminho.read_text(encoding="utf-8"))


def casos_da_vitrine(pagina: Path = config.VITRINE_PAGINA) -> dict[str, Any]:
    """Dados congelados da vitrine: as respostas reais da API, sem refazer a consulta."""
    if not pagina.exists():
        raise FileNotFoundError(f"{pagina} nao existe: rode scripts/montar_space.py na imagem `api`.")
    html = pagina.read_text(encoding="utf-8")
    bloco = re.search(r'<script id="dados" type="application/json">([\s\S]*?)</script>', html)
    return json.loads(bloco.group(1))


def execucao_servida(execucoes: list[dict[str, Any]]) -> dict[str, Any]:
    """A execucao de maior fitness do braco completo: a regra do modelo servido.

    A sala da evolucao anima o historico dela, nao uma execucao ilustrativa.
    """
    completas = [e for e in execucoes if e["braco"] == BRACO_COMPLETO]
    return max(completas, key=lambda e: e["historico"][-1]["melhor_fitness"])


def _sala_recepcao(contexto: dict[str, Any]) -> dict[str, Any]:
    dataset = contexto["metricas"]["dataset"]
    return {
        "id": "recepcao",
        "nome": "Recepção",
        "fase": "Boas-vindas",
        "robo": {"nome": "Lia", "papel": "Recepcionista", "cor": "#e0528c", "acessorio": "fone"},
        "fala": (
            "Bem-vindo ao hospital do Tech Challenge. Cada sala daqui é uma etapa real do "
            "projeto, e cada robô cuida de uma delas. Tudo o que eles contam sai dos arquivos "
            "de resultado, e a fonte aparece junto da resposta. O hospital é fictício; os "
            f"{dataset['amostras']} exames da base são públicos."
        ),
        "perguntas": [
            {
                "pergunta": "Qual é o desafio?",
                "resposta": (
                    "Um hospital quer apoio de IA no diagnóstico de câncer de mama. O erro que "
                    "mais importa é o falso negativo: um tumor maligno classificado como benigno. "
                    "O projeto cresce em fases, e cada uma resolve uma parte."
                ),
                "evidencias": [
                    {"rotulo": "Exames na base", "valor": str(dataset["amostras"])},
                    {"rotulo": "Medidas por exame", "valor": str(dataset["features"])},
                ],
                "fonte": FONTE_METRICAS,
            },
            {
                "pergunta": "O que tem em cada ala?",
                "resposta": (
                    "Ala verde: Fase 1, com dados, modelos e imagem. Ala violeta: Fase 2, com "
                    "evolução, laudos e central de monitoramento. Ala amarela: Fase 3, em obras. "
                    "As linhas coloridas no chão levam a cada uma."
                ),
                "evidencias": [],
                "fonte": "README.md",
            },
            {
                "pergunta": "Isso decide o diagnóstico?",
                "resposta": f"Não. {AVISO}",
                "evidencias": [],
                "fonte": "src/ui/app.py",
            },
        ],
    }


def _sala_dados(contexto: dict[str, Any]) -> dict[str, Any]:
    dataset = contexto["metricas"]["dataset"]
    return {
        "id": "dados",
        "nome": "Laboratório de Dados",
        "fase": "Fase 1",
        "robo": {"nome": "Dara", "papel": "Curadora de dados", "cor": "#1f9e8f", "acessorio": "oculos"},
        "fala": (
            f"Cada pasta da estante é um exame de verdade: {dataset['benignos']} azuis, benignos, "
            f"e {dataset['malignos']} laranja, malignos. Separei {dataset['treino']} para treino e "
            f"tranquei {dataset['teste']} para o teste, que ninguém abre até a avaliação final."
        ),
        "quadros": [
            {"figura": "05_correlacao.png", "titulo": "Correlação entre as medidas"},
            {"figura": "04_separacao_features.png", "titulo": "Medidas que separam as classes"},
        ],
        "perguntas": [
            {
                "pergunta": "Por que o teste fica trancado?",
                "resposta": (
                    "Se a escolha do modelo olhar o teste, o resultado final fica otimista e deixa "
                    "de medir o que acontece com um exame novo. Comparação, ajuste e limiar "
                    "acontecem por validação cruzada dentro do treino."
                ),
                "evidencias": [
                    {"rotulo": "Treino", "valor": str(dataset["treino"])},
                    {"rotulo": "Teste", "valor": str(dataset["teste"])},
                    {"rotulo": "Dobras da validação", "valor": str(config.CV_FOLDS)},
                ],
                "fonte": FONTE_METRICAS,
            },
            {
                "pergunta": "Como evitam vazamento de dados?",
                "resposta": (
                    "O pré-processamento mora dentro do Pipeline, junto do modelo. Assim o "
                    "StandardScaler aprende a escala só com a parte de treino de cada dobra."
                ),
                "evidencias": [],
                "fonte": "src/preprocessing.py",
            },
            {
                "pergunta": "Os outliers foram removidos?",
                "resposta": (
                    "Não. Em câncer, um valor extremo pode ser justamente o sinal que importa. "
                    "Apagar esses exames deixaria o modelo cego para os casos mais graves."
                ),
                "evidencias": [],
                "fonte": "docs/relatorio_tecnico.pdf",
            },
        ],
    }


def _sala_modelos(contexto: dict[str, Any]) -> dict[str, Any]:
    metricas = contexto["metricas"]
    escolhido = metricas["modelo_escolhido"]
    teste = por_nome(metricas["teste"], "modelo", escolhido)
    malignos = teste["falsos_negativos"] + teste["verdadeiros_positivos"]
    shap = metricas["explicabilidade"]["shap_global"]
    com_limiar_baixo = por_nome(metricas["limiares"], "limiar", 0.3)
    return {
        "id": "modelos",
        "nome": "Sala dos Modelos",
        "fase": "Fase 1",
        "robo": {"nome": "Téo", "papel": "Analista de modelos", "cor": "#2f7fd6", "acessorio": "jaleco"},
        "fala": (
            f"Três modelos disputaram por validação cruzada, e venceu {escolhido}, com F1 de "
            f"{num(metricas['validacao_cruzada'][0]['f1'])}. Ganhou por um motivo a mais: dá "
            f"para explicar cada decisão. No teste, deixou passar {teste['falsos_negativos']} de "
            f"{malignos} tumores malignos."
        ),
        "podio": [
            {"modelo": linha["modelo"], "f1": linha["f1"]} for linha in metricas["validacao_cruzada"]
        ],
        "quadros": [
            {"figura": "13_shap_beeswarm.png", "titulo": "SHAP: o peso de cada medida"},
            {"figura": "08_curvas_roc.png", "titulo": "Curvas ROC no teste"},
        ],
        "perguntas": [
            {
                "pergunta": "Por que F1, e não só recall?",
                "resposta": (
                    "Recall puro premia o modelo que chama tudo de maligno. O F1 equilibra achar os "
                    "tumores com não alarmar à toa. O custo do falso negativo entra depois, no "
                    "limiar e na Fase 2."
                ),
                "evidencias": [
                    {"rotulo": linha["modelo"], "valor": f"F1 {num(linha['f1'])}"}
                    for linha in metricas["validacao_cruzada"]
                ],
                "fonte": FONTE_METRICAS,
            },
            {
                "pergunta": "Quais medidas pesam mais?",
                "resposta": (
                    f"Três métodos foram comparados: coeficientes, permutação e SHAP. Em todos, "
                    f"{shap[0]['feature']} aparece no topo. O quadro da parede mostra o SHAP completo."
                ),
                "evidencias": [
                    {"rotulo": item["feature"], "valor": f"SHAP {num(item['shap_medio'])}"}
                    for item in shap[:3]
                ],
                "fonte": FONTE_METRICAS,
                "figura": "13_shap_beeswarm.png",
            },
            {
                "pergunta": "E o limiar de decisão?",
                "resposta": (
                    f"Com limiar 0,5, {teste['falsos_negativos']} falsos negativos. Com 0,3, caem para "
                    f"{com_limiar_baixo['falsos_negativos']}. Só que essa tabela olhou o teste, então "
                    "serviu de diagnóstico, não de calibração. A Fase 2 corrigiu isso, calibrando "
                    "dentro do treino."
                ),
                "evidencias": [
                    {"rotulo": "Falsos negativos, limiar 0,5", "valor": str(teste["falsos_negativos"])},
                    {"rotulo": "Falsos negativos, limiar 0,3", "valor": str(com_limiar_baixo["falsos_negativos"])},
                ],
                "fonte": FONTE_METRICAS,
                "figura": "10_limiar_decisao.png",
            },
        ],
    }


def _sala_imagem(contexto: dict[str, Any]) -> dict[str, Any]:
    visao = contexto["visao"]
    escolhido = por_nome(visao["teste"], "modelo", visao["modelo_escolhido"])
    do_zero = por_nome(visao["teste"], "modelo", "CNN do zero")
    duplicatas = visao["duplicatas"]
    return {
        "id": "imagem",
        "nome": "Sala de Imagem",
        "fase": "Fase 1, entrega extra",
        "robo": {"nome": "Íris", "papel": "Técnica de imagem", "cor": "#8a5cd6", "acessorio": "touca"},
        "fala": (
            f"Aqui a entrada é o ultrassom cru, sem medida feita por especialista. A "
            f"{visao['modelo_escolhido']} achou {pct(escolhido['recall_maligno'], 0)} dos malignos "
            f"do teste, {escolhido['malignos_no_teste'] - escolhido['malignos_nao_detectados']} de "
            f"{escolhido['malignos_no_teste']}. É demonstração de viabilidade, não sistema clínico."
        ),
        "quadros": [
            {"figura": "25_gradcam_acertos.png", "titulo": "Grad-CAM: onde a rede olhou nos acertos"},
            {"figura": "26_gradcam_erros.png", "titulo": "Grad-CAM: onde a rede olhou nos erros"},
        ],
        "perguntas": [
            {
                "pergunta": "Por que não treinar do zero?",
                "resposta": (
                    f"Tentamos. Com {visao['dataset']['imagens']} imagens, a rede do zero não achou "
                    f"nenhum maligno: {do_zero['malignos_nao_detectados']} de "
                    f"{do_zero['malignos_no_teste']} passaram. Partir de uma rede pré-treinada foi o "
                    "que fez funcionar."
                ),
                "evidencias": [
                    {"rotulo": "CNN do zero", "valor": pct(do_zero["recall_maligno"], 0)},
                    {"rotulo": visao["modelo_escolhido"], "valor": pct(escolhido["recall_maligno"], 0)},
                ],
                "fonte": FONTE_VISAO,
            },
            {
                "pergunta": "A base tinha algum problema?",
                "resposta": (
                    f"Tinha. Das {duplicatas['imagens']} imagens, {duplicatas['imagens_redundantes']} "
                    f"eram repetidas, e {duplicatas['grupos_com_rotulos_contraditorios']} grupos "
                    "tinham rótulos contraditórios. Sem separar as repetidas, a mesma imagem cairia "
                    "no treino e no teste."
                ),
                "evidencias": [
                    {"rotulo": "Imagens redundantes", "valor": str(duplicatas["imagens_redundantes"])},
                    {"rotulo": "Grupos contraditórios", "valor": str(duplicatas["grupos_com_rotulos_contraditorios"])},
                ],
                "fonte": FONTE_VISAO,
            },
            {
                "pergunta": "Como saber onde a rede olhou?",
                "resposta": (
                    "Pelo Grad-CAM, o mapa de calor dos quadros. Ele mostra a região que pesou na "
                    "decisão, e os erros também ficam expostos, para não parecer melhor do que é."
                ),
                "evidencias": [],
                "fonte": FONTE_VISAO,
                "figura": "26_gradcam_erros.png",
            },
        ],
    }


def _sala_evolucao(contexto: dict[str, Any]) -> dict[str, Any]:
    resumo = contexto["resumo"]
    comparacao = contexto["comparacao"]
    servido = contexto["servido"]
    execucao = contexto["execucao"]
    bracos = sorted(resumo["bracos"], key=lambda b: b["custo_medio"])
    referencia = por_nome(comparacao, "braco", BRACO_REFERENCIA)
    final = por_nome(comparacao, "braco", BRACO_COMPLETO)
    auc = [linha["roc_auc"] for linha in comparacao]
    return {
        "id": "evolucao",
        "nome": "Sala da Evolução",
        "fase": "Fase 2",
        "robo": {"nome": "Gene", "papel": "Pesquisador genético", "cor": "#7a4fd0", "acessorio": "jaleco"},
        "fala": (
            f"Cada esfera é um indivíduo: um conjunto de medidas, hiperparâmetros e limiar. A nuvem "
            f"encolhe porque a diversidade real desta execução cai. Foram {resumo['execucoes']} "
            f"execuções, com falso negativo custando {num(config.CUSTO_FALSO_NEGATIVO, 0)} vezes "
            "mais que falso positivo."
        ),
        "enxame": {
            "populacao": execucao["configuracao"]["populacao"],
            "historico": [
                {"fitness": g["melhor_fitness"], "diversidade": g["diversidade"]}
                for g in execucao["historico"]
            ],
        },
        "quadros": [
            {"figura": "genetic/11_convergencia.png", "titulo": "Convergência da busca"},
            {"figura": "genetic/13_frequencia_features.png", "titulo": "Medidas mais escolhidas"},
        ],
        "perguntas": [
            {
                "pergunta": "O algoritmo achou um modelo melhor?",
                "resposta": (
                    f"Não é essa a história. A AUC ficou entre {num(min(auc))} e {num(max(auc))} em "
                    "todos os braços: a capacidade de ordenar os casos é a mesma. O ganho em falso "
                    "negativo vem de mover o ponto de corte, não de escolher medidas."
                ),
                "evidencias": [
                    {"rotulo": braco["braco"], "valor": f"custo {num(braco['custo_medio'], 4)}"}
                    for braco in bracos
                ],
                "fonte": FONTE_RESUMO,
            },
            {
                "pergunta": "Qual modelo foi para produção?",
                "resposta": (
                    f"{servido['familia']} com {len(servido['features'])} medidas e limiar "
                    f"{num(servido['limiar'], 4)}. No teste, {final['falsos_negativos']} falsos "
                    f"negativos e {final['falsos_positivos']} falsos positivos, contra "
                    f"{referencia['falsos_negativos']} e {referencia['falsos_positivos']} da "
                    f"referência com {referencia['n_features']} medidas. Reduzir medidas custou pouca "
                    f"AUC: {num(final['roc_auc'], 4)} contra {num(referencia['roc_auc'], 4)}."
                ),
                "evidencias": [
                    {"rotulo": "Medidas", "valor": f"{final['n_features']} de {referencia['n_features']}"},
                    {"rotulo": "Limiar", "valor": num(servido["limiar"], 4)},
                ],
                "fonte": FONTE_COMPARACAO,
            },
            {
                "pergunta": "Dá para ordenar os braços pelo teste?",
                "resposta": (
                    f"Não. São {final['n_amostras']} casos, com "
                    f"{final['falsos_negativos'] + final['verdadeiros_positivos']} malignos. Um falso "
                    f"negativo a mais vale {num(config.CUSTO_FALSO_NEGATIVO / final['n_amostras'])} de "
                    "custo, então diferença de um ou dois é ruído de amostra. E a regra do modelo "
                    "servido foi fixada antes de abrir o teste."
                ),
                "evidencias": [
                    {"rotulo": "Casos no teste", "valor": str(final["n_amostras"])},
                    {"rotulo": "Malignos", "valor": str(final["falsos_negativos"] + final["verdadeiros_positivos"])},
                ],
                "fonte": "results/genetic/regra_do_modelo_servido.json",
                "figura": "genetic/12_fronteira_custo_features.png",
            },
        ],
    }


def _sala_laudos(contexto: dict[str, Any]) -> dict[str, Any]:
    narrativas = contexto["narrativas"]
    custo = contexto["custo"]
    aprovados = sum(1 for n in narrativas if n["qualidade"]["aprovado"])
    return {
        "id": "laudos",
        "nome": "Sala de Laudos",
        "fase": "Fase 2",
        "robo": {"nome": "Vera", "papel": "Revisora de textos", "cor": "#c2478f", "acessorio": "prancheta"},
        "fala": (
            f"Um LLM escreve a explicação de cada caso, e eu confiro antes de chegar à tela. "
            f"{aprovados} de {len(narrativas)} textos passaram nas quatro verificações. Eles foram "
            "gerados uma vez, num lote fora do ar: o serviço publicado não tem chave nenhuma."
        ),
        "perguntas": [
            {
                "pergunta": "O que você confere em cada texto?",
                "resposta": (
                    "Fidelidade: todo número do texto existe no caso. Segurança: nada de prescrever "
                    "ou dar diagnóstico fechado. Cobertura: as medidas que mais pesaram são citadas. "
                    "Formato: seções e tamanho certos."
                ),
                "evidencias": [
                    {"rotulo": "Aprovados", "valor": f"{aprovados} de {len(narrativas)}"},
                ],
                "fonte": FONTE_NARRATIVAS,
            },
            {
                "pergunta": "Quanto custou gerar tudo?",
                "resposta": (
                    f"US$ {num(custo['custo_usd'], 3)} com {custo['modelo']}, dentro do teto de "
                    f"US$ {num(custo['orcamento_usd'], 2)} conferido antes de cada chamada. Texto "
                    "gerado uma vez vira cache, e o custo não se repete."
                ),
                "evidencias": [
                    {"rotulo": "Custo total", "valor": f"US$ {num(custo['custo_usd'], 3)}"},
                    {"rotulo": "Teto", "valor": f"US$ {num(custo['orcamento_usd'], 2)}"},
                ],
                "fonte": FONTE_CUSTO,
            },
            {
                "pergunta": "Por que não gerar ao vivo?",
                "resposta": (
                    "Geração ao vivo exigiria publicar uma chave e deixaria custo recorrente. Com o "
                    "cache, o texto na tela é o mesmo que passou pela verificação."
                ),
                "evidencias": [],
                "fonte": "src/llm/narrator.py",
            },
        ],
    }


def _sala_central(contexto: dict[str, Any]) -> dict[str, Any]:
    eventos = contexto["eventos"]
    return {
        "id": "central",
        "nome": "Central de Monitoramento",
        "fase": "Fase 2",
        "robo": {"nome": "Argos", "papel": "Operador da API", "cor": "#d9822b", "acessorio": "fone"},
        "fala": (
            "Toda previsão passa por aqui. O modelo servido responde numa API, e cada chamada vira "
            "uma linha de log. O painel atrás de mim mostra uma sessão gravada de verdade, com "
            "rotas, status e latência."
        ),
        "log": [
            {
                "metodo": e.get("metodo"),
                "rota": e.get("rota"),
                "status": e.get("status"),
                "latencia_ms": e.get("latencia_ms"),
                "evento": e.get("evento"),
            }
            for e in eventos
        ],
        "perguntas": [
            {
                "pergunta": "O que é a sentinela?",
                "resposta": (
                    f"Ela compara cada medida com a distribuição do treino e alerta quando foge mais "
                    f"de {num(config.API_LIMIAR_DE_DESVIO, 0)} desvios padrão. Ela registra, não "
                    "bloqueia: quem decide se confia no resultado é o médico."
                ),
                "evidencias": [
                    {"rotulo": "Limite", "valor": f"{num(config.API_LIMIAR_DE_DESVIO, 0)} desvios padrão"},
                ],
                "fonte": "src/api/monitoring.py",
            },
            {
                "pergunta": "Quais rotas a API tem?",
                "resposta": (
                    "/health diz se está de pé, /model-info mostra o modelo, /predict devolve a "
                    "previsão, /explain soma a explicação, e /metrics e /logs/recentes abrem o "
                    "monitoramento."
                ),
                "evidencias": [],
                "fonte": "src/api/main.py",
            },
            {
                "pergunta": "O log guarda dado de paciente?",
                "resposta": (
                    "Não. Cada linha registra rota, status e latência, e marca explicitamente que "
                    "não há dado identificável."
                ),
                "evidencias": [
                    {"rotulo": "Eventos na sessão gravada", "valor": str(len(eventos))},
                ],
                "fonte": "src/api/registro.py",
            },
        ],
    }


def _sala_obras(contexto: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "obras",
        "nome": "Ala em Obras",
        "fase": "Fase 3",
        "robo": {"nome": "Bento", "papel": "Mestre de obras", "cor": "#d9a521", "acessorio": "capacete"},
        "fala": (
            "Aqui nada funciona ainda, e prefiro dizer isso. Nesta ala vai entrar o assistente "
            "médico da Fase 3. Os pedestais marcam o fluxo pedido no enunciado: cada etapa vai "
            "virar uma estação de verdade."
        ),
        "fluxo": ["Paciente", "Exames pendentes", "Sugestão", "Validação humana", "Alerta à equipe"],
        "perguntas": [
            {
                "pergunta": "O que a Fase 3 pede?",
                "resposta": (
                    "Fine-tuning de uma LLM com dados médicos, assistente com LangChain, fluxos com "
                    "LangGraph, limites de atuação, logs de auditoria e indicação da fonte de cada "
                    "resposta."
                ),
                "evidencias": [],
                "fonte": FONTE_ENUNCIADO,
            },
            {
                "pergunta": "O assistente vai prescrever?",
                "resposta": (
                    "Não. O enunciado exige o contrário: nunca prescrever diretamente, sem validação "
                    "humana. Por isso o fluxo tem uma estação só para isso."
                ),
                "evidencias": [],
                "fonte": FONTE_ENUNCIADO,
            },
        ],
    }


CONSTRUTORES = {
    "recepcao": _sala_recepcao,
    "dados": _sala_dados,
    "modelos": _sala_modelos,
    "imagem": _sala_imagem,
    "evolucao": _sala_evolucao,
    "laudos": _sala_laudos,
    "central": _sala_central,
    "obras": _sala_obras,
}


def _resumir_caso(indice: int, caso: dict[str, Any]) -> dict[str, Any]:
    """So o que a jornada do caso mostra; as medidas brutas ficam para a vitrine."""
    return {
        "indice": indice,
        "registrado": caso["diagnostico_registrado"],
        "classe": caso["classe"],
        "probabilidade": caso["probabilidade"],
        "limiar": caso["limiar"],
        "medidas": caso["medidas"],
        "contribuicoes": caso["contribuicoes"][:3],
        "texto": caso["texto"],
        "texto_motivo": caso["texto_motivo"],
    }


def montar_dados(vitrine: dict[str, Any], resultados: Path = config.RESULTS_DIR) -> dict[str, Any]:
    """Tudo o que a pagina mostra, lido dos artefatos e da vitrine montada."""
    genetico = resultados / "genetic"
    llm = resultados / "llm"
    contexto = {
        "metricas": _ler(resultados / "metrics.json"),
        "visao": _ler(resultados / "metrics_vision.json"),
        "resumo": _ler(genetico / "summary.json"),
        "comparacao": _ler(genetico / "comparacao_final.json"),
        "servido": _ler(genetico / "modelo_servido.json"),
        "execucao": execucao_servida(_ler(genetico / "execucoes.json")),
        "narrativas": _ler(llm / "narrativas.json"),
        "custo": _ler(llm / "custo.json"),
        "eventos": vitrine["console"]["gravacao"]["eventos"],
        "casos": vitrine["casos"],
    }
    salas = [CONSTRUTORES[identificador](contexto) for identificador in ORDEM_DA_VISITA]
    teste = por_nome(contexto["metricas"]["teste"], "modelo", contexto["metricas"]["modelo_escolhido"])
    return {
        "aviso": AVISO,
        "modelo": {
            "familia": contexto["servido"]["familia"],
            "limiar": contexto["servido"]["limiar"],
            "features": contexto["servido"]["features"],
        },
        "dataset": {
            "benignos": contexto["metricas"]["dataset"]["benignos"],
            "malignos": contexto["metricas"]["dataset"]["malignos"],
            "teste_benignos": teste["verdadeiros_negativos"] + teste["falsos_positivos"],
            "teste_malignos": teste["verdadeiros_positivos"] + teste["falsos_negativos"],
        },
        "salas": salas,
        "filmes": filmes.filmes(contexto),
        "casos": [_resumir_caso(i, caso) for i, caso in enumerate(vitrine["casos"])],
    }


def figuras_usadas(dados: dict[str, Any]) -> list[str]:
    """Quadros das paredes e figuras citadas nas respostas, sem repetir."""
    nomes: list[str] = []
    for sala in dados["salas"]:
        nomes += [quadro["figura"] for quadro in sala.get("quadros", [])]
        nomes += [p["figura"] for p in sala["perguntas"] if "figura" in p]
    for filme in dados["filmes"]:
        nomes += [p["figura"] for p in filme["planos"] if p["figura"]]
    return list(dict.fromkeys(nomes))


def _caminho_da_figura(nome: str, resultados: Path) -> Path:
    """Figuras do AG ficam em `results/genetic/`; as demais, em `results/figures/`."""
    return resultados / nome if nome.startswith("genetic/") else resultados / "figures" / nome


def embutir_figuras(dados: dict[str, Any], resultados: Path = config.RESULTS_DIR) -> dict[str, str]:
    """As figuras viram `data:` para a pagina continuar sendo um arquivo so."""
    saida = {}
    for nome in figuras_usadas(dados):
        conteudo = _caminho_da_figura(nome, resultados).read_bytes()
        saida[nome] = "data:image/png;base64," + base64.b64encode(conteudo).decode("ascii")
    return saida


def _json_seguro(valor: Any) -> str:
    """JSON que nao fecha o bloco `<script>` em que entra."""
    return json.dumps(valor, ensure_ascii=False).replace("</", "<\\/")


def _ativo(nome: str) -> str:
    return (config.HOSPITAL_DIR / nome).read_text(encoding="utf-8")


# Ordem importa: os arquivos sao concatenados num modulo so, e `hospital.js`, que
# usa os demais, vem por ultimo.
SCRIPTS = (
    "logica.js", "roteiro.js", "texturas.js", "predio.js", "mobilia.js", "robos.js", "camera.js",
    "ceu.js", "pos.js", "obra.js", "tomadas.js", "trilha.js", "letreiros.js", "diretor.js",
    "interface.js", "estudio.js", "hospital.js",
)


def pagina(dados: dict[str, Any], figuras: dict[str, str]) -> str:
    """A pagina inteira, com estilo, Three.js, scripts, dados e figuras embutidos."""
    return (
        _ativo("modelo.html")
        .replace("__ESTILO__", _ativo("estilo.css"))
        .replace("__THREE__", _ativo("vendor/three.module.min.js"))
        .replace("__APP__", "\n".join(_ativo(nome) for nome in SCRIPTS))
        .replace("__AVISO__", AVISO)
        .replace("__FIGURAS__", _json_seguro(figuras))
        .replace("__DADOS__", _json_seguro(dados))
    )
