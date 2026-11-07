"""Roteiros dos filmes do hospital: um video de apresentacao por fase.

O hospital nasce no video: a Fase 1 abre num terreno vazio, e cada fase ergue a
sua ala. Cada plano traz a narracao, a tomada de camera e as acoes que
`src/ui/hospital/diretor.js` executa. Os nomes de tomada e acao vivem la, junto
da geometria; aqui fica so o conteudo.

**Nenhum numero e escrito a mao:** toda narracao e um molde preenchido com
`results/`, pela mesma regra dos relatorios. O filme conta o achado real, inclusive
o que nao deu certo, e o da Fase 3 se declara rascunho enquanto nada existir.
"""

from __future__ import annotations

from typing import Any

from src import config
from src.ui.formato import num, pct, por_nome

BRACO_COMPLETO = "AG, completo"
BRACO_REFERENCIA = "Original"
LIMIAR_PADRAO = 0.5

FONTE_METRICAS = "results/metrics.json"
FONTE_VISAO = "results/metrics_vision.json"
FONTE_COMPARACAO = "results/genetic/comparacao_final.json"
FONTE_RESUMO = "results/genetic/summary.json"
FONTE_EXECUCOES = "results/genetic/execucoes.json"
FONTE_NARRATIVAS = "results/llm/narrativas.json"
FONTE_CUSTO = "results/llm/custo.json"
FONTE_ENUNCIADO = "docs/referencias/tc3_instrucoes.pdf"


def evidencia(rotulo: str, numero: float, casas: int = 0, prefixo: str = "", sufixo: str = "") -> dict[str, Any]:
    """Contador que sobe na tela: o valor final e o que a narracao diz."""
    return {"rotulo": rotulo, "numero": numero, "casas": casas, "prefixo": prefixo, "sufixo": sufixo}


def plano(
    identificador: str,
    capitulo: str,
    tomada: str,
    narracao: str,
    *,
    robo: str | None = None,
    acoes: tuple[str, ...] = (),
    evidencias: tuple[dict[str, Any], ...] = (),
    fonte: str | None = None,
    figura: str | None = None,
    legenda_figura: str | None = None,
    telao: bool = False,
) -> dict[str, Any]:
    return {
        "id": identificador,
        "capitulo": capitulo,
        "tomada": tomada,
        "narracao": narracao,
        "robo": robo,
        "acoes": list(acoes),
        "evidencias": list(evidencias),
        "fonte": fonte,
        "figura": figura,
        "legenda_figura": legenda_figura,
        "telao": telao,
    }


def erros_no_corte(casos: list[dict[str, Any]], limiar: float) -> tuple[int, int]:
    """Falsos negativos e positivos do modelo servido se o corte fosse `limiar`."""
    negativos = sum(1 for c in casos if c["diagnostico_registrado"] == "Maligno" and c["probabilidade"] < limiar)
    positivos = sum(1 for c in casos if c["diagnostico_registrado"] == "Benigno" and c["probabilidade"] >= limiar)
    return negativos, positivos


def _filme_fase1(contexto: dict[str, Any]) -> dict[str, Any]:
    metricas = contexto["metricas"]
    dataset = metricas["dataset"]
    escolhido = metricas["modelo_escolhido"]
    teste = por_nome(metricas["teste"], "modelo", escolhido)
    malignos = teste["falsos_negativos"] + teste["verdadeiros_positivos"]
    validacao = metricas["validacao_cruzada"]
    shap = metricas["explicabilidade"]["shap_global"][0]["feature"]
    visao = contexto["visao"]
    rede = por_nome(visao["teste"], "modelo", visao["modelo_escolhido"])
    do_zero = por_nome(visao["teste"], "modelo", "CNN do zero")
    achados = rede["malignos_no_teste"] - rede["malignos_nao_detectados"]
    return {
        "id": "fase1",
        "selo": "Fase 1",
        "titulo": "O primeiro diagnóstico",
        "hora": [7.2, 11.5],
        "alas_prontas": [],
        "planos": [
            plano(
                "abertura", "Abertura", "terreno",
                "Todo diagnóstico começa com uma pergunta: este tumor é benigno ou maligno? "
                "Este hospital é fictício, mas os exames são reais. E ele vai ser construído "
                "como o projeto foi: uma fase de cada vez.",
                acoes=("construir:acolhida", "construir:fase1"),
            ),
            plano(
                "desafio", "O desafio", "recepcao",
                "O desafio da Fase 1 é apoiar o diagnóstico de câncer de mama com aprendizado de "
                "máquina. Aqui, um erro pesa mais que os outros: o falso negativo, quando um tumor "
                "maligno é classificado como benigno.",
                robo="recepcao",
            ),
            plano(
                "base", "Os dados", "dados_estante",
                f"A base é a Breast Cancer Wisconsin. São {dataset['amostras']} exames, com "
                f"{dataset['features']} medidas cada. Em azul, {dataset['benignos']} benignos. Em "
                f"laranja, {dataset['malignos']} malignos. Cada pasta da estante é um exame de verdade.",
                acoes=("pastas",),
                evidencias=(
                    evidencia("Exames", dataset["amostras"]),
                    evidencia("Benignos", dataset["benignos"]),
                    evidencia("Malignos", dataset["malignos"]),
                ),
                fonte=FONTE_METRICAS,
            ),
            plano(
                "cofre", "O teste trancado", "dados_cofre",
                f"Antes de qualquer modelo, {dataset['teste']} exames vão para o cofre. Esse é o "
                "conjunto de teste, e ele só abre na avaliação final. Comparar modelos e ajustar "
                "parâmetros acontece por validação cruzada, dentro do treino.",
                robo="dados",
                acoes=("trancar",),
                evidencias=(
                    evidencia("Treino", dataset["treino"]),
                    evidencia("Teste", dataset["teste"]),
                ),
                fonte=FONTE_METRICAS,
            ),
            plano(
                "disputa", "Os modelos", "modelos_podio",
                f"Três modelos disputaram com o mesmo critério. {escolhido} venceu, com F1 de "
                f"{num(validacao[0]['f1'])} na validação cruzada, e tem uma vantagem a mais: dá para "
                "explicar cada decisão.",
                robo="modelos",
                acoes=("podio",),
                evidencias=tuple(evidencia(v["modelo"], v["f1"], 3, "F1 ") for v in validacao),
                fonte=FONTE_METRICAS,
            ),
            plano(
                "resultado", "O resultado", "modelos_podio_perto",
                f"Aberto o cofre, o modelo achou {pct(teste['recall'])} dos tumores malignos. "
                f"Parece muito, mas {teste['falsos_negativos']} de {malignos} passaram. E cada um "
                "deles seria uma paciente sem o alerta.",
                robo="modelos",
                evidencias=(
                    evidencia("Recall", teste["recall"] * 100, 1, sufixo="%"),
                    evidencia("Falsos negativos", teste["falsos_negativos"]),
                ),
                fonte=FONTE_METRICAS,
            ),
            plano(
                "explicar", "Por que o modelo decide", "modelos_shap",
                f"Um médico não confia em caixa preta. Coeficientes, permutação e SHAP mostram o peso "
                f"de cada medida, e {shap} aparece no topo. Cada ponto deste gráfico é um exame.",
                figura="13_shap_beeswarm.png",
                legenda_figura="SHAP: impacto de cada medida na previsão",
                fonte=FONTE_METRICAS,
            ),
            plano(
                "imagem", "A entrega extra", "imagem_sala",
                f"Na entrega extra, o ponto de partida é o ultrassom cru. Treinada do zero, a rede não "
                f"achou nenhum maligno. Com ajuste fino da MobileNetV2, achou {achados} de "
                f"{rede['malignos_no_teste']}. Serve como demonstração de viabilidade, não como "
                "sistema clínico.",
                robo="imagem",
                evidencias=(
                    evidencia("CNN do zero", do_zero["recall_maligno"] * 100, 0, sufixo="%"),
                    evidencia("Ajuste fino", rede["recall_maligno"] * 100, 0, sufixo="%"),
                ),
                fonte=FONTE_VISAO,
            ),
            plano(
                "gradcam", "Onde a rede olhou", "imagem_gradcam",
                "O Grad-CAM mostra a região que pesou na decisão. Mostramos os erros também, para o "
                "resultado não parecer melhor do que é.",
                figura="26_gradcam_erros.png",
                legenda_figura="Grad-CAM nos erros da rede",
                fonte=FONTE_VISAO,
            ),
            plano(
                "fim", "O que ficou em aberto", "planta_fase1",
                f"A Fase 1 entregou o diagnóstico e a explicação. Mas deixou uma pergunta: dá para "
                f"reduzir aqueles {teste['falsos_negativos']} falsos negativos sem olhar o teste? "
                "Essa resposta precisa de uma ala nova.",
            ),
        ],
    }


def _filme_fase2(contexto: dict[str, Any]) -> dict[str, Any]:
    resumo = contexto["resumo"]
    comparacao = contexto["comparacao"]
    servido = contexto["servido"]
    execucao = contexto["execucao"]
    casos = contexto["casos"]
    custo = contexto["custo"]
    narrativas = contexto["narrativas"]
    aprovados = sum(1 for n in narrativas if n["qualidade"]["aprovado"])
    bracos = sorted(resumo["bracos"], key=lambda b: b["custo_medio"])
    final = por_nome(comparacao, "braco", BRACO_COMPLETO)
    referencia = por_nome(comparacao, "braco", BRACO_REFERENCIA)
    auc = [linha["roc_auc"] for linha in comparacao]
    malignos = final["falsos_negativos"] + final["verdadeiros_positivos"]
    razao = num(config.CUSTO_FALSO_NEGATIVO / config.CUSTO_FALSO_POSITIVO, 0)
    negativos_padrao, positivos_padrao = erros_no_corte(casos, LIMIAR_PADRAO)
    limiar = servido["limiar"]
    historico = execucao["historico"]
    return {
        "id": "fase2",
        "selo": "Fase 2",
        "titulo": "A evolução",
        "hora": [14.5, 18.2],
        "alas_prontas": ["acolhida", "fase1"],
        "planos": [
            plano(
                "abertura", "Abertura", "terreno_tarde",
                "A Fase 1 deixou uma pergunta aberta: dá para errar menos justamente onde o erro "
                "custa mais? Para responder, o hospital ganha uma ala nova.",
                acoes=("construir:fase2",),
            ),
            plano(
                "custo", "O custo clínico", "evolucao_ampla",
                f"Primeiro, o problema virou conta. Aqui, um falso negativo custa {razao} vezes mais "
                "que um falso positivo. Com esse custo, um algoritmo genético procura medidas, "
                "hiperparâmetros e limiar ao mesmo tempo.",
                robo="evolucao",
                evidencias=(
                    evidencia("Custo do falso negativo", config.CUSTO_FALSO_NEGATIVO),
                    evidencia("Custo do falso positivo", config.CUSTO_FALSO_POSITIVO),
                ),
                fonte="src/config.py",
            ),
            plano(
                "enxame", "A busca", "enxame_orbita",
                f"Cada esfera é um indivíduo. Esta é a execução que venceu: "
                f"{execucao['configuracao']['populacao']} indivíduos por {len(historico)} gerações. "
                "A nuvem encolhe porque a diversidade real cai, até a população concordar. Foram "
                f"{resumo['execucoes']} execuções, todas dentro do treino.",
                acoes=("enxame",),
                evidencias=(
                    evidencia("Execuções", resumo["execucoes"]),
                    evidencia("Melhor fitness", historico[-1]["melhor_fitness"], 4),
                ),
                fonte=FONTE_EXECUCOES,
            ),
            plano(
                "ablacao", "De onde veio o ganho", "evolucao_convergencia",
                f"Para saber o que importava, cada parte foi testada sozinha. Em validação cruzada, "
                f"mexer só no limiar teve custo de {num(bracos[0]['custo_medio'], 4)}. Mexer só nas "
                f"medidas, {num(bracos[-1]['custo_medio'], 4)}. O ganho não veio de onde parecia.",
                figura="genetic/11_convergencia.png",
                legenda_figura="Convergência da busca, por execução",
                evidencias=tuple(
                    evidencia(b["braco"], b["custo_medio"], 4, "custo ") for b in bracos
                ),
                fonte=FONTE_RESUMO,
            ),
            plano(
                "regua", "O corte", "regua_limiar",
                f"Esta é a cena central. São os {final['n_amostras']} casos do teste, já depois da "
                f"avaliação final. Mesmo modelo, mesmos casos: só o corte anda. Em "
                f"{num(LIMIAR_PADRAO, 1)}, são {negativos_padrao} falsos negativos e "
                f"{positivos_padrao} falsos positivos. Em {num(limiar, 4)}, o limiar calibrado no "
                f"treino, {final['falsos_negativos']} e {final['falsos_positivos']}. Um alarme falso "
                "a mais é o preço de um tumor a menos passando.",
                acoes=("regua",),
            ),
            plano(
                "achado", "O achado honesto", "evolucao_ampla_perto",
                f"O algoritmo não achou um modelo melhor. A AUC ficou entre {num(min(auc))} e "
                f"{num(max(auc))} em todos os braços. E com {malignos} malignos no teste, diferença de "
                f"um ou dois erros é ruído. O modelo servido empata em falsos negativos com a "
                f"referência, usando {final['n_features']} medidas em vez de "
                f"{referencia['n_features']}.",
                robo="evolucao",
                evidencias=(
                    evidencia("Medidas", final["n_features"]),
                    evidencia("AUC", final["roc_auc"], 4),
                    evidencia("Falsos negativos", final["falsos_negativos"]),
                ),
                fonte=FONTE_COMPARACAO,
            ),
            plano(
                "central", "O serviço", "central_ampla",
                "Esse modelo agora atende numa API. Cada chamada vira uma linha de log, sem dado "
                f"identificável. Uma sentinela alerta quando uma medida foge mais de "
                f"{num(config.API_LIMIAR_DE_DESVIO, 0)} desvios padrão do treino. Ela avisa, não "
                "bloqueia: quem decide é o médico.",
                robo="central",
                fonte="src/api/monitoring.py",
            ),
            plano(
                "caminho", "Um caso em movimento", "corredor_portador",
                "Um caso sai da central com a previsão e a explicação, e segue para virar texto.",
                acoes=("portador:central:laudos",),
            ),
            plano(
                "laudos", "O laudo verificado", "laudos_selos",
                f"Um LLM escreve a explicação de cada caso, e nada chega à tela sem passar por quatro "
                f"verificações. {aprovados} de {len(narrativas)} textos foram aprovados. Tudo foi "
                f"gerado num lote fora do ar, por US$ {num(custo['custo_usd'], 3)}, e o serviço "
                "publicado não guarda chave nenhuma.",
                robo="laudos",
                acoes=("selos",),
                evidencias=(
                    evidencia("Textos aprovados", aprovados),
                    evidencia("Custo total", custo["custo_usd"], 3, "US$ "),
                ),
                fonte=FONTE_NARRATIVAS,
            ),
            plano(
                "fim", "O que vem depois", "planta_final",
                "A Fase 2 entregou um modelo calibrado, um serviço monitorado e textos verificados. "
                "O sol está baixando, e o terreno ao lado já foi demarcado.",
            ),
        ],
    }


def _filme_fase3(contexto: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "fase3",
        "selo": "Fase 3",
        "titulo": "O assistente",
        "hora": [18.4, 21.5],
        "rascunho": True,
        "alas_prontas": ["acolhida", "fase1", "fase2"],
        "planos": [
            plano(
                "abertura", "Abertura", "obra_noite",
                "Anoitece, e as luzes do hospital se acendem. Aqui a última ala começa a subir. Este "
                "filme é um rascunho: o que está nesta ala ainda não funciona.",
                acoes=("construir:fase3", "guindaste"),
            ),
            plano(
                "pedido", "O pedido", "obras_ampla",
                "O hospital quer um assistente médico treinado com dados próprios, capaz de responder "
                "dúvidas e sugerir procedimentos a partir dos protocolos internos.",
                robo="obras",
                fonte=FONTE_ENUNCIADO,
            ),
            plano(
                "fluxo", "O fluxo", "obras_fluxo",
                "O enunciado pede fluxos automatizados e seguros. Ao receber um paciente, o sistema "
                "verifica exames pendentes, sugere condutas e alerta a equipe. Cada pedestal vai "
                "virar uma estação de verdade.",
                acoes=("fluxo",),
                fonte=FONTE_ENUNCIADO,
            ),
            plano(
                "limite", "O limite", "obras_validacao",
                "Uma estação vale mais que as outras. O assistente nunca prescreve sem validação "
                "humana, e cada resposta precisa indicar a fonte.",
                acoes=("fluxo_completo",),
                fonte=FONTE_ENUNCIADO,
            ),
            plano(
                "demonstracao", "A demonstração real", "telao",
                "É aqui que entra a gravação verdadeira: o treino da LLM, um fluxo automatizado, "
                "perguntas clínicas com contexto, os logs e a validação das respostas.",
                telao=True,
                fonte=FONTE_ENUNCIADO,
            ),
            plano(
                "fim", "Continua", "planta_noite",
                "O hospital ainda não está pronto. Quando esta ala funcionar, este filme será "
                "gravado de novo.",
            ),
        ],
    }


def filmes(contexto: dict[str, Any]) -> list[dict[str, Any]]:
    return [_filme_fase1(contexto), _filme_fase2(contexto), _filme_fase3(contexto)]
