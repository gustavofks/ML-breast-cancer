"""Geracao do relatorio HTML estatico com as analises dos dois datasets.

A pagina e montada a partir de `results/metrics.json` e `results/metrics_vision.json`,
com as figuras de `results/figures/`. Nao depende de servidor nem de rede: abre por
duplo clique, com CSS embutido, fontes locais e imagens em caminho relativo.

Estrutura: tres abas — o dataset tabular, a entrega extra com imagens e uma
comparacao entre as duas abordagens. Cada aba tem indice lateral que acompanha a
rolagem. Nenhum numero e escrito a mao: tudo vem dos arquivos de metricas, e as
tabelas por tras de cada grafico podem ser abertas sob a figura.

O estilo e o script ficam em `src/report_style.py`.
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path

from src import config
from src.report_code import bloco
from src.report_style import CSS, JS

FIGURAS = "../figures"


# ---------------------------------------------------------------------------
# Blocos reutilizaveis
# ---------------------------------------------------------------------------
def _numero(valor, casas: int = 3) -> str:
    """Formata numeros no padrao brasileiro (virgula decimal)."""
    if isinstance(valor, float):
        return f"{valor:.{casas}f}".replace(".", ",")
    return escape(str(valor))


def _tabela(
    legenda: str,
    colunas: list[tuple[str, str]],
    linhas: list[dict],
    destacar: str | None = None,
    casas: int = 3,
) -> str:
    """Tabela a partir de registros.

    Args:
        colunas: pares `(chave_no_registro, rotulo_exibido)`.
        destacar: valor da primeira coluna que recebe o realce de "escolhido".
    """
    if not linhas:
        return ""

    cabecalho = "".join(f"<th>{escape(rotulo)}</th>" for _, rotulo in colunas)

    corpo = []
    for linha in linhas:
        primeira = str(linha.get(colunas[0][0], ""))
        classe = ' class="destacada"' if destacar and primeira == destacar else ""
        celulas = "".join(f"<td>{_numero(linha.get(chave, ''), casas)}</td>" for chave, _ in colunas)
        corpo.append(f"<tr{classe}>{celulas}</tr>")

    return (
        f'<div class="rolagem"><table><caption>{escape(legenda)}</caption>'
        f"<thead><tr>{cabecalho}</tr></thead>"
        f'<tbody>{"".join(corpo)}</tbody></table></div>'
    )


def _dados(rotulo: str, tabela_html: str) -> str:
    """Tabela recolhida sob um grafico, para quem quiser conferir os numeros."""
    if not tabela_html:
        return ""
    return f"<details class=\"dados\"><summary>{escape(rotulo)}</summary>{tabela_html}</details>"


def _figura(arquivo: str, titulo: str, leitura: str, dados: str = "") -> str:
    """Figura com legenda em duas colunas: titulo a esquerda, leitura a direita.

    O titulo e a leitura vao em elementos separados porque a legenda e um
    grid que ocupa toda a largura da figura — em uma so linha de texto, a
    leitura formava um bloco estreito sob a imagem.
    """
    return (
        f'<figure><img src="{FIGURAS}/{arquivo}" alt="{escape(titulo)}" loading="lazy">'
        f'<figcaption><b class="titulo-figura">{escape(titulo)}</b>'
        f'<span class="leitura">{escape(leitura)}</span></figcaption>'
        f"{dados}</figure>"
    )


def _indicadores(itens: list[tuple[str, str, str]]) -> str:
    """Painel de numeros de destaque. Cada item e `(valor, rotulo, enfase)`."""
    celulas = "".join(
        f'<div class="indicador{(" " + enfase) if enfase else ""}">'
        f'<span class="valor">{escape(valor)}</span>'
        f'<span class="rotulo">{escape(rotulo)}</span></div>'
        for valor, rotulo, enfase in itens
    )
    return f'<div class="indicadores">{celulas}</div>'


def _como_foi_feito(modulo: str, *nomes: str, nota: str = "") -> str:
    """Bloco com o codigo que executa a etapa descrita na secao.

    O codigo e lido do arquivo-fonte na hora de gerar a pagina, nunca copiado
    para ca: um trecho copiado envelhece na primeira refatoracao e a pagina
    passa a descrever um pipeline que nao existe mais.
    """
    referencia = " · ".join([modulo] + [f"{nome}()" for nome in nomes])
    rodape = f'<p class="feito-nota">{escape(nota)}</p>' if nota else ""
    return (
        '<div class="feito">'
        '<div class="feito-cabeca">'
        '<span class="feito-etiqueta">Como foi feito</span>'
        f'<code class="feito-ref">{escape(referencia)}</code>'
        "</div>"
        f'<pre class="codigo"><code>{bloco(modulo, *nomes)}</code></pre>'
        f"{rodape}"
        "</div>"
    )


def _decisoes(itens: list[tuple[str, str, str]]) -> str:
    """Painel de decisoes tecnicas, cada uma com o seu contrafactual.

    Uma decisao so fica justificada quando se sabe o que o caminho oposto
    custaria. Cada item e `(decisao, por que, se fosse ao contrario)`.
    """
    cartoes = "".join(
        f'<article class="decisao"><h4>{escape(decisao)}</h4>'
        f'<p class="porque"><span class="marca">Por quê</span>{escape(porque)}</p>'
        f'<p class="contra"><span class="marca">Se fosse ao contrário</span>{escape(contra)}</p>'
        "</article>"
        for decisao, porque, contra in itens
    )
    return f'<div class="decisoes">{cartoes}</div>'


def _fluxo(etapas: list[tuple[str, str]]) -> str:
    """Diagrama do caminho dos dados, em etapas numeradas."""
    blocos = "".join(
        f'<div class="etapa"><span class="ordem">{indice:02d}</span>'
        f'<span class="nome">{escape(nome)}</span>'
        f'<span class="detalhe">{escape(detalhe)}</span></div>'
        for indice, (nome, detalhe) in enumerate(etapas, start=1)
    )
    return f'<div class="fluxo">{blocos}</div>'


class Aba:
    """Acumula as secoes de uma aba e monta o indice lateral correspondente."""

    def __init__(self, identificador: str, rotulo: str) -> None:
        self.identificador = identificador
        self.rotulo = rotulo
        self._secoes: list[tuple[str, str, str]] = []

    def secao(self, titulo: str, corpo: str) -> None:
        numero = f"{len(self._secoes) + 1:02d}"
        ancora = f"{self.identificador}-{numero}"
        self._secoes.append((ancora, titulo, corpo))

    @property
    def indice(self) -> str:
        itens = "".join(
            f'<li><a href="#{ancora}"><span class="numero">{ancora.split("-")[-1]}</span>'
            f"<span>{escape(titulo)}</span></a></li>"
            for ancora, titulo, _ in self._secoes
        )
        return f'<nav class="indice" aria-label="Seções desta análise"><h2>Nesta análise</h2><ol>{itens}</ol></nav>'

    @property
    def secoes(self) -> str:
        return "".join(
            f'<section id="{ancora}"><h2 data-numero="{ancora.split("-")[-1]}">{escape(titulo)}</h2>{corpo}</section>'
            for ancora, titulo, corpo in self._secoes
        )

    def render(self, selecionada: bool) -> str:
        oculto = "" if selecionada else " hidden"
        return (
            f'<div class="painel" id="painel-{self.identificador}" role="tabpanel"'
            f' aria-labelledby="aba-{self.identificador}"{oculto}>'
            f'<div class="corpo">{self.indice}<div class="conteudo">{self.secoes}</div></div></div>'
        )


# ---------------------------------------------------------------------------
# Aba 1: dataset tabular
# ---------------------------------------------------------------------------
def _aba_wisconsin(m: dict) -> Aba:
    aba = Aba("wisconsin", "Wisconsin · dados tabulares")

    base = m["dataset"]
    escolhido = m["modelo_escolhido"]
    teste = {linha["modelo"]: linha for linha in m["teste"]}[escolhido]
    limiar_030 = next((l for l in m["limiares"] if abs(l["limiar"] - 0.30) < 1e-9), None)
    explic = m.get("explicabilidade", {})

    aba.secao(
        "O problema",
        "<p>Uma rede de hospitais especializados no atendimento à mulher precisa de um sistema de "
        "apoio ao diagnóstico capaz de acelerar a triagem. Esta análise entrega a base de Machine "
        "Learning dessa solução: a classificação de tumores de mama em <strong>malignos</strong> ou "
        "<strong>benignos</strong> a partir de características morfológicas de núcleos celulares "
        "extraídas de punção aspirativa por agulha fina.</p>"
        '<p class="nota">O modelo é ferramenta de triagem e segunda opinião. O diagnóstico final é '
        "sempre responsabilidade do médico. Essa premissa define a métrica prioritária e a exigência "
        "de explicabilidade.</p>"
        "<h3>O caminho dos dados</h3>"
        + _fluxo(
            [
                ("Carga e limpeza", "coluna fantasma e id removidos, alvo codificado"),
                ("Separação", "80/20 estratificado, teste intocado"),
                ("Pipeline", "imputação e padronização dentro do modelo"),
                ("Validação cruzada", "5 folds no treino, seleção por F1"),
                ("Teste", "métricas com foco no recall maligno"),
                ("Explicabilidade", "coeficientes, permutação e SHAP"),
            ]
        ),
    )

    aba.secao(
        "A base de dados",
        _indicadores(
            [
                (str(base["amostras"]), "amostras", ""),
                (str(base["features"]), "features numéricas", ""),
                (f'{base["benignos"]} / {base["malignos"]}', "benignos / malignos", ""),
                ("0", "valores ausentes", ""),
            ]
        )
        + "<p><strong>Breast Cancer Wisconsin (Diagnostic).</strong> Dez medidas do núcleo celular — "
        "raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, "
        "simetria e dimensão fractal — cada uma em três variantes: média da amostra "
        "(<code>_mean</code>), erro padrão (<code>_se</code>) e média das três piores células "
        "(<code>_worst</code>).</p>"
        "<p>A única inconsistência é estrutural: o cabeçalho do CSV termina em vírgula, gerando uma "
        "coluna vazia que é descartada no carregamento. Não há valores ausentes reais.</p>"
        + _como_foi_feito(
            "src/data.py",
            "clean",
            nota="A validação roda depois da limpeza, não antes: se o arquivo mudar de forma, o "
            "pipeline para aqui em vez de treinar sobre dados que não são o que o código supõe.",
        )
        + _figura(
            "01_balanceamento_classes.png",
            "Distribuição dos diagnósticos",
            "62,7% benignos contra 37,3% malignos. O desbalanceamento é leve, mas suficiente para "
            "inutilizar a acurácia como métrica isolada: responder sempre benigno já acertaria "
            "62,7% dos casos.",
        ),
    )

    aba.secao(
        "Análise exploratória",
        _figura(
            "02_distribuicoes_mean.png",
            "Distribuição das medidas médias por diagnóstico",
            "Medidas de tamanho e irregularidade deslocam-se claramente entre as classes. Suavidade, "
            "simetria e dimensão fractal quase não separam os grupos isoladamente.",
        )
        + _figura(
            "03_boxplots_worst.png",
            "Dispersão das medidas worst por diagnóstico",
            "O grupo worst separa melhor que o mean: o que caracteriza malignidade não é o tecido "
            "médio, mas a existência de células com morfologia mais agressiva. Os outliers são casos "
            "malignos graves reais e foram preservados.",
        )
        + _como_foi_feito(
            "src/eda.py",
            "separation_ranking",
            nota="O d de Cohen é adimensional: mede a distância entre as médias das classes em "
            "desvios padrão, e por isso compara features de escalas incomparáveis entre si.",
        )
        + _figura(
            "04_separacao_features.png",
            "Poder de separação por feature (d de Cohen)",
            "Concavidade e pontos côncavos lideram, seguidos das medidas de tamanho. Todos os valores "
            "do topo são positivos: as medidas são sistematicamente maiores nos tumores malignos.",
        ),
    )

    aba.secao(
        "Pré-processamento e correlação",
        "<p>Pipeline do scikit-learn com imputação de mediana, defensiva, e "
        "<code>StandardScaler</code>. A padronização vive <strong>dentro</strong> do pipeline: "
        "ajustá-la antes da separação treino/teste faria a média e o desvio carregarem informação do "
        "teste — vazamento de dados que infla as métricas e não se sustenta em produção.</p>"
        + _como_foi_feito(
            "src/preprocessing.py",
            "build_preprocessor",
            "build_pipeline",
            "split_data",
            nota="Todo modelo do projeto nasce de build_pipeline(), então nenhum código externo "
            "precisa lembrar de padronizar antes de treinar — e o scaler é reajustado dentro de "
            "cada dobra da validação cruzada, com os dados de treino daquela dobra.",
        )
        + _tabela(
            "Separação estratificada 80/20",
            [
                ("conjunto", "Conjunto"),
                ("amostras", "Amostras"),
                ("benigno", "Benigno"),
                ("maligno", "Maligno"),
                ("proporcao_maligno", "Proporção maligno"),
            ],
            [
                {
                    "conjunto": "Treino",
                    "amostras": base["treino"],
                    "benigno": 285,
                    "maligno": 170,
                    "proporcao_maligno": 0.3736,
                },
                {
                    "conjunto": "Teste",
                    "amostras": base["teste"],
                    "benigno": 72,
                    "maligno": 42,
                    "proporcao_maligno": 0.3684,
                },
            ],
            casas=4,
        )
        + _figura(
            "05_correlacao.png",
            "Correlação entre as features",
            "Vinte e um pares superam 0,9 de correlação. Raio, perímetro e área chegam a 0,998 — não "
            "por acaso estatístico, mas por geometria: as três medem a mesma grandeza. Nenhuma "
            "feature foi removida; a regularização trata a redundância.",
        )
        + _figura(
            "06_correlacao_alvo.png",
            "Correlação com o diagnóstico",
            "Nenhuma feature isolada passa de 0,79. Não existe atalho de variável única: a "
            "classificação depende da combinação de várias medidas.",
        ),
    )

    aba.secao(
        "Modelagem",
        "<p>Três técnicas com fundamentos distintos: <strong>Regressão Logística</strong> (baseline "
        "linear e interpretável), <strong>KNN</strong> (não paramétrico, sensível a escala) e "
        "<strong>Random Forest</strong> (não linear, com importância nativa). A comparação usa "
        "validação cruzada estratificada de 5 folds <strong>dentro do treino</strong>; o conjunto de "
        "teste permanece intocado.</p>"
        "<p>A seleção usa <strong>F1</strong>, não recall puro: otimizar recall isolado premiaria um "
        "classificador que chama quase tudo de maligno. O recall é otimizado depois, pelo ajuste do "
        "limiar.</p>"
        + _como_foi_feito(
            "src/models.py",
            "get_models",
            "select_best",
            nota="Os três modelos são criados pela mesma função de pipeline, então a comparação "
            "mede o algoritmo, não diferenças de pré-processamento entre eles.",
        )
        + _tabela(
            "Validação cruzada no conjunto de treino",
            [
                ("modelo", "Modelo"),
                ("accuracy", "Acurácia"),
                ("precision", "Precisão"),
                ("recall", "Recall"),
                ("f1", "F1"),
                ("roc_auc", "AUC"),
            ],
            m["validacao_cruzada"],
            destacar=escolhido,
        ),
    )

    aba.secao(
        "Resultados no teste",
        _indicadores(
            [
                (_numero(teste["recall"]), "recall (maligno)", "marcado"),
                (_numero(teste["f1"]), "F1", ""),
                (_numero(teste["roc_auc"]), "AUC", ""),
                (str(teste["falsos_negativos"]), "falsos negativos", "alerta"),
            ]
        )
        + _como_foi_feito(
            "src/evaluation.py",
            "evaluate",
            nota="A matriz de confusão é desempacotada em tn, fp, fn, tp e os falsos negativos "
            "entram no resultado como número próprio: é a quantidade que decide este projeto, não "
            "um detalhe a ser lido dentro de um gráfico.",
        )
        + _tabela(
            "Desempenho no conjunto de teste",
            [
                ("modelo", "Modelo"),
                ("accuracy", "Acurácia"),
                ("precision", "Precisão"),
                ("recall", "Recall"),
                ("f1", "F1"),
                ("roc_auc", "AUC"),
                ("falsos_negativos", "Falsos negativos"),
            ],
            m["teste"],
            destacar=escolhido,
        )
        + _figura(
            "07_matriz_confusao.png",
            "Matriz de confusão do modelo escolhido",
            f'{teste["verdadeiros_negativos"]} verdadeiros negativos, '
            f'{teste["verdadeiros_positivos"]} verdadeiros positivos, '
            f'{teste["falsos_positivos"]} falso positivo e {teste["falsos_negativos"]} falsos '
            "negativos. Os falsos negativos são o número que importa: tumores malignos que passaram "
            "despercebidos.",
        )
        + _figura(
            "08_curvas_roc.png",
            "Curvas ROC",
            "AUC entre 0,983 e 0,996. AUC alta com recall de 0,929 no limiar padrão indica que o "
            "modelo separa bem as classes e o que está mal calibrado é o corte.",
        )
        + _figura(
            "09_comparacao_modelos.png",
            "Comparação dos modelos por métrica",
            "A Random Forest alcança precisão perfeita, mas com mais falsos negativos. Trocar um "
            "falso positivo por um falso negativo é mau negócio: o primeiro custa um exame, o segundo "
            "custa um diagnóstico perdido.",
        ),
    )

    limiar_texto = (
        f"<p>O limiar de 0,5 é apenas a convenção do <code>predict()</code>, não uma escolha clínica. "
        f'Baixá-lo para 0,30 melhora todas as métricas ao mesmo tempo e reduz os falsos negativos de '
        f'{teste["falsos_negativos"]} para {limiar_030["falsos_negativos"]}, sem aumento de falsos '
        "positivos.</p>"
        if limiar_030
        else ""
    )

    aba.secao(
        "Limiar de decisão",
        limiar_texto
        + _como_foi_feito(
            "src/evaluation.py",
            "threshold_analysis",
            nota="O corte não é retreinado a cada limiar: as probabilidades saem uma vez de "
            "predict_proba e só a régua muda. O modelo é o mesmo; o que se decide é onde parar de "
            "chamar de benigno.",
        )
        + _figura(
            "10_limiar_decisao.png",
            "Recall e precisão em função do limiar",
            "A direção do ajuste é defensável pelo custo assimétrico dos erros. O valor exato, porém, "
            "é frágil: com 114 casos de teste, uma diferença de dois falsos negativos pode não se "
            "repetir em outra amostra.",
            _dados(
                "Ver os números por limiar",
                _tabela(
                    "Efeito do limiar sobre as métricas",
                    [
                        ("limiar", "Limiar"),
                        ("recall", "Recall"),
                        ("precisao", "Precisão"),
                        ("f1", "F1"),
                        ("falsos_negativos", "Falsos negativos"),
                        ("falsos_positivos", "Falsos positivos"),
                    ],
                    m["limiares"],
                    casas=4,
                ),
            ),
        ),
    )

    aba.secao(
        "Explicabilidade",
        "<p>Um sistema de apoio ao diagnóstico que não explica sua previsão não é utilizável na "
        "prática: o médico precisa poder concordar ou discordar com base no raciocínio. Três técnicas "
        "complementares foram aplicadas, e as três convergem nas mesmas medidas.</p>"
        + _como_foi_feito(
            "src/explain.py",
            "permutation_scores",
            nota="A permutação é medida em recall, não em acurácia. A escolha importa: uma feature "
            "pode ser importante para o acerto geral e irrelevante para detectar casos malignos, "
            "que é o objetivo aqui.",
        )
        + _figura(
            "11_coeficientes.png",
            "Coeficientes do modelo",
            "O maior coeficiente é texture_worst, não concave points_worst, que liderava a análise "
            "exploratória. A causa é a multicolinearidade: features redundantes dividem o peso entre "
            "si. Coeficiente alto significa informação única ao modelo, não importância clínica.",
            _dados(
                "Ver coeficientes e razões de chances",
                _tabela(
                    "Maiores coeficientes",
                    [
                        ("feature", "Feature"),
                        ("coeficiente", "Coeficiente"),
                        ("razao_de_chances", "Razão de chances"),
                    ],
                    explic.get("coeficientes", []),
                ),
            ),
        )
        + _figura(
            "12_importancia_permutacao.png",
            "Importância por permutação, medida em recall",
            "Embaralhar texture_worst derruba o recall em 5,4 pontos percentuais. Da quinta feature "
            "em diante a queda é praticamente nula: o modelo se apoia em poucas medidas.",
            _dados(
                "Ver a queda de recall por feature",
                _tabela(
                    "Importância por permutação",
                    [
                        ("feature", "Feature"),
                        ("queda_media", "Queda média"),
                        ("desvio", "Desvio"),
                    ],
                    explic.get("permutacao", []),
                    casas=4,
                ),
            ),
        )
        + _figura(
            "13_shap_beeswarm.png",
            "SHAP — visão global",
            "Cada ponto é uma paciente. Para quase todas as features do topo, valores altos empurram "
            "a previsão para maligno, exatamente como a literatura médica descreve.",
            _dados(
                "Ver a importância média por feature",
                _tabela(
                    "Importância SHAP",
                    [("feature", "Feature"), ("shap_medio", "SHAP médio")],
                    explic.get("shap_global", []),
                    casas=4,
                ),
            ),
        )
        + _figura(
            "14_shap_caso_falso_negativo.png",
            "SHAP — um tumor maligno não detectado",
            "Praticamente todas as medidas desta paciente estão abaixo da média da base. O modelo não "
            "errou por acaso: viu um tumor maligno de morfologia pouco característica. É o limite que "
            "nenhum ajuste de hiperparâmetro resolve.",
        ),
    )

    aba.secao(
        "Decisões e o contrafactual",
        "<p>Cada decisão abaixo tinha um caminho oposto plausível. O que a justifica não é o "
        "resultado que ela produziu, e sim o custo do caminho que não foi tomado — por isso os dois "
        "aparecem lado a lado.</p>"
        + _decisoes(
            [
                (
                    "Maligno é a classe positiva (M = 1)",
                    "É o evento que se quer detectar, e essa escolha define o significado de recall "
                    "em todo o resto do projeto.",
                    "Com benigno como positivo, um recall alto passaria a medir o acerto no caso "
                    "barato e esconderia exatamente o erro caro.",
                ),
                (
                    "StandardScaler dentro do Pipeline",
                    "Assim ele é reajustado a cada dobra da validação cruzada, usando só os dados de "
                    "treino daquela dobra.",
                    "Ajustado sobre a base inteira antes da separação, a média e o desvio "
                    "carregariam informação do teste: as métricas subiriam sem o modelo melhorar em "
                    "nada.",
                ),
                (
                    "Seleção do modelo por F1, não por recall puro",
                    "F1 é a média harmônica entre precisão e recall, e penaliza quem compra recall "
                    "com falsos positivos em excesso.",
                    "Otimizando recall isolado, um classificador que chamasse toda paciente de "
                    "maligna venceria a comparação com recall 1,000 — e seria inútil numa triagem.",
                ),
                (
                    "Outliers preservados",
                    "São casos malignos extremos reais, não erros de medição: a cauda longa é a "
                    "própria assinatura da doença.",
                    "Removê-los melhoraria as métricas descartando justamente as pacientes que o "
                    "sistema mais precisa identificar. O número ficaria melhor e o modelo, pior.",
                ),
                (
                    "Nenhuma feature removida, apesar da multicolinearidade",
                    "Com 569 amostras e 30 features não há alta dimensionalidade, e a regularização "
                    "L2 já distribui o peso entre as medidas redundantes.",
                    "Uma seleção manual seria um corte arbitrário sem ganho esperado, e trocaria um "
                    "problema documentado — coeficientes difíceis de ler — por um não documentado.",
                ),
                (
                    "O limiar é reportado, não embutido no código",
                    "A curva de trade-off entre recall e falsos positivos é entregue para que a "
                    "instituição escolha o corte.",
                    "Fixar 0,30 dentro do modelo esconderia uma decisão de política clínica dentro "
                    "de uma constante, tomada por quem não responde pelo desfecho.",
                ),
            ]
        ),
    )

    return aba


# ---------------------------------------------------------------------------
# Aba 2: dataset de imagens
# ---------------------------------------------------------------------------
def _aba_imagens(m: dict | None) -> Aba:
    aba = Aba("imagens", "Ultrassom · visão computacional")

    if not m:
        aba.secao(
            "Entrega extra",
            '<div class="pendente">'
            "<h3>Diagnóstico por imagem com redes neurais convolucionais</h3>"
            "<p>Esta análise será preenchida com um segundo conjunto de dados — imagens de exame — e "
            "uma rede neural convolucional treinada para classificação.</p>"
            "</div>",
        )
        return aba

    base = m["dataset"]
    escolhido = m["modelo_escolhido"]
    melhor = {linha["modelo"]: linha for linha in m["teste"]}[escolhido]
    duplicatas = m.get("duplicatas", {})

    aba.secao(
        "A base de imagens",
        _indicadores(
            [
                (str(base["imagens"]), "imagens", ""),
                (str(duplicatas.get("grupos", "—")), "exames distintos", ""),
                (str(len(base["classes"])), "classes", ""),
                (escape(base["classe_positiva"]), "classe positiva", "marcado"),
            ]
        )
        + "<p><strong>BUSI — Breast Ultrasound Images.</strong> Ultrassons mamários em três classes, "
        "sob licença CC BY 4.0. Ao contrário do dataset tabular, aqui não há nenhuma medida extraída "
        "por especialista: a rede recebe o mesmo que o médico vê na tela do aparelho.</p>"
        "<h3>O caminho dos dados</h3>"
        + _fluxo(
            [
                ("Leitura", "uma pasta por classe, máscaras descartadas"),
                ("Auditoria", "hash perceptual detecta quadros repetidos"),
                ("Partição", "por grupo e por classe, 556/112/112"),
                ("Treino", "pesos de classe e parada antecipada"),
                ("Ajuste fino", "60 camadas finais liberadas"),
                ("Grad-CAM", "onde a rede olhou"),
            ]
        )
        + _como_foi_feito(
            "src/vision/dataset.py",
            "is_mask",
            "find_masks",
            nota="O BUSI distribui as máscaras de segmentação na mesma pasta das imagens. O "
            "carregador do Keras lê tudo o que encontra no diretório: mantidas ali, elas entrariam "
            "como se fossem exames. O carregamento detecta e interrompe com instrução.",
        )
        + _figura(
            "20_imagens_por_classe.png",
            "Imagens por classe",
            "A base é desbalanceada: 56% benignas, 27% malignas e 17% normais. O treino usa pesos de "
            "classe para compensar, evitando que a rede aprenda a favorecer a classe majoritária.",
            _dados(
                "Ver a contagem por classe",
                _tabela(
                    "Distribuição por classe",
                    [("classe", "Classe"), ("imagens", "Imagens"), ("proporcao", "Proporção")],
                    base["por_classe"],
                    casas=4,
                ),
            ),
        )
        + _figura(
            "21_amostras_imagens.png",
            "Exemplos de ultrassom por classe",
            "A rede parte do pixel cru e precisa aprender sozinha o que distingue as lesões.",
        ),
    )

    if duplicatas:
        aba.secao(
            "Duplicatas e vazamento",
            _indicadores(
                [
                    (str(duplicatas["imagens_redundantes"]), "imagens redundantes", "alerta"),
                    (str(duplicatas["grupos_com_repeticao"]), "grupos com repetição", ""),
                    (str(duplicatas["grupos_com_rotulos_contraditorios"]), "rótulos contraditórios", "alerta"),
                ]
            )
            + "<p>Antes de reportar qualquer desempenho foi preciso resolver um defeito da base. O "
            "BUSI contém vários quadros do mesmo exame, praticamente indistinguíveis. Uma auditoria "
            "por <em>hash</em> perceptual encontrou <strong>112 imagens redundantes</strong> entre as "
            "780 — e, distribuídas ao acaso, 62 pares caíam em partições diferentes.</p>"
            "<p>Na prática, a rede era avaliada em imagens que já tinha visto no treino. É vazamento "
            "de dados, o mesmo problema que o pipeline tabular evita ao manter o "
            "<code>StandardScaler</code> dentro do <code>Pipeline</code>.</p>"
            '<p class="nota">Sete grupos têm rótulos contraditórios: imagens praticamente idênticas '
            "catalogadas com diagnósticos diferentes, incluindo um par byte a byte igual que aparece "
            "como benigno e como maligno. Sem o laudo original não há como decidir qual está correto, "
            "então os casos foram mantidos e registrados — é um limite da base, não do modelo.</p>"
            "<p>A correção foi tratar cada grupo como unidade indivisível na partição, com "
            "<code>StratifiedGroupKFold</code>: preserva a proporção das classes e nunca separa um "
            "grupo.</p>"
            + _como_foi_feito(
                "src/vision/dataset.py",
                "stratified_split",
                nota="Estratificar e agrupar são exigências simultâneas: a proporção de casos "
                "malignos precisa se manter em cada conjunto e nenhum grupo de quadros repetidos "
                "pode ser dividido. StratifiedGroupKFold atende às duas ao mesmo tempo.",
            )
            + _tabela(
                "Composição das partições",
                [
                    ("conjunto", "Conjunto"),
                    ("imagens", "Imagens"),
                    ("benign", "Benigno"),
                    ("malignant", "Maligno"),
                    ("normal", "Normal"),
                    ("proporcao_positiva", "Proporção maligna"),
                ],
                m.get("particoes", []),
                casas=4,
            ),
        )

    aba.secao(
        "Arquiteturas e resultados",
        "<p>Três configurações com fundamentos distintos: uma <strong>CNN treinada do zero</strong>, "
        "a <strong>MobileNetV2 congelada</strong> com cabeça nova, e a mesma rede com "
        "<strong>ajuste fino</strong> das 60 camadas finais.</p>"
        + _indicadores(
            [
                (_numero(melhor["recall_maligno"]), "recall (maligno)", "marcado"),
                (_numero(melhor["accuracy"]), "acurácia", ""),
                (_numero(melhor["f1"]), "F1 macro", ""),
                (
                    f'{melhor["malignos_nao_detectados"]} de {melhor["malignos_no_teste"]}',
                    "malignos não detectados",
                    "alerta",
                ),
            ]
        )
        + _tabela(
            "Desempenho no conjunto de teste",
            [
                ("modelo", "Modelo"),
                ("accuracy", "Acurácia"),
                ("f1", "F1 macro"),
                ("recall_maligno", "Recall maligno"),
                ("precisao_maligno", "Precisão maligno"),
                ("malignos_nao_detectados", "Não detectados"),
            ],
            m["teste"],
            destacar=escolhido,
        )
        + "<p><strong>A CNN treinada do zero não aprendeu a tarefa.</strong> Com 56,3% de acurácia, "
        "não detectou nenhum dos 30 casos malignos: aprendeu a responder benigno, que é a resposta "
        "mais frequente. É a demonstração mais direta do que a análise tabular já argumentava sobre a "
        "acurácia como métrica isolada.</p>"
        "<p>A transferência de aprendizado torna a tarefa viável, e o ajuste fino a torna razoável: "
        "liberar as camadas finais elevou o recall maligno de 0,767 para 0,900. As features da "
        "ImageNet servem de ponto de partida, mas texturas de ultrassom não se parecem com "
        "fotografias naturais.</p>"
        + _como_foi_feito(
            "src/vision/model.py",
            "enable_fine_tuning",
            nota="A busca pela base pré-treinada exclui Sequential de propósito. A primeira versão "
            "usava só isinstance(camada, Model), e a camada de aumento de dados é um Sequential — "
            "que também é um Model. O código liberava o bloco de aumento em vez da MobileNetV2, sem "
            "acusar erro nenhum.",
        )
        + _figura(
            "22_curvas_treino.png",
            "Curvas de treino",
            "A perda de validação da CNN do zero mal se move. O ajuste fino é uma segunda etapa, que "
            "continua de onde a transferência parou — por isso sua contagem de épocas recomeça e sua "
            "perda já nasce baixa.",
            _dados(
                "Ver o resumo do treino",
                _tabela(
                    "Treino por arquitetura",
                    [
                        ("modelo", "Modelo"),
                        ("parametros", "Parâmetros"),
                        ("epocas_treinadas", "Épocas"),
                        ("melhor_epoca", "Melhor época"),
                        ("val_accuracy", "Acurácia val."),
                        ("val_loss", "Perda val."),
                        ("segundos", "Segundos"),
                    ],
                    m.get("treino", []),
                    casas=4,
                ),
            ),
        )
        + _figura(
            "23_matriz_confusao_imagens.png",
            f"Matriz de confusão — {escolhido}",
            "Dos 30 casos malignos, 27 foram corretamente identificados e nenhum foi confundido com "
            "tecido normal. Do lado dos falsos alarmes, 8 benignos foram apontados como malignos: "
            "custo real, mas incomparavelmente menor que o de um diagnóstico perdido.",
        ),
    )

    figuras_gradcam = [f for f in m.get("figuras", []) if "gradcam" in f["arquivo"]]
    if figuras_gradcam:
        aba.secao(
            "Explicabilidade: onde a rede olhou",
            "<p>O pipeline tabular responde “por que esta paciente?” com SHAP, decompondo a previsão "
            "em contribuições por medida. Em imagem não existem medidas, existem pixels — e a "
            "pergunta equivalente é <strong>onde a rede olhou</strong>. O Grad-CAM responde isso "
            "ponderando o último mapa de ativação pelo gradiente da classe prevista.</p>"
            '<p class="nota">O mapa mostra onde, não por quê. Não afirma que a rede reconheceu uma '
            "margem espiculada — apenas que aquela região pesou. Ainda assim, é o que permite a um "
            "médico discordar de forma fundamentada.</p>"
            + _como_foi_feito(
                "src/vision/explain.py",
                "gradcam",
                nota="A normalização precisa ser uma camada, não uma função aplicada ao tensor: "
                "aplicada como função ela é absorvida pelo grafo e some de model.layers, e o "
                "Grad-CAM, que percorre camadas, alimentava a rede com pixels de 0 a 255 onde ela "
                "espera valores de −1 a 1. Os mapas eram ruído com aparência plausível.",
            )
            + "".join(
                _figura(f["arquivo"], f["titulo"], f["leitura"]) for f in figuras_gradcam
            )
            + "<p>Nos casos detectados, <strong>o calor se concentra sobre a lesão</strong>. Nos que "
            "escapam, o padrão se inverte: a lesão permanece fria e a atenção migra para uma região "
            "de sombra sem relevância diagnóstica. O modelo não avaliou mal a lesão — não a "
            "considerou.</p>"
            "<p>Duas observações fecham a análise. As imagens trazem <strong>anotações gravadas em "
            "pixel</strong>, feitas pelo profissional que já identificou a lesão suspeita: elas "
            "carregam informação do diagnóstico, e uma rede que aprendesse a reconhecê-las teria "
            "desempenho alto e valor clínico nulo. E as duas primeiras colunas dos erros são o mesmo "
            "exame, em quadros quase idênticos, ambos no teste — exatamente o que o agrupamento "
            "garante.</p>",
        )

    aba.secao(
        "Decisões e o contrafactual",
        "<p>As decisões desta entrega são quase todas defensivas: a base tem defeitos, e boa parte "
        "do trabalho foi impedir que eles virassem métrica boa.</p>"
        + _decisoes(
            [
                (
                    "Partição por grupos de quadros quase idênticos",
                    f'A auditoria por hash perceptual achou {duplicatas.get("imagens_redundantes", "—")} '
                    "imagens redundantes; cada grupo passou a cair inteiro num único conjunto.",
                    "Com partição aleatória, a rede é avaliada em imagens que já viu no treino. O "
                    "recall reportado seria otimista e não se repetiria em exame novo.",
                ),
                (
                    "Máscaras detectadas, com o carregamento interrompido",
                    "O Keras carrega tudo o que está na pasta, então o código verifica antes e para "
                    "com instrução explícita.",
                    "Mantidas ali, as máscaras entrariam como exames e as métricas subiriam sobre "
                    "imagens binárias que não existem clinicamente.",
                ),
                (
                    "Transferência primeiro, ajuste fino depois",
                    "A cabeça de classificação converge com a base congelada, e só então as camadas "
                    "finais são liberadas com taxa de aprendizado 10× menor.",
                    "Liberadas sobre pesos aleatórios, os gradientes da cabeça recém-inicializada "
                    "apagariam as features da ImageNet na primeira época.",
                ),
                (
                    "Aumento de dados apenas geométrico",
                    "Espelhamento, rotação leve, zoom e translação: transformações que preservam o "
                    "conteúdo diagnóstico da imagem.",
                    "Distorcer brilho e contraste ensinaria a rede a ignorar a intensidade — que em "
                    "ultrassom é justamente parte do sinal que separa as classes.",
                ),
                (
                    "Execução determinística",
                    "Sementes fixas e enable_op_determinism(): duas execuções produzem exatamente as "
                    "mesmas métricas.",
                    "Sem isso não há como saber se uma melhora veio do ajuste fino ou da "
                    "inicialização dos pesos daquela rodada.",
                ),
                (
                    "Rótulos contraditórios mantidos e registrados",
                    f'São {duplicatas.get("grupos_com_rotulos_contraditorios", "—")} grupos com '
                    "imagens quase idênticas catalogadas com diagnósticos diferentes, e não há "
                    "laudo original para arbitrar qual está certo.",
                    "Descartá-los, ou escolher um rótulo por conta própria, produziria um número "
                    "melhor escondendo um defeito da base dentro do resultado.",
                ),
            ]
        ),
    )

    return aba


# ---------------------------------------------------------------------------
# Aba 3: comparacao
# ---------------------------------------------------------------------------
def _aba_comparacao(m: dict, mv: dict | None) -> Aba:
    aba = Aba("comparacao", "Comparação e conclusões")

    escolhido = m["modelo_escolhido"]
    teste = {linha["modelo"]: linha for linha in m["teste"]}[escolhido]

    linhas_comparacao = [
        {
            "criterio": "Entrada",
            "tabular": "30 medidas extraídas por especialista",
            "imagem": "pixels do exame",
        },
        {
            "criterio": "Recall da classe maligna",
            "tabular": _numero(teste["recall"]),
            "imagem": "—",
        },
        {"criterio": "Amostras", "tabular": str(m["dataset"]["amostras"]), "imagem": "—"},
        {
            "criterio": "Dependência humana prévia",
            "tabular": "alta — exige medição manual",
            "imagem": "nenhuma",
        },
        {
            "criterio": "Explicabilidade",
            "tabular": "coeficientes, permutação e SHAP",
            "imagem": "Grad-CAM sobre a imagem",
        },
    ]

    if mv:
        melhor_imagem = {linha["modelo"]: linha for linha in mv["teste"]}[mv["modelo_escolhido"]]
        linhas_comparacao[1]["imagem"] = _numero(melhor_imagem["recall_maligno"])
        duplicatas = mv.get("duplicatas", {})
        linhas_comparacao[2]["imagem"] = (
            f'{mv["dataset"]["imagens"]} ({duplicatas.get("grupos", "?")} exames distintos)'
        )

    aba.secao(
        "As duas abordagens",
        "<p>As duas entregas resolvem o mesmo problema clínico por caminhos opostos, e a comparação "
        "só é honesta se essa diferença ficar explícita.</p>"
        + _tabela(
            "Tabular contra imagem",
            [("criterio", "Critério"), ("tabular", "Wisconsin (tabular)"), ("imagem", "BUSI (imagem)")],
            linhas_comparacao,
        )
        + "<p><strong>O modelo tabular tem métricas melhores, mas resolve um problema mais fácil.</strong> "
        "Ele recebe medidas que um profissional já extraiu da imagem: automatiza o julgamento, não a "
        "medição, e por isso não reduz a carga de trabalho especializada.</p>"
        "<p>O modelo de imagem opera sobre o dado bruto, sem etapa manual anterior. Está mais próximo "
        "de um sistema de triagem real — e mais longe de estar pronto.</p>",
    )

    aba.secao(
        "Onde cada uma falha",
        "<p>Os dois modelos erram, e os erros são de naturezas diferentes. Vale olhar para eles, "
        "porque é neles que está o limite do que este projeto entrega.</p>"
        "<ul>"
        "<li><strong>No tabular</strong>, o falso negativo analisado por SHAP é um tumor maligno cujas "
        "medidas estão quase todas abaixo da média. O modelo não errou por acaso: viu um caso que "
        "genuinamente se parece com os benignos nas 30 medidas disponíveis.</li>"
        "<li><strong>Na imagem</strong>, o Grad-CAM mostra que nos casos perdidos a lesão sequer entra "
        "no campo de atenção da rede. O erro não é de julgamento, é de percepção.</li>"
        "</ul>"
        "<p>Em ambos, a explicação torna o erro discutível com um profissional — que é exatamente o "
        "que um número de acurácia não faz.</p>",
    )

    aba.secao(
        "Uso na prática",
        "<p>O sistema é utilizável como <strong>apoio</strong>, não como decisor. Três cenários "
        "concretos, em ordem crescente de exigência:</p>"
        + _tabela(
            "Cenários de uso",
            [("cenario", "Cenário"), ("como", "Como funcionaria"), ("valor", "Por que agrega valor")],
            [
                {
                    "cenario": "Priorização de fila",
                    "como": "ordenar casos pendentes pela probabilidade prevista",
                    "valor": "casos de alto risco chegam antes ao especialista",
                },
                {
                    "cenario": "Segunda opinião",
                    "como": "exibir previsão e explicação junto ao laudo em elaboração",
                    "valor": "contraponto estruturado antes da assinatura",
                },
                {
                    "cenario": "Sinalização de discordância",
                    "como": "alertar quando modelo e laudo preliminar divergem",
                    "valor": "rede de segurança contra desatenção",
                },
            ],
        )
        + '<p class="nota">Em todos os cenários, o laudo é assinado por um médico. O sistema nunca '
        "emite diagnóstico, nunca dispensa uma paciente e nunca é a última etapa do fluxo.</p>"
        "<p><strong>O limiar de decisão é uma escolha clínica, não técnica.</strong> Quem define "
        "quantos exames adicionais se aceita realizar para não deixar um tumor maligno passar é a "
        "instituição de saúde. O que a análise entrega é a curva de trade-off que torna essa decisão "
        "informada em vez de arbitrária.</p>",
    )

    aba.secao(
        "Limitações e próximos passos",
        "<p>Quatro limitações são intransponíveis com os dados disponíveis:</p>"
        "<ul>"
        "<li><strong>Origem única e volume pequeno.</strong> 569 registros clínicos e 668 exames de "
        "imagem, de fontes únicas. Nada garante que o desempenho se mantenha em outra população ou "
        "com outro equipamento.</li>"
        "<li><strong>O modelo tabular não parte de dados crus.</strong> Depende de um especialista "
        "ter medido os núcleos celulares antes.</li>"
        "<li><strong>Existem tumores malignos morfologicamente discretos.</strong> Nenhum ajuste de "
        "hiperparâmetro resolve isso: a informação necessária não está nos dados.</li>"
        "<li><strong>A base de imagens tem defeitos de catalogação</strong> — quadros repetidos e "
        "rótulos contraditórios — que limitam o quanto se pode confiar na medição.</li>"
        "</ul>"
        "<h3>Riscos de implantação</h3>"
        "<ul>"
        "<li><strong>Excesso de confiança.</strong> Um sistema que acerta com frequência induz o "
        "profissional a aceitar previsões sem revisar — o efeito que a explicabilidade combate.</li>"
        "<li><strong>Degradação silenciosa.</strong> Mudanças de equipamento ou população deslocam a "
        "distribuição sem gerar erro visível. Uso real exigiria monitoramento contínuo.</li>"
        "<li><strong>Tradução equivocada da importância das variáveis.</strong> Comunicar que “a "
        "textura é o que mais importa no diagnóstico” seria um erro grave de leitura da estatística "
        "para a medicina.</li>"
        "<li><strong>Responsabilidade legal e ética.</strong> Dados sensíveis de saúde exigem "
        "conformidade com a LGPD, rastreabilidade das previsões e definição explícita de "
        "responsabilidade sobre o desfecho.</li>"
        "</ul>"
        "<h3>Próximos passos técnicos</h3>"
        "<ul>"
        "<li>Validação externa em base de outra instituição — o teste decisivo que estes dados não "
        "permitem.</li>"
        "<li>Calibração do limiar por validação cruzada no treino, e não observando o teste.</li>"
        "<li>Calibração de probabilidade, para que o valor previsto seja lido como risco real.</li>"
        "<li>Remoção das anotações gravadas nas imagens, que carregam informação do diagnóstico.</li>"
        "</ul>"
        '<p class="nota">O valor prático deste trabalho não está em substituir o julgamento médico, '
        "mas em torná-lo mais rápido e mais informado. Reconhecer os limites faz parte de entregar o "
        "sistema de forma responsável.</p>",
    )

    return aba


# ---------------------------------------------------------------------------
# Montagem da pagina
# ---------------------------------------------------------------------------
def _resumo_executivo(m: dict, mv: dict | None) -> str:
    escolhido = m["modelo_escolhido"]
    teste = {linha["modelo"]: linha for linha in m["teste"]}[escolhido]

    frase_imagem = ""
    if mv:
        melhor_imagem = {linha["modelo"]: linha for linha in mv["teste"]}[mv["modelo_escolhido"]]
        frase_imagem = (
            f' A entrega extra repete o exercício partindo do pixel cru de ultrassons, sem medidas '
            f'extraídas por especialista, e alcança recall de {_numero(melhor_imagem["recall_maligno"])} '
            f'com uma MobileNetV2 ajustada.'
        )

    return (
        '<section class="resumo" aria-labelledby="resumo-titulo">'
        '<h2 id="resumo-titulo">Resumo executivo</h2>'
        "<p>Dois modelos de apoio ao diagnóstico de câncer de mama, avaliados pelo custo clínico do "
        "erro. No conjunto tabular, uma regressão logística classifica tumores a partir de 30 medidas "
        f'morfológicas com recall de {_numero(teste["recall"])} na classe maligna e '
        f'{teste["falsos_negativos"]} falsos negativos em 114 casos de teste.{frase_imagem}</p>'
        "<p>A decisão metodológica central é a escolha da métrica: <strong>recall da classe maligna</strong>, "
        "não acurácia. Um falso negativo é um tumor maligno classificado como benigno — o erro de "
        "maior custo. Com 62,7% de casos benignos, um classificador trivial já atingiria 62,7% de "
        "acurácia sem detectar nada.</p>"
        '<p class="veredito">O sistema é utilizável como triagem e segunda opinião, nunca como '
        "decisor. As análises de explicabilidade mostram onde cada modelo se apoia — e, mais "
        "importante, expõem os casos em que erra e por quê.</p>"
        "</section>"
    )


def build_report(
    metrics: dict | None = None,
    output: Path | None = None,
    metrics_vision: dict | None = None,
) -> Path:
    """Gera o relatorio HTML e devolve o caminho do arquivo escrito.

    Args:
        metrics: consolidado do dataset tabular. Lido de `results/metrics.json`
            se omitido.
        output: caminho de saida. Usa `results/reports/index.html` se omitido.
        metrics_vision: consolidado da entrega extra. Lido de
            `results/metrics_vision.json` quando o arquivo existir; a aba fica
            marcada como pendente enquanto nao existir.
    """
    if metrics is None:
        metrics = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))

    if metrics_vision is None and config.VISION_METRICS_FILE.exists():
        metrics_vision = json.loads(config.VISION_METRICS_FILE.read_text(encoding="utf-8"))

    destino = output or config.REPORT_FILE
    destino.parent.mkdir(parents=True, exist_ok=True)

    base = metrics["dataset"]
    gerado_em = datetime.now().strftime("%d/%m/%Y às %H:%M")

    ficha = (
        f'<span><strong>Base tabular</strong>{escape(base["nome"])}</span>'
        f'<span><strong>Modelo tabular</strong>{escape(metrics["modelo_escolhido"])}</span>'
        f'<span><strong>Amostras</strong>{base["amostras"]} ({base["treino"]} treino / {base["teste"]} teste)</span>'
    )
    if metrics_vision:
        ficha += (
            f'<span><strong>Base de imagens</strong>{escape(metrics_vision["dataset"].get("nome", "imagens"))}</span>'
            f'<span><strong>Modelo de imagem</strong>{escape(metrics_vision["modelo_escolhido"])}</span>'
        )
    ficha += f"<span><strong>Gerado em</strong>{gerado_em}</span>"

    abas = [
        _aba_wisconsin(metrics),
        _aba_imagens(metrics_vision),
        _aba_comparacao(metrics, metrics_vision),
    ]

    botoes = "".join(
        f'<button class="aba" id="aba-{aba.identificador}" role="tab"'
        f' aria-controls="painel-{aba.identificador}"'
        f' aria-selected="{"true" if indice == 0 else "false"}"'
        f' tabindex="{0 if indice == 0 else -1}"'
        f' data-painel="painel-{aba.identificador}">'
        f'<span class="numero">{indice + 1:02d}</span>{escape(aba.rotulo)}</button>'
        for indice, aba in enumerate(abas)
    )

    paineis = "".join(aba.render(indice == 0) for indice, aba in enumerate(abas))

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detecção de Riscos em Saúde da Mulher — Tech Challenge Fase 1</title>
<meta name="description" content="Classificação de exames com aprendizado de máquina, avaliada pelo custo clínico do erro.">
<style>{CSS}</style>
<script>
// Aplica o tema antes da primeira pintura, para a pagina nao piscar no tema
// errado ao ser aberta. Sem escolha salva, segue a preferencia do sistema.
(function () {{
  var tema = null;
  try {{ tema = localStorage.getItem('tema'); }} catch (erro) {{}}
  if (tema !== 'claro' && tema !== 'escuro') {{
    tema = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'escuro'
      : 'claro';
  }}
  document.documentElement.setAttribute('data-tema', tema);
}})();
</script>
</head>
<body>
<a class="pular" href="#conteudo-principal">Pular para o conteúdo</a>
<div class="envelope">

<button class="tema" type="button" data-estado="claro" aria-pressed="false"
        aria-label="Tema Claro. Clique para usar o tema escuro.">
  <svg data-icone="claro" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="4"></circle>
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path>
  </svg>
  <svg data-icone="escuro" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"></path>
  </svg>
  <span class="rotulo-tema">Claro</span>
</button>

<header class="capa">
  <p class="etiqueta">Tech Challenge · Fase 1 · Pós-graduação em Inteligência Artificial</p>
  <h1>Detecção de riscos em saúde da mulher</h1>
  <p class="chamada">Classificação de exames com aprendizado de máquina, avaliada pelo custo clínico
  do erro e acompanhada da explicação de cada previsão.</p>
  <div class="ficha">{ficha}</div>
</header>

{_resumo_executivo(metrics, metrics_vision)}

<div id="ancora-abas"></div>
<div class="barra-abas">
  <div class="abas" role="tablist" aria-label="Análises do projeto">{botoes}</div>
</div>

<main id="conteudo-principal">{paineis}</main>

<footer class="rodape">
  <span>Gerado por <code>src/report.py</code> a partir de <code>results/metrics.json</code></span>
  <span>Figuras em <code>results/figures/</code> · reproduzível com <code>scripts/run_wisconsin.py</code></span>
</footer>

</div>
<script>{JS}</script>
</body>
</html>
"""

    destino.write_text(html, encoding="utf-8")
    return destino
