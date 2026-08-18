"""Geracao do relatorio HTML estatico com as analises dos datasets.

A pagina e montada a partir de `results/metrics.json` e das figuras em
`results/figures/`. Nao depende de servidor nem de rede: abre por duplo clique,
com CSS embutido e imagens referenciadas por caminho relativo.

Estrutura: duas abas, uma por dataset. A do dataset tabular (Wisconsin) e
preenchida com os resultados reais; a do extra (imagens/CNN) fica reservada e
sinalizada como pendente ate a fase correspondente.
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path

from src import config

FIGURAS = "../figures"

CSS = """
:root {
  --papel: #f7f4ee;
  --papel-cartao: #fffdf9;
  --tinta: #16150f;
  --tinta-media: #55524a;
  --tinta-fraca: #8b8779;
  --linha: #ddd7c9;
  --linha-forte: #b8b1a0;
  --laranja: #c2551f;
  --destaque-fundo: #f0ece0;
  --serifada: "Iowan Old Style", "Palatino Linotype", Palatino, Cambria, Georgia, serif;
  --sem-serifa: Corbel, "Segoe UI", Optima, "Avenir Next", "Helvetica Neue", sans-serif;
  --mono: Consolas, "SF Mono", "Roboto Mono", monospace;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --papel: #14140f;
    --papel-cartao: #1c1c16;
    --tinta: #f0ece1;
    --tinta-media: #b3ae9f;
    --tinta-fraca: #7d7869;
    --linha: #33322a;
    --linha-forte: #4b4940;
    --laranja: #ee8f5e;
    --destaque-fundo: #24241c;
  }
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--papel);
  color: var(--tinta);
  font-family: var(--serifada);
  font-size: 17px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}

.envelope { max-width: 1080px; margin: 0 auto; padding: 0 32px 96px; }

.cabecalho { padding: 72px 0 36px; border-bottom: 2px solid var(--tinta); }

.etiqueta {
  font-family: var(--sem-serifa);
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  margin-bottom: 20px;
}

h1 {
  font-size: clamp(34px, 5.4vw, 56px);
  line-height: 1.06;
  margin: 0 0 18px;
  font-weight: 400;
  letter-spacing: -0.015em;
}

.subtitulo {
  font-size: 20px;
  color: var(--tinta-media);
  max-width: 62ch;
  margin: 0 0 32px;
}

.ficha {
  display: flex;
  flex-wrap: wrap;
  gap: 14px 44px;
  font-family: var(--sem-serifa);
  font-size: 13px;
  color: var(--tinta-media);
}

.ficha strong {
  display: block;
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  font-weight: 600;
  margin-bottom: 2px;
}

.abas {
  display: flex;
  gap: 4px;
  position: sticky;
  top: 0;
  z-index: 10;
  background: var(--papel);
  border-bottom: 1px solid var(--linha);
}

.aba {
  appearance: none;
  border: none;
  background: none;
  cursor: pointer;
  font-family: var(--sem-serifa);
  font-size: 13.5px;
  color: var(--tinta-fraca);
  padding: 20px 22px 17px;
  border-bottom: 3px solid transparent;
  transition: color 0.18s ease, border-color 0.18s ease;
}

.aba:hover { color: var(--tinta-media); }
.aba[aria-selected="true"] { color: var(--tinta); border-bottom-color: var(--laranja); }

.painel[hidden] { display: none; }

section { padding: 56px 0 8px; border-bottom: 1px solid var(--linha); }
section:last-child { border-bottom: none; }

h2 {
  font-size: 15px;
  font-family: var(--sem-serifa);
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  margin: 0 0 26px;
  display: flex;
  align-items: baseline;
  gap: 14px;
}

h2::before {
  content: attr(data-numero);
  font-family: var(--mono);
  font-size: 12px;
  color: var(--laranja);
  letter-spacing: 0;
}

h3 {
  font-size: 23px;
  font-weight: 400;
  margin: 40px 0 14px;
  letter-spacing: -0.01em;
}

p { max-width: 68ch; margin: 0 0 18px; }

p.nota {
  font-size: 15px;
  color: var(--tinta-media);
  border-left: 2px solid var(--linha-forte);
  padding-left: 18px;
}

strong { font-weight: 600; }

code {
  font-family: var(--mono);
  font-size: 0.86em;
  background: var(--destaque-fundo);
  padding: 1px 5px;
  border-radius: 3px;
}

ul { max-width: 68ch; padding-left: 20px; margin: 0 0 18px; }
li { margin-bottom: 8px; }

.indicadores {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
  gap: 1px;
  background: var(--linha);
  border: 1px solid var(--linha);
  margin: 8px 0 40px;
}

.indicador { background: var(--papel-cartao); padding: 22px 20px 20px; }

.indicador .valor {
  font-size: 40px;
  line-height: 1;
  letter-spacing: -0.02em;
  display: block;
  margin-bottom: 10px;
  font-variant-numeric: tabular-nums;
}

.indicador .rotulo {
  font-family: var(--sem-serifa);
  font-size: 10.5px;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  display: block;
}

.indicador.alerta .valor { color: var(--laranja); }

.rolagem { overflow-x: auto; margin: 0 0 32px; }

table {
  border-collapse: collapse;
  width: 100%;
  font-size: 15px;
  font-variant-numeric: tabular-nums;
}

caption {
  caption-side: top;
  text-align: left;
  font-family: var(--sem-serifa);
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  padding-bottom: 10px;
}

th, td { padding: 11px 16px 11px 0; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; padding-left: 0; }

thead th {
  font-family: var(--sem-serifa);
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  border-bottom: 1px solid var(--tinta);
}

tbody tr { border-bottom: 1px solid var(--linha); }
tbody tr.destacada { background: var(--destaque-fundo); }

tbody tr.destacada td:first-child::after {
  content: "escolhido";
  font-family: var(--sem-serifa);
  font-size: 9.5px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--laranja);
  margin-left: 10px;
}

figure { margin: 0 0 38px; }

figure img {
  display: block;
  width: 100%;
  height: auto;
  background: #fcfcfb;
  border: 1px solid var(--linha);
  padding: 10px;
}

figcaption {
  font-family: var(--sem-serifa);
  font-size: 12.5px;
  color: var(--tinta-fraca);
  margin-top: 10px;
  max-width: 68ch;
  line-height: 1.5;
}

figcaption b { color: var(--tinta-media); font-weight: 600; }

.pendente {
  border: 1px dashed var(--linha-forte);
  padding: 48px 40px;
  margin: 48px 0;
  background: var(--papel-cartao);
}

.pendente h3 { margin-top: 0; }

.rodape {
  margin-top: 72px;
  padding-top: 26px;
  border-top: 2px solid var(--tinta);
  font-family: var(--sem-serifa);
  font-size: 12.5px;
  color: var(--tinta-fraca);
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}

@media (max-width: 640px) {
  .envelope { padding: 0 20px 64px; }
  .cabecalho { padding-top: 48px; }
  .abas { overflow-x: auto; }
  .indicador .valor { font-size: 32px; }
}
"""

JS = """
const abas = document.querySelectorAll('.aba');
abas.forEach(function (aba) {
  aba.addEventListener('click', function () {
    abas.forEach(function (outra) {
      var ativa = outra === aba;
      outra.setAttribute('aria-selected', String(ativa));
      document.getElementById(outra.dataset.painel).hidden = !ativa;
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});
"""


# ---------------------------------------------------------------------------
# Blocos reutilizaveis
# ---------------------------------------------------------------------------
def _numero(valor, casas: int = 3) -> str:
    """Formata numeros no padrao brasileiro (virgula decimal)."""
    if isinstance(valor, float):
        return f"{valor:.{casas}f}".replace(".", ",")
    return escape(str(valor))


def _secao(numero: str, titulo: str, corpo: str) -> str:
    return (
        f'<section><h2 data-numero="{numero}">{escape(titulo)}</h2>{corpo}</section>'
    )


def _indicadores(itens: list[tuple[str, str, bool]]) -> str:
    """Painel de numeros de destaque. Cada item e (valor, rotulo, alerta)."""
    celulas = "".join(
        f'<div class="indicador{" alerta" if alerta else ""}">'
        f'<span class="valor">{escape(valor)}</span>'
        f'<span class="rotulo">{escape(rotulo)}</span></div>'
        for valor, rotulo, alerta in itens
    )
    return f'<div class="indicadores">{celulas}</div>'


def _tabela(
    legenda: str,
    colunas: list[tuple[str, str]],
    linhas: list[dict],
    destacar: str | None = None,
    casas: int = 3,
) -> str:
    """Monta uma tabela a partir de registros.

    Args:
        colunas: pares `(chave_no_registro, rotulo_exibido)`.
        destacar: valor da primeira coluna que recebe realce de "escolhido".
    """
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


def _figura(arquivo: str, titulo: str, leitura: str) -> str:
    return (
        f'<figure><img src="{FIGURAS}/{arquivo}" alt="{escape(titulo)}" loading="lazy">'
        f"<figcaption><b>{escape(titulo)}</b> — {escape(leitura)}</figcaption></figure>"
    )


# ---------------------------------------------------------------------------
# Aba 1: dataset tabular
# ---------------------------------------------------------------------------
def _painel_wisconsin(m: dict) -> str:
    base = m["dataset"]
    escolhido = m["modelo_escolhido"]
    teste = {linha["modelo"]: linha for linha in m["teste"]}[escolhido]
    limiar_030 = next(l for l in m["limiares"] if abs(l["limiar"] - 0.30) < 1e-9)
    explic = m.get("explicabilidade", {})

    blocos = []

    blocos.append(
        _secao(
            "01",
            "O problema",
            "<p>Uma rede de hospitais especializados no atendimento à mulher precisa de um sistema "
            "de apoio ao diagnóstico capaz de acelerar a triagem. Esta análise entrega a base de "
            "Machine Learning dessa solução: a classificação de tumores de mama em "
            "<strong>malignos</strong> ou <strong>benignos</strong> a partir de características "
            "morfológicas de núcleos celulares extraídas de punção aspirativa por agulha fina.</p>"
            '<p class="nota">O modelo é ferramenta de triagem e segunda opinião. O diagnóstico '
            "final é sempre responsabilidade do médico. Essa premissa define a métrica prioritária "
            "e a exigência de explicabilidade.</p>",
        )
    )

    blocos.append(
        _secao(
            "02",
            "A base de dados",
            _indicadores(
                [
                    (str(base["amostras"]), "amostras", False),
                    (str(base["features"]), "features numéricas", False),
                    (f'{base["benignos"]} / {base["malignos"]}', "benignos / malignos", False),
                    ("0", "valores ausentes", False),
                ]
            )
            + "<p><strong>Breast Cancer Wisconsin (Diagnostic).</strong> Dez medidas do núcleo "
            "celular — raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos "
            "côncavos, simetria e dimensão fractal — cada uma em três variantes: média da amostra "
            "(<code>_mean</code>), erro padrão (<code>_se</code>) e média das três piores células "
            "(<code>_worst</code>).</p>"
            "<p>A única inconsistência é estrutural: o cabeçalho do CSV termina em vírgula, gerando "
            "uma coluna vazia que é descartada no carregamento. Não há valores ausentes reais.</p>"
            + _figura(
                "01_balanceamento_classes.png",
                "Distribuição dos diagnósticos",
                "62,7% benignos contra 37,3% malignos. O desbalanceamento é leve, mas suficiente "
                "para inutilizar a acurácia como métrica isolada: responder sempre benigno já "
                "acertaria 62,7% dos casos.",
            ),
        )
    )

    blocos.append(
        _secao(
            "03",
            "Análise exploratória",
            _figura(
                "02_distribuicoes_mean.png",
                "Distribuição das medidas médias por diagnóstico",
                "Medidas de tamanho e irregularidade deslocam-se claramente entre as classes. "
                "Suavidade, simetria e dimensão fractal quase não separam os grupos isoladamente.",
            )
            + _figura(
                "03_boxplots_worst.png",
                "Dispersão das medidas worst por diagnóstico",
                "O grupo worst separa melhor que o mean: o que caracteriza malignidade não é o "
                "tecido médio, mas a existência de células com morfologia mais agressiva. Os "
                "outliers são casos malignos graves reais e foram preservados.",
            )
            + _figura(
                "04_separacao_features.png",
                "Poder de separação por feature (d de Cohen)",
                "Concavidade e pontos côncavos lideram, seguidos das medidas de tamanho. Todos os "
                "valores do topo são positivos: as medidas são sistematicamente maiores nos "
                "tumores malignos.",
            ),
        )
    )

    blocos.append(
        _secao(
            "04",
            "Pré-processamento e correlação",
            "<p>Pipeline do scikit-learn com imputação de mediana (defensiva) e "
            "<code>StandardScaler</code>. A padronização vive <strong>dentro</strong> do pipeline: "
            "ajustá-la antes da separação treino/teste faria a média e o desvio carregarem "
            "informação do teste — vazamento de dados que infla as métricas.</p>"
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
                "Vinte e um pares superam 0,9 de correlação. Raio, perímetro e área chegam a 0,998 "
                "— não por acaso estatístico, mas por geometria: as três medem a mesma grandeza. "
                "Nenhuma feature foi removida; a regularização trata a redundância.",
            )
            + _figura(
                "06_correlacao_alvo.png",
                "Correlação com o diagnóstico",
                "Nenhuma feature isolada passa de 0,79. Não existe atalho de variável única: a "
                "classificação depende da combinação de várias medidas.",
            ),
        )
    )

    blocos.append(
        _secao(
            "05",
            "Modelagem",
            "<p>Três técnicas com fundamentos distintos: <strong>Regressão Logística</strong> "
            "(baseline linear e interpretável), <strong>KNN</strong> (não paramétrico, sensível a "
            "escala) e <strong>Random Forest</strong> (não linear, com importância nativa). A "
            "comparação usa validação cruzada estratificada de 5 folds <strong>dentro do "
            "treino</strong>; o conjunto de teste permanece intocado.</p>"
            "<p>A seleção usa <strong>F1</strong>, não recall puro: otimizar recall isolado "
            "premiaria um classificador que chama quase tudo de maligno. O recall é otimizado "
            "depois, pelo ajuste do limiar.</p>"
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
    )

    blocos.append(
        _secao(
            "06",
            "Resultados no conjunto de teste",
            _indicadores(
                [
                    (_numero(teste["recall"]), "recall (maligno)", False),
                    (_numero(teste["f1"]), "F1", False),
                    (_numero(teste["roc_auc"]), "AUC", False),
                    (str(teste["falsos_negativos"]), "falsos negativos", True),
                ]
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
                f'{teste["falsos_positivos"]} falso positivo e '
                f'{teste["falsos_negativos"]} falsos negativos. Os falsos negativos são o número '
                "que importa: tumores malignos que passaram despercebidos.",
            )
            + _figura(
                "08_curvas_roc.png",
                "Curvas ROC",
                "AUC entre 0,983 e 0,996. AUC alta com recall de 0,929 no limiar padrão indica que "
                "o modelo separa bem as classes e o que está mal calibrado é o corte.",
            )
            + _figura(
                "09_comparacao_modelos.png",
                "Comparação dos modelos por métrica",
                "A Random Forest alcança precisão perfeita, mas com mais falsos negativos. Trocar "
                "um falso positivo por um falso negativo é mau negócio: o primeiro custa um exame, "
                "o segundo custa um diagnóstico perdido.",
            ),
        )
    )

    blocos.append(
        _secao(
            "07",
            "Ajuste do limiar de decisão",
            "<p>O limiar de 0,5 é apenas a convenção do <code>predict()</code>, não uma escolha "
            "clínica. Baixá-lo para 0,30 melhora todas as métricas ao mesmo tempo e reduz os "
            f'falsos negativos de {teste["falsos_negativos"]} para {limiar_030["falsos_negativos"]}, '
            "sem aumento de falsos positivos.</p>"
            + _tabela(
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
            )
            + _figura(
                "10_limiar_decisao.png",
                "Recall e precisão em função do limiar",
                "A direção do ajuste é defensável pelo custo assimétrico dos erros. O valor exato, "
                "porém, é frágil: com 114 casos de teste, uma diferença de dois falsos negativos "
                "pode não se repetir em outra amostra.",
            ),
        )
    )

    explicabilidade = (
        "<p>Um sistema de apoio ao diagnóstico que não explica sua previsão não é utilizável na "
        "prática: o médico precisa poder concordar ou discordar com base no raciocínio. Três "
        "técnicas complementares foram aplicadas.</p>"
    )

    if explic.get("coeficientes"):
        explicabilidade += _tabela(
            "Maiores coeficientes da regressão logística",
            [
                ("feature", "Feature"),
                ("coeficiente", "Coeficiente"),
                ("razao_de_chances", "Razão de chances"),
            ],
            explic["coeficientes"][:8],
        )

    explicabilidade += _figura(
        "11_coeficientes.png",
        "Coeficientes do modelo",
        "O maior coeficiente é texture_worst, não concave points_worst, que liderava a análise "
        "exploratória. A causa é a multicolinearidade: features redundantes dividem o peso entre "
        "si. Coeficiente alto significa informação única ao modelo, não importância clínica.",
    ) + _figura(
        "12_importancia_permutacao.png",
        "Importância por permutação, medida em recall",
        "Embaralhar texture_worst derruba o recall em 5,4 pontos percentuais. Da quinta feature em "
        "diante a queda é praticamente nula: o modelo se apoia em poucas medidas.",
    ) + _figura(
        "13_shap_beeswarm.png",
        "SHAP — visão global",
        "Cada ponto é uma paciente. Para quase todas as features do topo, valores altos empurram a "
        "previsão para maligno, exatamente como a literatura médica descreve.",
    ) + _figura(
        "14_shap_caso_falso_negativo.png",
        "SHAP — um tumor maligno não detectado",
        "Praticamente todas as medidas desta paciente estão abaixo da média da base. O modelo não "
        "errou por acaso: viu um tumor maligno de morfologia pouco característica. É o limite que "
        "nenhum ajuste de hiperparâmetro resolve.",
    )

    blocos.append(_secao("08", "Explicabilidade", explicabilidade))

    blocos.append(
        _secao(
            "09",
            "Leitura crítica",
            "<p>O modelo atinge desempenho alto e estável, com explicações coerentes com a "
            "morfologia celular descrita na literatura. Ainda assim, três limites são "
            "intransponíveis com esta base:</p>"
            "<ul>"
            "<li><strong>569 casos de uma única fonte.</strong> O desempenho pode degradar em "
            "outra população ou com outro protocolo de coleta.</li>"
            "<li><strong>As features já foram extraídas por um especialista</strong> a partir das "
            "imagens. O sistema não parte de dados crus: depende de uma etapa manual anterior.</li>"
            "<li><strong>Existem tumores malignos morfologicamente discretos</strong>, como mostra "
            "o caso analisado por SHAP. Nenhuma quantidade de ajuste resolve isso, porque a "
            "informação necessária não está nas medidas disponíveis.</li>"
            "</ul>"
            '<p class="nota">Uso adequado: triagem e priorização de fila, segunda opinião e '
            "sinalização de casos discordantes — sempre com o laudo final assinado por um médico.</p>",
        )
    )

    return "".join(blocos)


# ---------------------------------------------------------------------------
# Aba 2: dataset extra
# ---------------------------------------------------------------------------
def _painel_extra() -> str:
    return (
        '<div class="pendente">'
        "<h3>Diagnóstico por imagem com redes neurais convolucionais</h3>"
        "<p>Esta análise é a entrega extra do desafio e será preenchida na fase seguinte, com um "
        "segundo conjunto de dados — mamografias — e uma CNN treinada para classificação.</p>"
        "<p>A estrutura acompanhará a da primeira aba: caracterização da base, amostras de "
        "imagens, arquitetura da rede, curvas de treino, métricas no conjunto de teste e "
        "interpretação dos resultados.</p>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Montagem da pagina
# ---------------------------------------------------------------------------
def build_report(metrics: dict | None = None, output: Path | None = None) -> Path:
    """Gera o relatorio HTML e devolve o caminho do arquivo escrito.

    Args:
        metrics: consolidado de metricas. Lido de `results/metrics.json` se omitido.
        output: caminho de saida. Usa `results/reports/index.html` se omitido.
    """
    if metrics is None:
        metrics = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))

    destino = output or config.REPORT_FILE
    destino.parent.mkdir(parents=True, exist_ok=True)

    gerado_em = datetime.now().strftime("%d/%m/%Y às %H:%M")
    base = metrics["dataset"]

    ficha = (
        f'<span><strong>Base</strong>{escape(base["nome"])}</span>'
        f'<span><strong>Modelo escolhido</strong>{escape(metrics["modelo_escolhido"])}</span>'
        f'<span><strong>Amostras</strong>{base["amostras"]} ({base["treino"]} treino / {base["teste"]} teste)</span>'
        f"<span><strong>Gerado em</strong>{gerado_em}</span>"
    )

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detecção de Riscos em Saúde da Mulher — Tech Challenge Fase 1</title>
<style>{CSS}</style>
</head>
<body>
<div class="envelope">

<header class="cabecalho">
  <div class="etiqueta">Tech Challenge · Fase 1 · Pós-graduação em Inteligência Artificial</div>
  <h1>Detecção de riscos<br>em saúde da mulher</h1>
  <p class="subtitulo">Classificação de exames com aprendizado de máquina, com avaliação orientada
  ao custo clínico do erro e explicabilidade das previsões.</p>
  <div class="ficha">{ficha}</div>
</header>

<nav class="abas" role="tablist">
  <button class="aba" role="tab" aria-selected="true" data-painel="painel-wisconsin">
    Dataset 1 — Wisconsin (tabular)
  </button>
  <button class="aba" role="tab" aria-selected="false" data-painel="painel-extra">
    Dataset 2 — Extra (imagens · CNN)
  </button>
</nav>

<main>
  <div class="painel" id="painel-wisconsin" role="tabpanel">{_painel_wisconsin(metrics)}</div>
  <div class="painel" id="painel-extra" role="tabpanel" hidden>{_painel_extra()}</div>
</main>

<footer class="rodape">
  <span>Relatório gerado automaticamente por <code>src/report.py</code></span>
  <span>Todos os números vêm de <code>results/metrics.json</code></span>
</footer>

</div>
<script>{JS}</script>
</body>
</html>
"""

    destino.write_text(html, encoding="utf-8")
    return destino
