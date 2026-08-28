"""Folha de estilo e script do relatorio HTML.

Separado de `src/report.py` para que o modulo de montagem trate de conteudo e
este trate de aparencia. Nada aqui depende de rede: fontes sao pilhas locais,
porque a pagina precisa abrir por duplo clique, sem servidor e sem internet.

Direcao visual: editorial clinico. Tipografia serifada de periodico medico,
papel quente, filetes finos, numeracao de secao na margem. A paleta segue o
perfil academico (marinho de referencia, ambar de citacao) sobre papel claro.
"""

CSS = """
:root {
  color-scheme: light dark;

  --papel: #f6f3ec;
  --papel-cartao: #fffdf8;
  --papel-fundo: #efebe1;
  --tinta: #14161c;
  --tinta-media: #4a5160;
  --tinta-fraca: #7c8494;
  --linha: #ded8cb;
  --linha-forte: #bab3a3;
  --marinho: #1e3a5f;
  --ambar: #b45309;
  --destaque: #efe9dc;
  --sombra: 0 1px 2px rgba(20, 22, 28, 0.04);

  --serifada: "Iowan Old Style", "Palatino Linotype", Palatino, Cambria, Georgia, serif;
  --sem-serifa: Corbel, "Segoe UI", Optima, "Avenir Next", "Helvetica Neue", sans-serif;
  --mono: Consolas, "SF Mono", "Roboto Mono", ui-monospace, monospace;

  /* A largura do texto e a unidade de medida da pagina: a coluna de conteudo
     tem exatamente a mesma largura da linha de texto confortavel, entao
     paragrafos, figuras, tabelas e legendas dividem a mesma margem esquerda e
     a mesma margem direita. Nada fica em bloco estreito sob uma imagem larga. */
  --medida: 74ch;
  --largura: 1160px;
  --coluna-indice: 232px;
}

/* Paleta escura, definida uma vez e aplicada em dois casos: quando o sistema
   pede modo escuro e o leitor nao escolheu nada, ou quando ele escolheu escuro
   explicitamente. A escolha do leitor sempre vence a do sistema. */
@media (prefers-color-scheme: dark) {
  :root:not([data-tema]) {
    --papel: #12141a;
    --papel-cartao: #191c23;
    --papel-fundo: #0d0f14;
    --tinta: #eef1f6;
    --tinta-media: #aab3c2;
    --tinta-fraca: #79818f;
    --linha: #2a2e38;
    --linha-forte: #3d434f;
    --marinho: #7ea9dd;
    --ambar: #e79a4a;
    --destaque: #1e2129;
    --sombra: none;
  }
}

:root[data-tema="escuro"] {
  --papel: #12141a;
  --papel-cartao: #191c23;
  --papel-fundo: #0d0f14;
  --tinta: #eef1f6;
  --tinta-media: #aab3c2;
  --tinta-fraca: #79818f;
  --linha: #2a2e38;
  --linha-forte: #3d434f;
  --marinho: #7ea9dd;
  --ambar: #e79a4a;
  --destaque: #1e2129;
  --sombra: none;
}

* { box-sizing: border-box; }

html { scroll-behavior: smooth; scroll-padding-top: 96px; }

@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}

body {
  margin: 0;
  background: var(--papel);
  color: var(--tinta);
  font-family: var(--serifada);
  /* Corpo grande de proposito: a pagina e usada como apoio visual em video,
     onde o texto e lido em tela compartilhada e nao a 40 cm de distancia. */
  font-size: 20px;
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}

a { color: inherit; }

:focus-visible {
  outline: 2px solid var(--ambar);
  outline-offset: 3px;
  border-radius: 2px;
}

.pular {
  position: absolute;
  left: -9999px;
  top: 0;
  background: var(--papel-cartao);
  color: var(--tinta);
  padding: 12px 18px;
  border: 1px solid var(--linha-forte);
  font-family: var(--sem-serifa);
  font-size: 14px;
  z-index: 100;
}
.pular:focus { left: 16px; top: 16px; }

/* `relative` para ancorar o alternador de tema no canto da pagina. */
.envelope { position: relative; max-width: var(--largura); margin: 0 auto; padding: 0 40px 96px; }

/* ---------------------------------------------------------------- capa --- */
.capa { padding: 84px 0 40px; }

.etiqueta {
  font-family: var(--sem-serifa);
  font-size: 11px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  margin-bottom: 22px;
}

h1 {
  font-size: clamp(38px, 6vw, 66px);
  line-height: 1.03;
  margin: 0 0 20px;
  font-weight: 400;
  letter-spacing: -0.02em;
  max-width: 18ch;
}

.chamada {
  font-size: 24px;
  line-height: 1.45;
  color: var(--tinta-media);
  max-width: 60ch;
  margin: 0 0 36px;
}

/* Grade, nao flex: em linha unica os campos ficam em colunas alinhadas, e
   quando a tela encolhe eles quebram em fileiras regulares em vez de deixar
   um campo solto na segunda linha. */
.ficha {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
  gap: 20px 28px;
  font-family: var(--sem-serifa);
  font-size: 14px;
  color: var(--tinta-media);
  padding-top: 26px;
  border-top: 1px solid var(--linha);
}
.ficha strong {
  display: block;
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  font-weight: 600;
  margin-bottom: 3px;
}

/* ------------------------------------------------------ resumo executivo --- */
.resumo {
  margin: 56px 0 48px;
  padding: 40px 44px;
  background: var(--papel-cartao);
  border: 1px solid var(--linha);
  box-shadow: var(--sombra);
}

.resumo h2 {
  font-family: var(--sem-serifa);
  font-size: 12px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  margin: 0 0 20px;
  font-weight: 600;
}

.resumo p { font-size: 21px; max-width: 72ch; }
.resumo p:last-of-type { margin-bottom: 0; }

.veredito {
  border-left: 3px solid var(--ambar);
  padding-left: 22px;
  margin: 28px 0 0;
  font-size: 19px;
  color: var(--tinta-media);
}

/* ---------------------------------------------------------------- abas --- */
.barra-abas {
  position: sticky;
  top: 0;
  z-index: 20;
  background: var(--papel);
  border-bottom: 1px solid var(--linha);
  display: flex;
  align-items: center;
  gap: 16px;
}

.abas { display: flex; gap: 2px; overflow-x: auto; scrollbar-width: none; min-width: 0; }
.abas::-webkit-scrollbar { display: none; }

.aba {
  appearance: none;
  border: none;
  background: none;
  cursor: pointer;
  white-space: nowrap;
  font-family: var(--sem-serifa);
  font-size: 14.5px;
  letter-spacing: 0.01em;
  color: var(--tinta-fraca);
  padding: 19px 22px 16px;
  border-bottom: 3px solid transparent;
  transition: color 0.18s ease, border-color 0.18s ease;
}
.aba:hover { color: var(--tinta-media); }
.aba[aria-selected="true"] { color: var(--tinta); border-bottom-color: var(--ambar); }
.aba .numero {
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--ambar);
  margin-right: 8px;
}

/* Alternador de tema no topo direito da pagina — posicionado no envelope, nao
   na janela: ele fica onde a pagina comeca e sai de cena com a rolagem, em vez
   de acompanhar a tela e cobrir conteudo. */
.tema {
  position: absolute;
  top: 22px;
  right: 40px;
  z-index: 60;
  display: inline-flex;
  align-items: center;
  gap: 9px;
  appearance: none;
  cursor: pointer;
  background: var(--papel-cartao);
  border: 1px solid var(--linha-forte);
  color: var(--tinta-media);
  font-family: var(--sem-serifa);
  font-size: 12px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  padding: 10px 15px;
  border-radius: 2px;
  box-shadow: var(--sombra);
  transition: color 0.18s ease, border-color 0.18s ease;
}
.tema:hover { color: var(--tinta); border-color: var(--tinta-fraca); }
.tema svg { width: 16px; height: 16px; flex: none; }
.tema [data-icone] { display: none; }
.tema[data-estado="claro"] [data-icone="claro"],
.tema[data-estado="escuro"] [data-icone="escuro"] { display: block; }

@media (max-width: 860px) {
  .tema .rotulo-tema { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
  .tema { padding: 10px 11px; top: 16px; right: 20px; }
}

.painel[hidden] { display: none; }

/* --------------------------------------------------------------- corpo --- */
.corpo {
  display: grid;
  grid-template-columns: var(--coluna-indice) minmax(0, 1fr);
  gap: 56px;
  align-items: start;
}

/* --------------------------------------------------------------- indice --- */
.indice { position: sticky; top: 78px; padding-top: 52px; }

.indice h2 {
  font-family: var(--sem-serifa);
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  margin: 0 0 14px;
  font-weight: 600;
}

.indice ol { list-style: none; margin: 0; padding: 0; }

.indice a {
  display: flex;
  gap: 10px;
  align-items: baseline;
  text-decoration: none;
  font-family: var(--sem-serifa);
  font-size: 14px;
  line-height: 1.35;
  color: var(--tinta-fraca);
  padding: 9px 0 9px 14px;
  border-left: 2px solid var(--linha);
  transition: color 0.18s ease, border-color 0.18s ease;
}
.indice a:hover { color: var(--tinta-media); }
.indice a[aria-current="true"] { color: var(--tinta); border-left-color: var(--ambar); }
.indice .numero { font-family: var(--mono); font-size: 10px; color: var(--tinta-fraca); }

/* -------------------------------------------------------------- secoes --- */
/* Toda a coluna de leitura cabe na medida do texto. Como figuras, tabelas,
   indicadores e legendas sao filhos dela, todos terminam na mesma margem
   direita — o alinhamento vem da estrutura, nao de ajuste caso a caso. */
.conteudo { min-width: 0; max-width: var(--medida); padding-top: 8px; }

section { padding: 52px 0 4px; border-bottom: 1px solid var(--linha); scroll-margin-top: 96px; }
section:last-child { border-bottom: none; }

section > h2 {
  font-size: 16px;
  font-family: var(--sem-serifa);
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  margin: 0 0 28px;
  display: flex;
  align-items: baseline;
  gap: 14px;
}
section > h2::before {
  content: attr(data-numero);
  font-family: var(--mono);
  font-size: 12px;
  color: var(--ambar);
  letter-spacing: 0;
}

h3 {
  font-size: 26px;
  font-weight: 400;
  margin: 42px 0 14px;
  letter-spacing: -0.012em;
}

p { max-width: none; margin: 0 0 20px; }

p.nota {
  font-size: 18px;
  color: var(--tinta-media);
  border-left: 2px solid var(--linha-forte);
  padding-left: 20px;
}

strong { font-weight: 600; }

code {
  font-family: var(--mono);
  font-size: 0.85em;
  background: var(--destaque);
  padding: 1px 6px;
  border-radius: 3px;
}

ul { max-width: none; padding-left: 24px; margin: 0 0 22px; }
li { margin-bottom: 10px; }
li::marker { color: var(--tinta-fraca); }

/* --------------------------------------------------------- indicadores --- */
.indicadores {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 1px;
  background: var(--linha);
  border: 1px solid var(--linha);
  margin: 4px 0 40px;
}

.indicador {
  background: var(--papel-cartao);
  padding: 24px 20px 20px;
  display: flex;
  flex-direction: column;
}

/* O numero nunca quebra em duas linhas: um "357 / 212" partido ao meio
   desalinha o rotulo do cartao vizinho. O rotulo fica colado na base para
   que todos os cartoes leiam na mesma altura. */
.indicador .valor {
  font-size: 34px;
  line-height: 1.05;
  letter-spacing: -0.03em;
  display: block;
  margin-bottom: 14px;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

.indicador .rotulo:last-child { margin-top: auto; }

.indicador .rotulo {
  font-family: var(--sem-serifa);
  font-size: 11.5px;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  display: block;
}

.indicador.alerta .valor { color: var(--ambar); }
.indicador.marcado .valor { color: var(--marinho); }

/* -------------------------------------------------------------- fluxo --- */
/* Tres colunas, nao seis: na medida do texto, seis cartoes deixavam cada
   etapa com uma palavra por linha. Duas fileiras de tres respiram melhor e
   sao legiveis a distancia, que e como a pagina sera lida no video. */
.fluxo {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 8px 0 40px;
}

@media (max-width: 620px) { .fluxo { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 420px) { .fluxo { grid-template-columns: minmax(0, 1fr); } }

.etapa {
  background: var(--papel-cartao);
  border: 1px solid var(--linha);
  padding: 16px 16px 14px;
  position: relative;
  box-shadow: var(--sombra);
}

.etapa .ordem {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--ambar);
  display: block;
  margin-bottom: 8px;
}

.etapa .nome {
  font-family: var(--sem-serifa);
  font-size: 15px;
  font-weight: 600;
  display: block;
  margin-bottom: 6px;
  line-height: 1.3;
}

.etapa .detalhe {
  font-family: var(--sem-serifa);
  font-size: 13px;
  color: var(--tinta-fraca);
  line-height: 1.45;
}

/* ------------------------------------------------------------- tabelas --- */
.rolagem { overflow-x: auto; margin: 0 0 30px; }

table {
  border-collapse: collapse;
  width: 100%;
  font-size: 16.5px;
  font-variant-numeric: tabular-nums;
}

caption {
  caption-side: top;
  text-align: left;
  font-family: var(--sem-serifa);
  font-size: 12.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  padding-bottom: 12px;
}

/* As celulas quebram linha. Numeros nao tem onde quebrar e continuam inteiros;
   quem quebra e o texto corrido — antes ele forcava a tabela para alem da
   coluna de leitura e a ultima coluna saia cortada na tela. */
th, td { padding: 11px 18px 11px 0; text-align: right; vertical-align: top; }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
td { text-wrap: pretty; }

thead th {
  font-family: var(--sem-serifa);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  border-bottom: 1px solid var(--tinta);
}

tbody tr { border-bottom: 1px solid var(--linha); }
tbody tr.destacada { background: var(--destaque); }

tbody tr.destacada td:first-child::after {
  content: "escolhido";
  font-family: var(--sem-serifa);
  font-size: 9.5px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ambar);
  margin-left: 10px;
}

/* ------------------------------------------------------------- figuras --- */
figure { margin: 0 0 34px; }

figure img {
  display: block;
  width: 100%;
  height: auto;
  background: #fcfcfb;
  border: 1px solid var(--linha);
  padding: 12px;
}

/* Legenda em duas colunas, ocupando toda a largura da figura: o titulo fica
   na coluna estreita da esquerda e a leitura corre pela linha inteira. Antes a
   legenda tinha medida propria e formava um bloco estreito sob a imagem. */
figcaption {
  display: grid;
  grid-template-columns: minmax(0, 13em) minmax(0, 1fr);
  gap: 6px 32px;
  align-items: baseline;
  max-width: none;
  margin-top: 14px;
  padding-top: 13px;
  border-top: 1px solid var(--linha);
  font-family: var(--sem-serifa);
  font-size: 14px;
  line-height: 1.6;
  color: var(--tinta-media);
  text-wrap: pretty;
}

figcaption .titulo-figura {
  color: var(--tinta);
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  line-height: 1.45;
}

@media (max-width: 620px) {
  figcaption { grid-template-columns: minmax(0, 1fr); }
}

details.dados { margin-top: 12px; }

details.dados > summary {
  cursor: pointer;
  font-family: var(--sem-serifa);
  font-size: 12.5px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  padding: 7px 0;
  list-style: none;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  transition: color 0.18s ease;
}
details.dados > summary::-webkit-details-marker { display: none; }
details.dados > summary:hover { color: var(--tinta-media); }
details.dados > summary::before {
  content: "+";
  font-family: var(--mono);
  font-size: 13px;
  color: var(--ambar);
}
details.dados[open] > summary::before { content: "−"; }

details.dados .rolagem {
  margin: 8px 0 0;
  padding: 18px 20px;
  background: var(--papel-cartao);
  border: 1px solid var(--linha);
}

/* --------------------------------------------------- como foi feito --- */
/* O codigo que executa a etapa, ao lado da prosa que a explica. E o bloco que
   sustenta a narracao do video: da para apontar para a linha enquanto se fala
   sobre a decisao. */
.feito {
  margin: 10px 0 38px;
  border: 1px solid var(--linha);
  background: var(--papel-cartao);
  box-shadow: var(--sombra);
}

.feito-cabeca {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 6px 20px;
  padding: 12px 20px;
  background: var(--destaque);
  border-bottom: 1px solid var(--linha);
}

.feito-etiqueta {
  font-family: var(--sem-serifa);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.17em;
  text-transform: uppercase;
  color: var(--ambar);
}

.feito-ref {
  font-family: var(--mono);
  font-size: 12.5px;
  color: var(--tinta-fraca);
  background: none;
  padding: 0;
  overflow-wrap: anywhere;
}

pre.codigo {
  margin: 0;
  padding: 20px;
  overflow-x: auto;
  font-family: var(--mono);
  font-size: 13.5px;
  line-height: 1.62;
  color: var(--tinta-media);
  tab-size: 4;
}
pre.codigo code {
  background: none;
  padding: 0;
  font-size: inherit;
  border-radius: 0;
}

/* Realce sobrio: palavra reservada, texto, numero e comentario. Quatro cores,
   nao um arco-iris — o bloco continua sendo parte da pagina. */
pre.codigo .k { color: var(--marinho); font-weight: 600; }
pre.codigo .d { color: var(--tinta); font-weight: 600; }
pre.codigo .b { color: var(--tinta); }
pre.codigo .s { color: var(--ambar); }
pre.codigo .n { color: var(--marinho); }
pre.codigo .c { color: var(--tinta-fraca); font-style: italic; }

.feito-nota {
  margin: 0;
  padding: 14px 20px 16px;
  border-top: 1px solid var(--linha);
  font-family: var(--sem-serifa);
  font-size: 14px;
  line-height: 1.6;
  color: var(--tinta-media);
  text-wrap: pretty;
}

/* ----------------------------------------------------------- decisoes --- */
.decisoes {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 14px;
  margin: 10px 0 38px;
}

/* `subgrid` alinha titulo, justificativa e contrafactual na mesma altura em
   toda a fileira, mesmo com textos de comprimentos diferentes. Sem isso, o
   filete do contrafactual cai em alturas distintas em cada cartao. */
.decisao {
  display: grid;
  grid-template-rows: subgrid;
  grid-row: span 3;
  align-content: start;
  background: var(--papel-cartao);
  border: 1px solid var(--linha);
  padding: 20px 22px 18px;
  box-shadow: var(--sombra);
}

.decisao h4 {
  margin: 0 0 14px;
  font-family: var(--sem-serifa);
  font-size: 16px;
  font-weight: 600;
  line-height: 1.35;
  letter-spacing: 0;
  color: var(--tinta);
  text-wrap: balance;
}

.decisao p {
  margin: 0;
  font-size: 16px;
  line-height: 1.55;
  color: var(--tinta-media);
  text-wrap: pretty;
}

.decisao .marca {
  display: block;
  font-family: var(--sem-serifa);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--tinta-fraca);
  margin-bottom: 4px;
}

/* O caminho que nao foi tomado leva o filete ambar: e o que justifica a
   decisao, e precisa se distinguir dela a distancia. */
.decisao .contra {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--linha);
}
.decisao .contra .marca { color: var(--ambar); }

/* -------------------------------------------------------------- aviso --- */
.pendente {
  border: 1px dashed var(--linha-forte);
  padding: 48px 44px;
  margin: 44px 0;
  background: var(--papel-cartao);
}
.pendente h3 { margin-top: 0; }

/* ------------------------------------------------------------- rodape --- */
.rodape {
  margin-top: 64px;
  padding-top: 26px;
  border-top: 2px solid var(--tinta);
  font-family: var(--sem-serifa);
  font-size: 13.5px;
  color: var(--tinta-fraca);
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 14px;
}

/* ----------------------------------------------------------- responsivo --- */
@media (max-width: 1080px) {
  .corpo { grid-template-columns: 1fr; gap: 0; }
  .indice {
    position: static;
    padding: 32px 0 0;
    border-bottom: 1px solid var(--linha);
  }
  .indice ol { display: flex; flex-wrap: wrap; gap: 4px 6px; padding-bottom: 20px; }
  .indice a { border-left: none; border-bottom: 2px solid var(--linha); padding: 6px 10px; }
  .indice a[aria-current="true"] { border-bottom-color: var(--ambar); }
}

@media (max-width: 720px) {
  .envelope { padding: 0 20px 64px; }
  .capa { padding-top: 52px; }
  .resumo { padding: 28px 24px; }
  .indicador .valor { font-size: 32px; }
  th, td { padding-right: 14px; }
}

@media print {
  .barra-abas, .indice, .pular, .tema { display: none; }
  .painel[hidden] { display: block; }
  .corpo { grid-template-columns: 1fr; }
  section { break-inside: avoid; }
  figure { break-inside: avoid; }
  details.dados[open] > summary { display: none; }
  pre.codigo { white-space: pre-wrap; font-size: 11px; }
  .feito, .decisao { break-inside: avoid; }
}
"""

JS = """
(function () {
  // Alternador de tema. Dois estados apenas: claro e escuro. Na primeira
  // visita o estado inicial vem da preferencia do sistema, mas passa a ser
  // explicito para que um clique so ja inverta o tema. A escolha fica no
  // navegador de quem le, entao sobrevive a recarga e nao afeta mais ninguem.
  var ROTULOS = { claro: 'Claro', escuro: 'Escuro' };
  var botao = document.querySelector('.tema');

  function aplicar(estado, guardar) {
    document.documentElement.setAttribute('data-tema', estado);

    if (botao) {
      botao.dataset.estado = estado;
      botao.querySelector('.rotulo-tema').textContent = ROTULOS[estado];
      botao.setAttribute('aria-pressed', String(estado === 'escuro'));
      botao.setAttribute(
        'aria-label',
        'Tema ' + ROTULOS[estado] + '. Clique para usar o tema ' +
        (estado === 'escuro' ? 'claro' : 'escuro') + '.'
      );
    }

    if (guardar) {
      try { localStorage.setItem('tema', estado); } catch (erro) { /* modo privado */ }
    }
  }

  if (botao) {
    var salvo = null;
    try { salvo = localStorage.getItem('tema'); } catch (erro) { /* modo privado */ }

    var inicial = salvo === 'claro' || salvo === 'escuro' ? salvo : null;
    if (!inicial) {
      var escuroNoSistema = window.matchMedia
        && window.matchMedia('(prefers-color-scheme: dark)').matches;
      inicial = escuroNoSistema ? 'escuro' : 'claro';
    }
    aplicar(inicial, false);

    botao.addEventListener('click', function () {
      aplicar(botao.dataset.estado === 'escuro' ? 'claro' : 'escuro', true);
    });
  }
})();

(function () {
  var abas = Array.prototype.slice.call(document.querySelectorAll('.aba'));
  var ancora = document.getElementById('ancora-abas');

  // Posicao da barra de abas no documento. A ancora e um elemento comum, e
  // nao a barra em si: a barra e `sticky`, entao o seu retangulo devolve a
  // posicao grudada no topo, nao a posicao real no fluxo da pagina.
  function topoDasAbas() {
    if (!ancora) { return 0; }
    return ancora.getBoundingClientRect().top + window.pageYOffset;
  }

  function selecionar(aba, mover) {
    // Trocar de aba recomeca a leitura: a pagina vai para o inicio da aba, que
    // e a propria barra de abas — nunca para o topo do documento, senao a capa
    // e o resumo executivo voltariam a ocupar a tela a cada troca.
    abas.forEach(function (outra) {
      var ativa = outra === aba;
      outra.setAttribute('aria-selected', String(ativa));
      outra.tabIndex = ativa ? 0 : -1;
      document.getElementById(outra.dataset.painel).hidden = !ativa;
    });
    if (mover) { aba.focus({ preventScroll: true }); }

    // Aba curta demais para sustentar essa posicao: o navegador pararia antes.
    var maximo = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
    window.scrollTo({ top: Math.min(topoDasAbas(), maximo), behavior: 'instant' });
  }

  abas.forEach(function (aba, indice) {
    aba.addEventListener('click', function () { selecionar(aba, false); });
    aba.addEventListener('keydown', function (evento) {
      var passo = evento.key === 'ArrowRight' ? 1 : evento.key === 'ArrowLeft' ? -1 : 0;
      if (!passo) { return; }
      evento.preventDefault();
      selecionar(abas[(indice + passo + abas.length) % abas.length], true);
    });
  });

  // Indice lateral: marca a secao visivel enquanto a pagina rola.
  var secoes = Array.prototype.slice.call(document.querySelectorAll('section[id]'));
  var elos = {};
  document.querySelectorAll('.indice a').forEach(function (elo) {
    elos[elo.getAttribute('href').slice(1)] = elo;
  });

  if ('IntersectionObserver' in window) {
    var visiveis = new Set();
    var observador = new IntersectionObserver(function (entradas) {
      entradas.forEach(function (entrada) {
        if (entrada.isIntersecting) { visiveis.add(entrada.target.id); }
        else { visiveis.delete(entrada.target.id); }
      });

      var atual = secoes.filter(function (s) { return visiveis.has(s.id); })[0];
      Object.keys(elos).forEach(function (id) {
        elos[id].setAttribute('aria-current', String(Boolean(atual) && atual.id === id));
      });
    }, { rootMargin: '-90px 0px -65% 0px' });

    secoes.forEach(function (secao) { observador.observe(secao); });
  }
})();
"""
