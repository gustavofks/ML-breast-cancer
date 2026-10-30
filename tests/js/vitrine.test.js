// Testes da logica da vitrine (src/ui/vitrine/logica.js), no Node, sem navegador.
//
// Rodam com `node --test tests/js/`. Os que comparam com a API leem a pagina
// montada em build/space/index.html (scripts/montar_space.py, na imagem `api`);
// sem ela, sao pulados com o motivo.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const RAIZ = path.resolve(__dirname, "..", "..");
const L = require(path.join(RAIZ, "src", "ui", "vitrine", "logica.js"));
const PAGINA = path.join(RAIZ, "build", "space", "index.html");

function dadosDaPagina() {
  if (!fs.existsSync(PAGINA)) return null;
  const html = fs.readFileSync(PAGINA, "utf8");
  const bloco = html.match(/<script id="dados" type="application\/json">([\s\S]*?)<\/script>/);
  return JSON.parse(bloco[1]);
}
const dados = dadosDaPagina();
const semPagina = dados ? false : "sem build/space/index.html: rode scripts/montar_space.py";

test("simulador no navegador reproduz a API nos casos do teste", { skip: semPagina }, () => {
  const parametros = dados.simulador.parametros;
  assert.ok(parametros, "modelo servido linear deveria exportar parametros");
  dados.casos.forEach((caso, i) => {
    const obtido = L.calcular(parametros, caso.medidas);
    assert.ok(Math.abs(obtido.probabilidade - caso.probabilidade) < 1e-9, `probabilidade do caso ${i}`);
    assert.equal(obtido.classe, caso.classe, `classe do caso ${i}`);
    assert.deepEqual(obtido.contribuicoes.map((c) => c.feature), caso.contribuicoes.map((c) => c.feature));
    obtido.contribuicoes.forEach((c, j) => {
      assert.ok(Math.abs(c.contribuicao - caso.contribuicoes[j].contribuicao) < 1e-9, `contribuicao ${j} do caso ${i}`);
    });
  });
});

test("metricas gravadas viram resumo por rota e alertas", { skip: semPagina }, () => {
  const resumo = L.resumirMetricas(L.lerPrometheus(dados.console.gravacao.metricas));
  const predict = resumo.rotas.find((r) => r.rota === "/predict");
  assert.ok(predict.requisicoes >= 3);
  assert.ok(predict.erros >= 1, "o cenario de medida faltante responde 422");
  assert.ok(predict.latencia_media_ms > 0);
  assert.ok(resumo.alertas.length >= 1, "o cenario de desvio gera alerta");
});

test("console gravado tem resposta para cada cenario", { skip: semPagina }, () => {
  dados.console.cenarios.forEach((c) => assert.ok(dados.console.gravacao.respostas[c.id], c.id));
});

test("lerPrometheus le rotulos e ignora comentarios", () => {
  const texto = [
    "# HELP x ajuda",
    "# TYPE x counter",
    'api_requisicoes_total{rota="/predict",status="200"} 3',
    'api_requisicoes_total{rota="/predict",status="422"} 1',
    'api_latencia_segundos_sum{rota="/predict"} 0.02',
    'api_latencia_segundos_count{rota="/predict"} 4',
    'api_alertas_de_desvio_total{feature="area_se"} 2',
  ].join("\n");
  const resumo = L.resumirMetricas(L.lerPrometheus(texto));
  assert.deepEqual(resumo.rotas, [{ rota: "/predict", requisicoes: 4, erros: 1, latencia_media_ms: 5 }]);
  assert.deepEqual(resumo.alertas, [{ feature: "area_se", alertas: 2 }]);
});

test("modo ao vivo so quando /health confirma o modelo carregado", async () => {
  const resposta = (ok, corpo) => () => Promise.resolve({ ok, json: () => Promise.resolve(corpo) });
  assert.equal(await L.detectarModo(resposta(true, { status: "ok", modelo_carregado: true })), "ao-vivo");
  assert.equal(await L.detectarModo(resposta(true, { status: "ok" })), "congelado");
  assert.equal(await L.detectarModo(resposta(false, null)), "congelado");
  assert.equal(await L.detectarModo(() => Promise.reject(new TypeError("Failed to fetch"))), "congelado");
});

test("modo congelado quando a resposta nao e JSON (pagina 404 do Space)", async () => {
  const html = () => Promise.resolve({ ok: true, json: () => Promise.reject(new SyntaxError("Unexpected token <")) });
  assert.equal(await L.detectarModo(html), "congelado");
});

test("texto gerado vira blocos e trechos, nunca marcacao", () => {
  const blocos = L.blocosDoTexto("## Leitura\nTexto com **negrito** e <b>tag</b>.\n- item um\n\n- item dois");
  assert.deepEqual(blocos.map((b) => b.tipo), ["titulo", "paragrafo", "item", "item"]);
  assert.deepEqual(blocos[1].trechos, [
    { texto: "Texto com ", negrito: false },
    { texto: "negrito", negrito: true },
    { texto: " e <b>tag</b>.", negrito: false },
  ]);
});

test("achados apontam o servido, a referencia e se a trava disparou", () => {
  const linha = (braco, fn, fp) => ({ braco, falsos_negativos: fn, falsos_positivos: fp, verdadeiros_positivos: 42 - fn, n_amostras: 114 });
  const base = { processo: { pesos: { falso_negativo: 10 }, teste: [linha("Original", 3, 1), linha("AG, completo", 4, 2), linha("AG, só limiar", 1, 4)] } };
  const a = L.achados(base);
  assert.equal(a.trava_disparou, true);
  assert.equal(a.menor_fn, 1);
  assert.deepEqual(a.bracos_com_menor_fn, ["AG, só limiar"]);
  assert.equal(a.malignos_no_teste, 42);
  assert.ok(Math.abs(a.custo_de_um_fn - 10 / 114) < 1e-12);
});

test("casosComErro devolve so previsao diferente do registrado, com o indice original", () => {
  const casos = [
    { classe: "Benigno", diagnostico_registrado: "Benigno" },
    { classe: "Benigno", diagnostico_registrado: "Maligno" },
    { classe: "Maligno", diagnostico_registrado: "Benigno" },
  ];
  assert.deepEqual(L.casosComErro(casos).map((p) => p.indice), [1, 2]);
});
