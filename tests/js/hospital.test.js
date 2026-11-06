// Testes da logica do hospital (src/ui/hospital/logica.js), no Node, sem navegador.
//
// Os que usam casos reais leem a pagina montada em build/space/hospital.html
// (scripts/montar_hospital.py); sem ela, sao pulados com o motivo.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const RAIZ = path.resolve(__dirname, "..", "..");
const L = require(path.join(RAIZ, "src", "ui", "hospital", "logica.js"));
const PAGINA = path.join(RAIZ, "build", "space", "hospital.html");

function dadosDaPagina() {
  if (!fs.existsSync(PAGINA)) return null;
  const html = fs.readFileSync(PAGINA, "utf8");
  const bloco = html.match(/<script id="dados" type="application\/json">([\s\S]*?)<\/script>/);
  return JSON.parse(bloco[1].replace(/<\\\//g, "</"));
}
const dados = dadosDaPagina();
const semPagina = dados ? false : "sem build/space/hospital.html: rode scripts/montar_hospital.py";

const dentro = (s, p) => p.x >= s.x0 && p.x <= s.x1 && p.z >= s.z0 && p.z <= s.z1;

test("postos, paradas e cameras ficam dentro da propria sala", () => {
  for (const [id, sala] of Object.entries(L.SALAS)) {
    assert.ok(dentro(sala, sala.posto), `posto de ${id}`);
    assert.ok(dentro(sala, sala.parada), `parada de ${id}`);
    assert.ok(dentro(sala, { x: sala.camera.de[0], z: sala.camera.de[2] }), `camera de ${id}`);
  }
});

test("salas nao se sobrepoem nem invadem o corredor", () => {
  const ids = Object.keys(L.SALAS);
  const cruzam = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.z0 < b.z1 && b.z0 < a.z1;
  ids.forEach((a, i) => {
    assert.ok(!cruzam(L.SALAS[a], L.CORREDOR), `${a} invade o corredor`);
    ids.slice(i + 1).forEach((b) => assert.ok(!cruzam(L.SALAS[a], L.SALAS[b]), `${a} cruza ${b}`));
  });
});

test("rota entre salas passa pelas duas portas e so anda em sala ou corredor", () => {
  const ids = Object.keys(L.SALAS);
  const noPredio = (p) => L.salaEm(p.x, p.z) || dentro(L.CORREDOR, p);
  for (const de of ids) {
    for (const para of ids) {
      if (de === para) continue;
      const caminho = L.rota(de, para);
      assert.ok(caminho.includes(L.SALAS[de].porta), `${de} para ${para} sai pela porta`);
      assert.ok(caminho.includes(L.SALAS[para].porta), `${de} para ${para} entra pela porta`);
      assert.deepEqual(caminho.at(-1), L.SALAS[para].posto);
      caminho.forEach((p) => assert.ok(noPredio(p), `ponto fora do predio em ${de} para ${para}`));
      // Trechos sempre em linha reta num eixo, exceto o ultimo, ja dentro da sala:
      // assim o robo nunca corta parede na diagonal.
      for (let i = 1; i < caminho.length - 1; i++) {
        const a = caminho[i - 1];
        const b = caminho[i];
        assert.ok(a.x === b.x || a.z === b.z, `diagonal em ${de} para ${para}`);
      }
    }
  }
});

test("colisao tira o visitante de dentro da parede, a qualquer angulo", () => {
  const parede = { x0: -1, x1: 1, z0: -0.1, z1: 0.1 };
  const raio = 0.3;
  for (let a = 0; a < Math.PI * 2; a += 0.2) {
    const saida = L.colidir({ x: Math.cos(a) * 0.5, z: Math.sin(a) * 0.15 }, raio, [parede]);
    const px = Math.max(parede.x0, Math.min(saida.x, parede.x1));
    const pz = Math.max(parede.z0, Math.min(saida.z, parede.z1));
    assert.ok(Math.hypot(saida.x - px, saida.z - pz) >= raio - 1e-9, `angulo ${a}`);
  }
});

test("colisao nao mexe em quem esta livre", () => {
  const ponto = { x: 5, z: 5 };
  assert.deepEqual(L.colidir(ponto, 0.3, [{ x0: 0, x1: 1, z0: 0, z1: 1 }]), ponto);
});

test("tipo do caso cruza saida do modelo e diagnostico registrado", () => {
  assert.equal(L.tipoDoCaso({ registrado: "Maligno", classe: "Benigno" }), "falso negativo");
  assert.equal(L.tipoDoCaso({ registrado: "Benigno", classe: "Maligno" }), "falso positivo");
  assert.equal(L.tipoDoCaso({ registrado: "Maligno", classe: "Maligno" }), "verdadeiro positivo");
  assert.equal(L.tipoDoCaso({ registrado: "Benigno", classe: "Benigno" }), "verdadeiro negativo");
});

test("resumo do texto pula titulos e aceita texto ausente", () => {
  assert.equal(L.resumoDoTexto("## Leitura\nPrimeira frase.\n\n## Outra\nSegunda."), "Primeira frase.");
  assert.equal(L.resumoDoTexto(null), null);
});

test("tempo de leitura cresce com o texto e tem piso", () => {
  assert.equal(L.tempoDeLeitura("curto"), 4);
  assert.ok(L.tempoDeLeitura("palavra ".repeat(60)) > L.tempoDeLeitura("palavra ".repeat(20)));
});

test("salas da pagina cobrem a planta, com perguntas e fonte", { skip: semPagina }, () => {
  assert.deepEqual(dados.salas.map((s) => s.id).sort(), Object.keys(L.SALAS).sort());
  dados.salas.forEach((sala) => {
    assert.ok(sala.perguntas.length >= 2, `${sala.id} sem perguntas`);
    sala.perguntas.forEach((p) => assert.ok(p.fonte, `${sala.id}: "${p.pergunta}" sem fonte`));
  });
});

test("toda jornada termina na revisao medica e declara o erro do modelo", { skip: semPagina }, () => {
  dados.casos.forEach((caso) => {
    const etapas = L.etapasDoCaso(caso, dados.modelo);
    const final = etapas.at(-1);
    assert.equal(final.titulo, "Revisão médica");
    assert.equal(final.erro, caso.classe !== caso.registrado, `caso ${caso.indice}`);
    assert.match(final.fala, /médico/);
    etapas.forEach((e) => {
      assert.ok(L.SALAS[e.sala], `sala ${e.sala}`);
      assert.ok(e.fala.length > 0);
      assert.ok(!/undefined|NaN/.test(e.fala + JSON.stringify(e.evidencias)), `caso ${caso.indice}`);
    });
  });
});

test("contagem de erros dos casos bate com a matriz de confusao do modelo servido", { skip: semPagina }, () => {
  const grupos = L.casosPorTipo(dados.casos);
  const comparacao = JSON.parse(fs.readFileSync(path.join(RAIZ, "results", "genetic", "comparacao_final.json"), "utf8"));
  const servido = comparacao.find((b) => b.braco === "AG, completo");
  assert.equal(grupos["falso negativo"].length, servido.falsos_negativos);
  assert.equal(grupos["falso positivo"].length, servido.falsos_positivos);
  assert.equal(grupos["verdadeiro positivo"].length, servido.verdadeiros_positivos);
  assert.equal(grupos["verdadeiro negativo"].length, servido.verdadeiros_negativos);
});
