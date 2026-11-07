// Logica pura do hospital: planta, rotas, colisao e jornada do caso.
//
// Nenhuma linha aqui toca o DOM ou o Three.js. Fica separada para ser testada no
// Node (tests/js/), como a logica da vitrine.
const HospitalLogica = (function () {
  "use strict";

  const ALTURA_PAREDE = 3.4;
  const LARGURA_PORTA = 2.4;
  const CORREDOR = { x0: -10, x1: 23, z0: -2, z1: 2 };

  // Planta em metros. Salas ao norte (z negativo) e ao sul do corredor, na ordem
  // da historia: a visita guiada anda sempre para leste.
  // `posto` e onde o robo fala; `parada`, onde o portador do caso espera; `camera`, de
  // onde a visita olha, sempre com o robo a direita, longe do painel de dialogo.
  const SALAS = {
    recepcao: {
      x0: -22, x1: -10, z0: -8, z1: 8, porta: { x: -10, z: 0 }, ala: "acolhida",
      posto: { x: -16, z: -6.2 }, parada: { x: -16, z: -3.4 },
      camera: { de: [-19.2, 1.9, -0.8], para: [-17.3, 1.25, -6.4] },
    },
    dados: {
      x0: -10, x1: 1, z0: -11, z1: -2, porta: { x: -4.5, z: -2 }, ala: "fase1",
      posto: { x: -3.2, z: -5.4 }, parada: { x: -5.6, z: -4.2 },
      camera: { de: [-7.8, 2.1, -2.8], para: [-4.3, 1.15, -7.6] },
    },
    modelos: {
      x0: -10, x1: 1, z0: 2, z1: 11, porta: { x: -4.5, z: 2 }, ala: "fase1",
      posto: { x: -4, z: 6.8 }, parada: { x: -2.8, z: 4.4 },
      camera: { de: [-1.8, 2.2, 2.8], para: [-5.2, 1.25, 8.6] },
    },
    imagem: {
      x0: 1, x1: 12, z0: -11, z1: -2, porta: { x: 6.5, z: -2 }, ala: "fase1",
      posto: { x: 8.6, z: -5.6 }, parada: { x: 5, z: -4.2 },
      camera: { de: [3.2, 2.2, -2.8], para: [7.2, 1.3, -8] },
    },
    evolucao: {
      x0: 1, x1: 12, z0: 2, z1: 11, porta: { x: 6.5, z: 2 }, ala: "fase2",
      posto: { x: 4.6, z: 5.2 }, parada: { x: 4.4, z: 3.4 },
      camera: { de: [10.6, 2.3, 2.8], para: [6.2, 1.25, 7.2] },
    },
    laudos: {
      x0: 12, x1: 23, z0: -11, z1: -2, porta: { x: 17.5, z: -2 }, ala: "fase2",
      posto: { x: 19.2, z: -5.4 }, parada: { x: 16, z: -4.2 },
      camera: { de: [14.2, 2.1, -2.8], para: [17.8, 1.2, -8.4] },
    },
    central: {
      x0: 12, x1: 23, z0: 2, z1: 11, porta: { x: 17.5, z: 2 }, ala: "fase2",
      posto: { x: 19.2, z: 5.6 }, parada: { x: 16, z: 4.2 },
      camera: { de: [13.4, 2.3, 2.8], para: [17.6, 1.5, 8.6] },
    },
    obras: {
      x0: 23, x1: 35, z0: -8, z1: 8, porta: { x: 23, z: 0 }, ala: "fase3",
      posto: { x: 25.4, z: 1.8 }, parada: { x: 25.6, z: -1.4 },
      camera: { de: [24.2, 2.3, -3.4], para: [28.6, 1, 3.2] },
    },
  };

  const CORES_DA_ALA = { acolhida: "#e0528c", fase1: "#1f9e8f", fase2: "#7a4fd0", fase3: "#e0a91f" };

  function salaEm(x, z) {
    for (const id of Object.keys(SALAS)) {
      const s = SALAS[id];
      if (x >= s.x0 && x <= s.x1 && z >= s.z0 && z <= s.z1) return id;
    }
    return null;
  }

  // Ponto do corredor na frente da porta. Recepcao e obras abrem no eixo do corredor.
  function eixoDaPorta(id) {
    return { x: SALAS[id].porta.x, z: 0 };
  }

  // Um passo para dentro da sala, para o robo nao cortar o batente na diagonal.
  function soleira(id) {
    const s = SALAS[id];
    const p = s.porta;
    if (p.x === s.x0) return { x: p.x + 1, z: p.z };
    if (p.x === s.x1) return { x: p.x - 1, z: p.z };
    if (p.z === s.z0) return { x: p.x, z: p.z + 1 };
    return { x: p.x, z: p.z - 1 };
  }

  // Caminho de uma sala a outra pelo corredor. O predio e uma espinha com salas nas
  // costelas, entao nao precisa de busca: sai pela porta, anda no eixo, entra pela porta.
  function rota(de, para, destino) {
    const final = destino || SALAS[para].posto;
    if (de === para) return [final];
    return [soleira(de), SALAS[de].porta, eixoDaPorta(de), eixoDaPorta(para), SALAS[para].porta, soleira(para), final];
  }

  // Empurra um circulo para fora de retangulos. Duas passadas resolvem os cantos,
  // onde a saida de uma parede cai dentro da vizinha.
  function colidir(ponto, raio, colisores) {
    let x = ponto.x;
    let z = ponto.z;
    for (let passada = 0; passada < 2; passada++) {
      for (const c of colisores) {
        const px = Math.max(c.x0, Math.min(x, c.x1));
        const pz = Math.max(c.z0, Math.min(z, c.z1));
        const dx = x - px;
        const dz = z - pz;
        const d2 = dx * dx + dz * dz;
        if (d2 >= raio * raio) continue;
        if (d2 > 1e-9) {
          const d = Math.sqrt(d2);
          x = px + (dx / d) * raio;
          z = pz + (dz / d) * raio;
        } else {
          // Centro dentro do retangulo: sai pelo lado mais proximo.
          const saidas = [
            [x - c.x0, c.x0 - raio, z],
            [c.x1 - x, c.x1 + raio, z],
            [z - c.z0, x, c.z0 - raio],
            [c.z1 - z, x, c.z1 + raio],
          ].sort(function (a, b) { return a[0] - b[0]; });
          x = saidas[0][1];
          z = saidas[0][2];
        }
      }
    }
    return { x: x, z: z };
  }

  function numero(valor, casas) {
    return Number(valor).toFixed(casas).replace(".", ",");
  }

  // Confronta a saida do modelo com o diagnostico registrado.
  function tipoDoCaso(caso) {
    const maligno = caso.registrado === "Maligno";
    const acertou = caso.classe === caso.registrado;
    if (acertou) return maligno ? "verdadeiro positivo" : "verdadeiro negativo";
    return maligno ? "falso negativo" : "falso positivo";
  }

  function casosPorTipo(casos) {
    const grupos = {};
    casos.forEach(function (caso) {
      const tipo = tipoDoCaso(caso);
      (grupos[tipo] = grupos[tipo] || []).push(caso);
    });
    return grupos;
  }

  // Primeiro paragrafo depois do titulo: o texto inteiro nao cabe na fala do robo.
  function resumoDoTexto(texto) {
    if (!texto) return null;
    const blocos = texto.split(/\n+/).map(function (l) { return l.trim(); });
    return blocos.find(function (l) { return l && !l.startsWith("#"); }) || null;
  }

  // Paradas do caso pelo hospital. A ultima e a mesa do medico: o sistema termina
  // numa pessoa, e o erro do modelo aparece sem disfarce.
  function etapasDoCaso(caso, modelo) {
    const tipo = tipoDoCaso(caso);
    const contas = modelo.features.map(function (f) {
      return { rotulo: f, valor: numero(caso.medidas[f], 3) };
    });
    const pesos = caso.contribuicoes.map(function (c) {
      return { rotulo: c.feature, valor: (c.contribuicao >= 0 ? "+" : "") + numero(c.contribuicao, 2) };
    });
    const resumo = resumoDoTexto(caso.texto);
    const erro = tipo.startsWith("falso");
    return [
      {
        sala: "recepcao",
        titulo: "Chegada",
        fala: "Chegou o caso " + caso.indice + " do conjunto de teste. O modelo servido só precisa de " +
          modelo.features.length + " medidas, e elas seguem para a central.",
        evidencias: contas,
        fonte: "POST /explain, corpo da requisição",
      },
      {
        sala: "central",
        titulo: "Previsão",
        fala: "A API calculou probabilidade de malignidade de " + numero(caso.probabilidade, 4) +
          ". O limiar é " + numero(caso.limiar, 4) + ", então a saída é " + caso.classe.toLowerCase() + ".",
        evidencias: [
          { rotulo: "Probabilidade", valor: numero(caso.probabilidade, 4) },
          { rotulo: "Limiar", valor: numero(caso.limiar, 4) },
          { rotulo: "Saída do modelo", valor: caso.classe },
        ],
        fonte: "resposta de POST /explain",
      },
      {
        sala: "modelos",
        titulo: "Explicação",
        fala: "Estas são as medidas que mais pesaram neste caso. Valor positivo empurra para maligno; negativo, para benigno.",
        evidencias: pesos,
        fonte: "resposta de POST /explain",
      },
      {
        sala: "laudos",
        titulo: "Laudo",
        fala: resumo || "Este caso não tem texto no cache: " + (caso.texto_motivo || "motivo não informado") + ".",
        evidencias: [],
        fonte: "texto verificado, results/llm/cache",
      },
      {
        sala: "recepcao",
        titulo: "Revisão médica",
        fala: (erro
          ? "Aqui o modelo errou: foi um " + tipo + ". "
          : "O modelo acertou este caso: " + tipo + ". ") +
          "Saída do modelo: " + caso.classe.toLowerCase() + ". Diagnóstico registrado: " +
          caso.registrado.toLowerCase() + ". Por isso a decisão final é sempre do médico.",
        evidencias: [
          { rotulo: "Saída do modelo", valor: caso.classe },
          { rotulo: "Diagnóstico registrado", valor: caso.registrado },
        ],
        fonte: "conjunto de teste",
        erro: erro,
      },
    ];
  }

  // Erros do modelo servido se o corte fosse outro. E a conta da regua do limiar.
  function contarErros(casos, limiar) {
    let negativos = 0;
    let positivos = 0;
    casos.forEach(function (c) {
      const alerta = c.probabilidade >= limiar;
      if (c.registrado === "Maligno" && !alerta) negativos += 1;
      if (c.registrado === "Benigno" && alerta) positivos += 1;
    });
    return { falsosNegativos: negativos, falsosPositivos: positivos };
  }

  // Luz de cada hora do dia. A manha, a tarde e a noite marcam a passagem das fases.
  const HORAS = [
    { h: 5.5, topo: "#1f2f55", horizonte: "#eba97f", sol: "#ffb27a", solI: 0.6, ceu: "#a4b4d4", chao: "#b9a191", hemi: 1.0, luzes: 0.7 },
    { h: 8, topo: "#74b1e6", horizonte: "#f6e4cf", sol: "#ffe6c0", solI: 1.7, ceu: "#fff4e6", chao: "#e9dfcf", hemi: 1.15, luzes: 0.3 },
    { h: 12, topo: "#5ea6e6", horizonte: "#dcecf3", sol: "#fff4e2", solI: 1.9, ceu: "#fffaf2", chao: "#e9e4da", hemi: 1.25, luzes: 0.25 },
    { h: 17, topo: "#6488cc", horizonte: "#f8c897", sol: "#ffbe80", solI: 1.9, ceu: "#ffe6cc", chao: "#dcc7b3", hemi: 1.0, luzes: 0.5 },
    { h: 19, topo: "#28316a", horizonte: "#ea8468", sol: "#ff8f5c", solI: 0.8, ceu: "#a9a2cc", chao: "#9a8890", hemi: 0.6, luzes: 1 },
    { h: 22, topo: "#091026", horizonte: "#1b2848", sol: "#8fa8d8", solI: 0.28, ceu: "#5a6a96", chao: "#3a3f52", hemi: 0.34, luzes: 1.25 },
  ];

  function misturarCor(a, b, p) {
    const canal = function (cor, i) { return parseInt(cor.slice(1 + i * 2, 3 + i * 2), 16); };
    let saida = "#";
    for (let i = 0; i < 3; i++) {
      saida += Math.round(canal(a, i) + (canal(b, i) - canal(a, i)) * p).toString(16).padStart(2, "0");
    }
    return saida;
  }

  function ceuDaHora(h) {
    const limitada = Math.min(HORAS[HORAS.length - 1].h, Math.max(HORAS[0].h, h));
    let i = 0;
    while (i < HORAS.length - 2 && HORAS[i + 1].h < limitada) i++;
    const a = HORAS[i];
    const b = HORAS[i + 1];
    const p = (limitada - a.h) / (b.h - a.h);
    const saida = {};
    Object.keys(a).forEach(function (chave) {
      saida[chave] = typeof a[chave] === "string" ? misturarCor(a[chave], b[chave], p) : a[chave] + (b[chave] - a[chave]) * p;
    });
    return saida;
  }

  // Tempo de leitura de uma fala, para a visita automatica nao atropelar o publico.
  function tempoDeLeitura(texto) {
    const palavras = texto.trim().split(/\s+/).length;
    return Math.max(4, palavras / 2.6 + 1.5);
  }

  const api = {
    ALTURA_PAREDE, LARGURA_PORTA, CORREDOR, SALAS, CORES_DA_ALA,
    salaEm, rota, colidir, numero, tipoDoCaso, casosPorTipo, resumoDoTexto, etapasDoCaso, tempoDeLeitura,
    contarErros, ceuDaHora, misturarCor,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  return api;
})();
