// O predio: chao, paredes com portas e janelas, teto, placas e arredores.
//
// `construirPredio` devolve os colisores, as portas automaticas e as pecas de cada ala,
// que os filmes erguem aos poucos. Toda parede nasce de `parede`, que recorta aberturas
// e registra a colisao.

const L = HospitalLogica;
const COR_PAREDE = "#fbf8f1";
const PISO_DA_ALA = { acolhida: "#f4ebe6", fase1: "#e4efe9", fase2: "#ebe6f2", fase3: "#ded7c8" };
const ESPESSURA = 0.2;

// Um vidro so para o predio inteiro: de noite, todas as janelas acendem juntas.
const VIDRO = { material: null };

function material(cor, opcoes) {
  return new THREE.MeshStandardMaterial(Object.assign({ color: cor, roughness: 0.82, metalness: 0 }, opcoes));
}

function caixa(largura, altura, profundidade, mat, x, y, z, sombra) {
  const malha = new THREE.Mesh(new THREE.BoxGeometry(largura, altura, profundidade), mat);
  malha.position.set(x, y, z);
  malha.castShadow = sombra !== false;
  malha.receiveShadow = true;
  return malha;
}

// Parede reta ao longo de x ou z, com faixa de rodape colorida. `aberturas` sao portas
// (vao ate o chao, sem colisao) ou janelas (vao com vidro, colisao mantida).
function parede(cena, colisores, a, b, opcoes) {
  const op = opcoes || {};
  const H = L.ALTURA_PAREDE;
  const noEixoX = a.z === b.z;
  const inicio = noEixoX ? Math.min(a.x, b.x) : Math.min(a.z, b.z);
  const fim = noEixoX ? Math.max(a.x, b.x) : Math.max(a.z, b.z);
  const fixo = noEixoX ? a.z : a.x;
  const grupo = new THREE.Group();
  const matParede = material(COR_PAREDE);
  const matRodape = material(op.cor || "#c9c2b4");
  VIDRO.material = VIDRO.material || new THREE.MeshPhysicalMaterial({
    color: "#cfe9ee", transparent: true, opacity: 0.22, roughness: 0.08, metalness: 0, depthWrite: false,
    emissive: "#ffcf8f", emissiveIntensity: 0,
  });
  const matVidro = VIDRO.material;

  function peca(de, ate, base, topo, mat, sombra) {
    if (ate - de < 1e-3) return;
    const comprimento = ate - de;
    const meio = (de + ate) / 2;
    const y = (base + topo) / 2;
    const m = noEixoX
      ? caixa(comprimento, topo - base, ESPESSURA, mat, meio, y, fixo, sombra)
      : caixa(ESPESSURA, topo - base, comprimento, mat, fixo, y, meio, sombra);
    grupo.add(m);
  }

  function solido(de, ate) {
    peca(de, ate, 0, H, matParede);
    const m = noEixoX
      ? caixa(ate - de, 0.34, ESPESSURA + 0.04, matRodape, (de + ate) / 2, 0.17, fixo, false)
      : caixa(ESPESSURA + 0.04, 0.34, ate - de, matRodape, fixo, 0.17, (de + ate) / 2, false);
    grupo.add(m);
  }

  function colisao(de, ate) {
    const meia = ESPESSURA / 2;
    colisores.push(noEixoX
      ? { x0: de, x1: ate, z0: fixo - meia, z1: fixo + meia }
      : { x0: fixo - meia, x1: fixo + meia, z0: de, z1: ate });
  }

  const vaos = (op.aberturas || []).slice().sort(function (p, q) { return p.centro - q.centro; });
  let cursor = inicio;
  let inicioColisao = inicio;
  vaos.forEach(function (vao) {
    const de = vao.centro - vao.largura / 2;
    const ate = vao.centro + vao.largura / 2;
    solido(cursor, de);
    if (vao.tipo === "janela") {
      peca(de, ate, 0, 1, matParede);
      peca(de, ate, 2.5, H, matParede);
      peca(de, ate, 1, 2.5, matVidro, false);
      peca(de, ate, 0.98, 1.04, material("#8e9a9c"), false);
    } else {
      peca(de, ate, 2.5, H, matParede);
      batente(grupo, noEixoX, fixo, vao.centro, vao.largura, op.cor);
      if (vao.automatica) portaAutomatica(grupo, op.portas, noEixoX, fixo, vao.centro, vao.largura);
      colisao(inicioColisao, de);
      inicioColisao = ate;
    }
    cursor = ate;
  });
  solido(cursor, fim);
  colisao(inicioColisao, fim);
  cena.add(grupo);
  return grupo;
}

// Batentes coloridos: a porta se le como porta, e a cor diz de que ala ela e.
function batente(grupo, noEixoX, fixo, centro, largura, cor) {
  const mat = material(cor || "#a9b5b5", { roughness: 0.5 });
  [centro - largura / 2, centro + largura / 2].forEach(function (p) {
    grupo.add(noEixoX
      ? caixa(0.1, 2.5, ESPESSURA + 0.08, mat, p, 1.25, fixo, false)
      : caixa(ESPESSURA + 0.08, 2.5, 0.1, mat, fixo, 1.25, p, false));
  });
  grupo.add(noEixoX
    ? caixa(largura + 0.1, 0.1, ESPESSURA + 0.08, mat, centro, 2.5, fixo, false)
    : caixa(ESPESSURA + 0.08, 0.1, largura + 0.1, mat, fixo, 2.5, centro, false));
}

// Duas folhas de vidro que abrem quando alguem chega perto, robo ou visitante.
function portaAutomatica(grupo, portas, noEixoX, fixo, centro, largura) {
  const matFolha = new THREE.MeshPhysicalMaterial({
    color: "#d8eef2", transparent: true, opacity: 0.32, roughness: 0.1, depthWrite: false,
  });
  const matAro = material("#8e9a9c", { metalness: 0.5, roughness: 0.4 });
  const meia = largura / 2;
  const folhas = [-1, 1].map(function (lado) {
    const folha = new THREE.Group();
    folha.add(new THREE.Mesh(new THREE.BoxGeometry(meia, 2.42, 0.03), matFolha));
    [meia / 2, -meia / 2].forEach(function (d) {
      const aro = new THREE.Mesh(new THREE.BoxGeometry(0.05, 2.42, 0.05), matAro);
      aro.position.x = d;
      folha.add(aro);
    });
    folha.position.y = 1.22;
    if (!noEixoX) folha.rotation.y = Math.PI / 2;
    grupo.add(folha);
    return { grupo: folha, lado: lado };
  });
  const porta = {
    x: noEixoX ? centro : fixo,
    z: noEixoX ? fixo : centro,
    abertura: 0,
    posicionar: function () {
      folhas.forEach(function (f) {
        const d = f.lado * (meia / 2 + porta.abertura * meia * 0.94);
        f.grupo.position.x = noEixoX ? centro + d : fixo;
        f.grupo.position.z = noEixoX ? fixo : centro + d;
      });
    },
  };
  porta.posicionar();
  portas.push(porta);
}

function atualizarPortas(portas, pontos, dt) {
  portas.forEach(function (porta) {
    const perto = pontos.some(function (p) { return Math.hypot(p.x - porta.x, p.z - porta.z) < 2.7; });
    const alvo = perto ? 1 : 0;
    if (porta.abertura === alvo) return;
    const passo = dt * 2.6;
    porta.abertura = alvo > porta.abertura ? Math.min(alvo, porta.abertura + passo) : Math.max(alvo, porta.abertura - passo);
    porta.posicionar();
  });
}

function janelasEntre(de, ate, largura, vaoLivre) {
  const lista = [];
  const passo = largura + vaoLivre;
  const quantas = Math.floor((ate - de - vaoLivre) / passo);
  const sobra = ate - de - quantas * passo + vaoLivre;
  for (let i = 0; i < quantas; i++) {
    lista.push({ tipo: "janela", centro: de + sobra / 2 + largura / 2 + i * passo, largura: largura });
  }
  return lista;
}

function porta(centro, largura, automatica) {
  return { tipo: "porta", centro: centro, largura: largura || L.LARGURA_PORTA, automatica: automatica !== false };
}

function chao(cena, pecas) {
  const predio = new THREE.Group();
  const placa = function (x0, x1, z0, z1, cor, ala) {
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(x1 - x0, z1 - z0),
      new THREE.MeshStandardMaterial({ map: texturaGranilite(cor, [(x1 - x0) / 3, (z1 - z0) / 3]), roughness: 0.55 })
    );
    m.rotation.x = -Math.PI / 2;
    m.position.set((x0 + x1) / 2, 0.01, (z0 + z1) / 2);
    m.receiveShadow = true;
    predio.add(m);
    pecas[ala].push({ objeto: m, tipo: "piso" });
  };
  Object.keys(L.SALAS).forEach(function (id) {
    const s = L.SALAS[id];
    placa(s.x0, s.x1, s.z0, s.z1, PISO_DA_ALA[s.ala], s.ala);
  });
  placa(L.CORREDOR.x0, 12, L.CORREDOR.z0, L.CORREDOR.z1, "#e6e0d3", "fase1");
  placa(12, L.CORREDOR.x1, L.CORREDOR.z0, L.CORREDOR.z1, "#e6e0d3", "fase2");
  cena.add(predio);
}

// As linhas coloridas de hospital: seguem o corredor e entram na ala de cada fase.
function linhasGuia(cena, pecas) {
  const grupos = { fase1: new THREE.Group(), fase2: new THREE.Group() };
  cena.add(grupos.fase1, grupos.fase2);
  pecas.fase1.push({ objeto: grupos.fase1, tipo: "ergue" });
  pecas.fase2.push({ objeto: grupos.fase2, tipo: "ergue" });
  const linha = function (x0, z0, x1, z1, cor, deslocamento) {
    const largura = 0.14;
    const horizontal = z0 === z1;
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(horizontal ? Math.abs(x1 - x0) : largura, horizontal ? largura : Math.abs(z1 - z0)),
      new THREE.MeshStandardMaterial({ color: cor, roughness: 0.5 })
    );
    m.rotation.x = -Math.PI / 2;
    m.position.set((x0 + x1) / 2, 0.022, (z0 + z1) / 2 + (deslocamento || 0));
    (cor === L.CORES_DA_ALA.fase1 ? grupos.fase1 : grupos.fase2).add(m);
  };
  const C = L.CORES_DA_ALA;
  linha(-21, 0.5, 6.5, 0.5, C.fase1, 0);
  linha(-21, 0, 17.5, 0, C.fase2, 0);
  linha(-21, -0.5, 25, -0.5, C.fase3, 0);
  ["dados", "imagem"].forEach(function (id) {
    const p = L.SALAS[id].porta;
    linha(p.x, 0.5, p.x, -4, C.fase1);
  });
  const m = L.SALAS.modelos.porta;
  linha(m.x, 0.5, m.x, 4, C.fase1);
  const ev = L.SALAS.evolucao.porta;
  linha(ev.x, 0, ev.x, 4, C.fase2);
  linha(17.5, 0, 17.5, -4, C.fase2);
  linha(17.5, 0, 17.5, 4, C.fase2);
}

function placaDaPorta(cena, sala, dados, pecas) {
  const s = L.SALAS[sala];
  const cor = L.CORES_DA_ALA[s.ala];
  const mat = new THREE.MeshStandardMaterial({ map: texturaPlaca(dados.nome, dados.fase, cor), roughness: 0.6 });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 0.6), mat);
  const p = s.porta;
  if (p.z === -2) {
    m.position.set(p.x, 2.95, -1.88);
  } else if (p.z === 2) {
    m.position.set(p.x, 2.95, 1.88);
    m.rotation.y = Math.PI;
  } else if (sala === "recepcao") {
    m.position.set(-9.88, 2.95, -2.9);
    m.rotation.y = Math.PI / 2;
  } else {
    m.position.set(22.88, 2.95, -3.6);
    m.rotation.y = -Math.PI / 2;
  }
  cena.add(m);
  pecas[s.ala].push({ objeto: m, tipo: "ergue" });
}

function teto(cena) {
  const grupo = new THREE.Group();
  const porAla = {};
  Object.keys(L.CORES_DA_ALA).forEach(function (ala) {
    porAla[ala] = new THREE.Group();
    grupo.add(porAla[ala]);
  });
  const matTeto = material("#fbf9f4", { side: THREE.DoubleSide, emissive: "#fbf9f4", emissiveIntensity: 0.32 });
  // Acima de 1: passa do limiar do brilho, e o painel parece mesmo aceso.
  const matLuz = new THREE.MeshBasicMaterial({ color: new THREE.Color(2.3, 2.25, 2.1) });
  const laje = function (x0, x1, z0, z1, ala) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, z1 - z0), matTeto);
    m.rotation.x = Math.PI / 2;
    m.position.set((x0 + x1) / 2, L.ALTURA_PAREDE, (z0 + z1) / 2);
    porAla[ala].add(m);
  };
  Object.keys(L.SALAS).forEach(function (id) {
    const s = L.SALAS[id];
    if (id !== "obras") laje(s.x0, s.x1, s.z0, s.z1, s.ala);
    const cx = (s.x0 + s.x1) / 2;
    const cz = (s.z0 + s.z1) / 2;
    [-2.4, 2.4].forEach(function (d) {
      const painel = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 0.7), matLuz);
      painel.rotation.x = Math.PI / 2;
      painel.position.set(cx + d, L.ALTURA_PAREDE - 0.02, cz);
      if (id !== "obras") porAla[s.ala].add(painel);
    });
  });
  laje(L.CORREDOR.x0, 12, L.CORREDOR.z0, L.CORREDOR.z1, "fase1");
  laje(12, L.CORREDOR.x1, L.CORREDOR.z0, L.CORREDOR.z1, "fase2");
  for (let x = -7; x < 23; x += 5) {
    const painel = new THREE.Mesh(new THREE.PlaneGeometry(1.8, 0.6), matLuz);
    painel.rotation.x = Math.PI / 2;
    painel.position.set(x, L.ALTURA_PAREDE - 0.02, 0);
    porAla[x < 12 ? "fase1" : "fase2"].add(painel);
  }
  cena.add(grupo);
  return { grupo: grupo, porAla: porAla };
}

function arvore(cena, x, z, escala) {
  const tronco = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.2, 1.4, 7), material("#8a6a4c"));
  tronco.position.set(x, 0.7, z);
  tronco.castShadow = true;
  cena.add(tronco);
  [[0, 1.9, 0, 1], [0.5, 1.6, 0.2, 0.7], [-0.4, 1.7, -0.3, 0.75]].forEach(function (p) {
    const copa = new THREE.Mesh(new THREE.IcosahedronGeometry(0.95 * p[3] * escala, 1), material("#7fae78", { flatShading: true }));
    copa.position.set(x + p[0], p[1] * (0.9 + escala * 0.15), z + p[2]);
    copa.castShadow = true;
    cena.add(copa);
  });
}

// Laco rosa da fachada: a curva fecha e cruza como o simbolo da conscientizacao.
function lacoRosa(cena, x, y, z) {
  const pontos = [
    [-0.62, -1.15, 0.05], [-0.1, -0.2, 0.09], [0.36, 0.5, 0.05], [0.33, 1.02, 0], [0, 1.2, 0],
    [-0.33, 1.02, 0], [-0.36, 0.5, -0.05], [0.1, -0.2, -0.09], [0.62, -1.15, -0.05],
  ].map(function (p) { return new THREE.Vector3(p[0], p[1], p[2]); });
  const m = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pontos), 160, 0.15, 14, false),
    material("#e0528c", { roughness: 0.35 })
  );
  m.scale.z = 0.5;
  m.position.set(x, y, z);
  m.rotation.y = -Math.PI / 2;
  m.castShadow = true;
  cena.add(m);
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.9, 0.3, 28), material("#e7e1d4"));
  base.position.set(x, 0.15, z);
  base.castShadow = true;
  cena.add(base);
}

function arredores(cena) {
  const gramado = new THREE.Mesh(
    new THREE.CircleGeometry(320, 48),
    new THREE.MeshStandardMaterial({ map: texturaGrama([150, 150]), roughness: 1 })
  );
  gramado.rotation.x = -Math.PI / 2;
  gramado.position.set(6, -0.02, 0);
  gramado.receiveShadow = true;
  cena.add(gramado);

  const calcada = new THREE.Mesh(new THREE.PlaneGeometry(16, 5), material("#d6d0c3"));
  calcada.rotation.x = -Math.PI / 2;
  calcada.position.set(-30, 0, 0);
  calcada.receiveShadow = true;
  cena.add(calcada);

  [[-27, -9, 1.1], [-27, 9, 1.2], [-32, -6, 0.9], [-33, 7, 1], [-12, -15, 1.3], [4, -15, 1], [18, -15, 1.2],
    [-10, 15, 1.1], [8, 15, 1.3], [24, 15, 1], [38, 6, 1], [30, 13, 0.9]]
    .forEach(function (p) { arvore(cena, p[0], p[1], p[2]); });

  lacoRosa(cena, -25.4, 1.6, -4.6);

  const letreiro = new THREE.Mesh(
    new THREE.PlaneGeometry(6.4, 1.0),
    new THREE.MeshStandardMaterial({ map: texturaPlaca("Hospital do Tech Challenge", "Unidade fictícia de demonstração", "#e0528c") })
  );
  letreiro.position.set(-22.14, 2.92, 0);
  letreiro.rotation.y = -Math.PI / 2;
  cena.add(letreiro);
}

function construirPredio(cena, salasPorId) {
  const colisores = [];
  const portas = [];
  const pecas = { acolhida: [], fase1: [], fase2: [], fase3: [] };
  const C = L.CORES_DA_ALA;
  const add = function (ala, a, b, op) {
    const grupo = parede(cena, colisores, a, b, Object.assign({ portas: portas }, op));
    pecas[ala].push({ objeto: grupo, tipo: "ergue" });
  };

  chao(cena, pecas);
  linhasGuia(cena, pecas);
  arredores(cena);

  // Fachadas norte e sul.
  add("fase1", { x: -10, z: -11 }, { x: 12, z: -11 }, { aberturas: janelasEntre(-10, 12, 2.6, 1.1), cor: "#b7c4c2" });
  add("fase2", { x: 12, z: -11 }, { x: 23, z: -11 }, { aberturas: janelasEntre(12, 23, 2.6, 1.1), cor: "#b7c4c2" });
  add("fase1", { x: -10, z: 11 }, { x: 1, z: 11 }, { aberturas: janelasEntre(-10, 1, 2.6, 1.1), cor: "#b7c4c2" });
  add("fase2", { x: 1, z: 11 }, { x: 23, z: 11 }, { aberturas: janelasEntre(1, 23, 2.6, 1.1), cor: "#b7c4c2" });

  // Corredor: uma porta por sala e janela interna para ver os robos de fora.
  const janelasDoCorredor = function (centros) {
    return centros.map(function (c) { return { tipo: "janela", centro: c, largura: 2.2 }; });
  };
  [-2, 2].forEach(function (z) {
    add("fase1", { x: -10, z: z }, { x: 1, z: z }, { aberturas: [porta(-4.5)].concat(janelasDoCorredor([-7.6, -1.4])), cor: C.fase1 });
    add(z === -2 ? "fase1" : "fase2", { x: 1, z: z }, { x: 12, z: z }, {
      aberturas: [porta(6.5)].concat(janelasDoCorredor([3.4, 9.6])),
      cor: z === -2 ? C.fase1 : C.fase2,
    });
    add("fase2", { x: 12, z: z }, { x: 23, z: z }, { aberturas: [porta(17.5)].concat(janelasDoCorredor([14.4, 20.6])), cor: C.fase2 });
  });

  // Divisorias entre salas.
  add("fase1", { x: 1, z: -11 }, { x: 1, z: -2 }, {});
  add("fase1", { x: 1, z: 2 }, { x: 1, z: 11 }, {});
  add("fase1", { x: 12, z: -11 }, { x: 12, z: -2 }, {});
  add("fase2", { x: 12, z: 2 }, { x: 12, z: 11 }, {});

  // Recepcao e obras nas pontas, abertas no eixo do corredor.
  add("acolhida", { x: -10, z: -11 }, { x: -10, z: 11 }, { aberturas: [porta(0, 4, false)], cor: C.acolhida });
  add("fase2", { x: 23, z: -11 }, { x: 23, z: 11 }, { aberturas: [porta(0, 4, false)], cor: C.fase3 });
  add("acolhida", { x: -22, z: -8 }, { x: -22, z: 8 }, {
    aberturas: [porta(0, 4)].concat([{ tipo: "janela", centro: -5.6, largura: 3 }, { tipo: "janela", centro: 5.6, largura: 3 }]),
    cor: C.acolhida,
  });
  add("acolhida", { x: -22, z: -8 }, { x: -10, z: -8 }, { cor: C.acolhida });
  add("acolhida", { x: -22, z: 8 }, { x: -10, z: 8 }, { aberturas: janelasEntre(-22, -10, 2.6, 1.3), cor: C.acolhida });
  add("fase3", { x: 23, z: -8 }, { x: 35, z: -8 }, { cor: C.fase3 });
  add("fase3", { x: 23, z: 8 }, { x: 35, z: 8 }, { cor: C.fase3 });
  add("fase3", { x: 35, z: -8 }, { x: 35, z: 8 }, { cor: C.fase3 });

  Object.keys(salasPorId).forEach(function (id) { placaDaPorta(cena, id, salasPorId[id], pecas); });

  const tetos = teto(cena);
  return { colisores: colisores, portas: portas, pecas: pecas, teto: tetos.grupo, tetos: tetos.porAla };
}
