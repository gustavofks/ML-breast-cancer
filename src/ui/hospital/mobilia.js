// Mobilia de cada sala. Os elementos que mostram dados reais ficam aqui:
// as pastas da base, as barras de F1, o enxame da populacao e o paredao de log.

const CORES = {
  madeira: "#c9a37a", metal: "#9aa6a8", escuro: "#2c3539", branco: "#f7f5ef", tela: "#101c21",
  benigno: "#3a86d8", maligno: "#eb6834",
};

function bloco(grupo, colisores, x0, x1, z0, z1, altura, cor, y) {
  const m = caixa(x1 - x0, altura, z1 - z0, material(cor), (x0 + x1) / 2, (y || 0) + altura / 2, (z0 + z1) / 2);
  grupo.add(m);
  if (colisores) colisores.push({ x0: x0, x1: x1, z0: z0, z1: z1 });
  return m;
}

function mesa(grupo, colisores, x, z, largura, profundidade, rotacionada) {
  const w = rotacionada ? profundidade : largura;
  const d = rotacionada ? largura : profundidade;
  const tampo = caixa(w, 0.07, d, material(CORES.madeira, { roughness: 0.5 }), x, 0.76, z);
  grupo.add(tampo);
  [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(function (s) {
    grupo.add(caixa(0.06, 0.74, 0.06, material(CORES.metal, { metalness: 0.4 }), x + s[0] * (w / 2 - 0.08), 0.37, z + s[1] * (d / 2 - 0.08)));
  });
  colisores.push({ x0: x - w / 2, x1: x + w / 2, z0: z - d / 2, z1: z + d / 2 });
}

function monitor(grupo, x, z, giro, cor) {
  const g = new THREE.Group();
  g.add(caixa(0.62, 0.38, 0.04, material(CORES.escuro), 0, 1.13, 0));
  const tela = new THREE.Mesh(new THREE.PlaneGeometry(0.56, 0.32), new THREE.MeshBasicMaterial({ map: texturaTela(cor || "#3fc5b5") }));
  tela.position.set(0, 1.13, 0.024);
  g.add(tela);
  g.add(caixa(0.05, 0.16, 0.05, material(CORES.metal), 0, 0.88, 0));
  g.add(caixa(0.22, 0.02, 0.16, material(CORES.metal), 0, 0.805, 0));
  g.position.set(x, 0, z);
  g.rotation.y = giro || 0;
  grupo.add(g);
}

function cadeira(grupo, x, z, giro, cor) {
  const g = new THREE.Group();
  const mat = material(cor || "#5d8f88");
  g.add(caixa(0.44, 0.07, 0.44, mat, 0, 0.46, 0));
  g.add(caixa(0.44, 0.46, 0.06, mat, 0, 0.72, -0.2));
  g.add(caixa(0.05, 0.44, 0.05, material(CORES.metal), 0, 0.22, 0));
  g.add(caixa(0.36, 0.03, 0.36, material(CORES.metal), 0, 0.03, 0));
  g.position.set(x, 0, z);
  g.rotation.y = giro || 0;
  grupo.add(g);
}

function planta(grupo, colisores, x, z) {
  const vaso = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.2, 0.46, 12), material("#d8cdbb"));
  vaso.position.set(x, 0.23, z);
  vaso.castShadow = true;
  grupo.add(vaso);
  [[0, 0.82, 0, 0.36], [0.16, 0.66, 0.08, 0.26], [-0.14, 0.7, -0.1, 0.28]].forEach(function (p) {
    const folha = new THREE.Mesh(new THREE.IcosahedronGeometry(p[3], 1), material("#6fa577", { flatShading: true }));
    folha.position.set(x + p[0], p[1], z + p[2]);
    folha.castShadow = true;
    grupo.add(folha);
  });
  colisores.push({ x0: x - 0.26, x1: x + 0.26, z0: z - 0.26, z1: z + 0.26 });
}

// Quadro de parede com figura real. Fica em `quadros` para a mira e o clique abrirem
// a figura inteira; a textura carrega depois, e a moldura nunca fica vazia.
function quadro(grupo, quadros, dados, posicao, giro, largura) {
  const altura = largura * 0.66;
  const g = new THREE.Group();
  g.add(caixa(largura + 0.14, altura + 0.14, 0.05, material(CORES.escuro), 0, 0, 0, false));
  const mat = new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.7 });
  const face = new THREE.Mesh(new THREE.PlaneGeometry(largura, altura), mat);
  face.position.z = 0.03;
  g.add(face);
  const fonte = FIGURAS[dados.figura];
  if (fonte) {
    new THREE.TextureLoader().load(fonte, function (textura) {
      textura.colorSpace = THREE.SRGBColorSpace;
      textura.anisotropy = 8;
      const proporcao = textura.image.width / textura.image.height;
      face.scale.x = Math.min(1, altura * proporcao / largura);
      face.scale.y = Math.min(1, (largura / proporcao) / altura);
      mat.map = textura;
      mat.needsUpdate = true;
    });
  }
  const rotulo = new THREE.Mesh(
    new THREE.PlaneGeometry(largura * 0.8, largura * 0.2),
    new THREE.MeshBasicMaterial({ map: texturaRotulo(dados.titulo), transparent: true })
  );
  rotulo.position.set(0, -altura / 2 - largura * 0.1, 0.03);
  g.add(rotulo);
  g.position.set(posicao.x, posicao.y, posicao.z);
  g.rotation.y = giro;
  grupo.add(g);
  face.userData.quadro = dados;
  quadros.push({ dados: dados, malha: face, posicao: new THREE.Vector3(posicao.x, posicao.y, posicao.z) });
}

function moveisDaRecepcao(g, colisores) {
  // Balcao encostado ao norte: o eixo entre a entrada e o corredor fica livre.
  const balcao = caixa(6, 0.8, 1.1, material("#f7f5ef"), -16, 0.4, -5);
  const tampo = caixa(6.2, 0.08, 1.3, material("#e0528c", { roughness: 0.4 }), -16, 0.84, -5);
  g.add(balcao, tampo);
  colisores.push({ x0: -19.1, x1: -12.9, z0: -5.65, z1: -4.35 });
  const posto = new THREE.Group();
  monitor(posto, -17.6, -5.1, 0, "#f2a6c6");
  posto.position.y = -0.28;
  g.add(posto);
  const mural = new THREE.Mesh(
    new THREE.PlaneGeometry(9, 2.8),
    new THREE.MeshStandardMaterial({
      map: texturaMural([
        { titulo: "Fase 1", cor: L.CORES_DA_ALA.fase1, linhas: ["Modelos sobre exames", "e entrega de imagem"] },
        { titulo: "Fase 2", cor: L.CORES_DA_ALA.fase2, linhas: ["Algoritmo genético,", "API e laudos com LLM"] },
        { titulo: "Fase 3", cor: L.CORES_DA_ALA.fase3, linhas: ["Assistente médico,", "em obras"] },
      ]),
      roughness: 0.7,
    })
  );
  mural.position.set(-16, 1.95, -7.87);
  g.add(mural);
  for (let i = 0; i < 5; i++) {
    cadeira(g, -20 + i * 1, 6.9, Math.PI, "#7fb3ad");
    cadeira(g, -20 + i * 1, 4.2, 0, "#7fb3ad");
  }
  planta(g, colisores, -21.2, 7.2);
  planta(g, colisores, -21.2, -7.2);
  planta(g, colisores, -10.8, 7.2);
  planta(g, colisores, -10.8, -7.2);
  // Mesa do medico: ponto final de todo caso.
  mesa(g, colisores, -12, 5.6, 1.8, 0.9);
  monitor(g, -12, 5.8, Math.PI);
  cadeira(g, -12, 6.5, Math.PI, "#e0528c");
}

// 569 pastas, uma por exame, na cor da classe. Uma malha instanciada: uma chamada de
// desenho so, e a quantidade vem do dataset.
function estanteDePastas(g, colisores, benignos, malignos) {
  const total = benignos + malignos;
  const porPrateleira = 72;
  const prateleiras = Math.ceil(total / porPrateleira / 2);
  const geometria = new THREE.BoxGeometry(0.05, 0.27, 0.22);
  const malha = new THREE.InstancedMesh(geometria, new THREE.MeshStandardMaterial({ roughness: 0.6 }), total);
  const m = new THREE.Matrix4();
  const cor = new THREE.Color();
  const origens = [{ x: -9.2, fim: -5.6 }, { x: -3.6, fim: 0 }];
  for (let i = 0; i < total; i++) {
    const estante = Math.floor(i / (porPrateleira * prateleiras));
    const resto = i % (porPrateleira * prateleiras);
    const nivel = Math.floor(resto / porPrateleira);
    const coluna = resto % porPrateleira;
    m.makeTranslation(origens[estante].x + coluna * 0.0505, 0.34 + nivel * 0.46, -10.55);
    malha.setMatrixAt(i, m);
    malha.setColorAt(i, cor.set(i < benignos ? CORES.benigno : CORES.maligno));
  }
  malha.castShadow = true;
  g.add(malha);
  origens.forEach(function (o) {
    for (let n = 0; n <= prateleiras; n++) {
      g.add(caixa(3.8, 0.04, 0.4, material("#b9c4c4"), o.x + 1.8, 0.18 + n * 0.46, -10.55));
    }
    [-0.1, 3.75].forEach(function (d) {
      g.add(caixa(0.05, 0.2 + prateleiras * 0.46, 0.4, material("#b9c4c4"), o.x + d, 0.1 + prateleiras * 0.23, -10.55));
    });
    colisores.push({ x0: o.x - 0.1, x1: o.x + 3.8, z0: -10.8, z1: -10.3 });
  });
}

function moveisDosDados(g, colisores, quadros, sala, dataset) {
  estanteDePastas(g, colisores, dataset.benignos, dataset.malignos);
  mesa(g, colisores, -4.5, -7.2, 4.2, 1);
  monitor(g, -5.6, -7.4, 0);
  monitor(g, -3.4, -7.4, 0, "#6fb0e8");
  quadro(g, quadros, sala.quadros[0], { x: -9.86, y: 1.85, z: -6.4 }, Math.PI / 2, 2.5);
  quadro(g, quadros, sala.quadros[1], { x: 0.86, y: 1.85, z: -6.4 }, -Math.PI / 2, 2.5);
  planta(g, colisores, -9.2, -2.8);
}

// Podio dos tres modelos. A altura diz so a colocacao: em barras, F1 tao proximos so
// pareceriam diferentes com o eixo cortado, e isso exageraria a disputa. O numero fica na placa.
function podio(g, colisores, itens) {
  const cores = ["#2f7fd6", "#7fb3ad", "#9db5c9"];
  const alturas = [1.15, 0.8, 0.5];
  itens.forEach(function (item, i) {
    const x = -7.8 + i * 2.1;
    const altura = alturas[i];
    g.add(caixa(1.5, altura, 1.3, material(cores[i], { roughness: 0.4 }), x, altura / 2, 9.4));
    colisores.push({ x0: x - 0.75, x1: x + 0.75, z0: 8.75, z1: 10.05 });
    const rotulo = new THREE.Sprite(new THREE.SpriteMaterial({ map: texturaRotulo(item.modelo + " " + L.numero(item.f1, 3)), depthTest: true }));
    rotulo.scale.set(2.2, 0.55, 1);
    rotulo.position.set(x, altura + 0.42, 9.2);
    g.add(rotulo);
  });
}

function moveisDosModelos(g, colisores, quadros, sala) {
  podio(g, colisores, sala.podio);
  mesa(g, colisores, -8, 5, 2.6, 0.9, true);
  monitor(g, -8.2, 5, Math.PI / 2, "#6fb0e8");
  quadro(g, quadros, sala.quadros[0], { x: 0.86, y: 1.85, z: 6.4 }, -Math.PI / 2, 2.9);
  quadro(g, quadros, sala.quadros[1], { x: -1.4, y: 1.85, z: 10.86 }, Math.PI, 2.5);
  planta(g, colisores, 0.3, 2.8);
}

function moveisDaImagem(g, colisores, quadros, sala) {
  // Maca e aparelho de ultrassom.
  g.add(caixa(0.9, 0.12, 2.2, material("#dfe8ea"), 3.1, 0.72, -8.2));
  g.add(caixa(0.8, 0.1, 0.5, material("#f7f5ef"), 3.1, 0.82, -9.05));
  g.add(caixa(0.5, 0.66, 1.6, material(CORES.metal), 3.1, 0.33, -8.2));
  colisores.push({ x0: 2.6, x1: 3.6, z0: -9.4, z1: -7.1 });
  g.add(caixa(0.56, 1.0, 0.5, material("#e8edee"), 4.5, 0.5, -9.2));
  monitor(g, 4.5, -9.2, 0.3, "#b8c8d0");
  g.add(caixa(0.6, 0.06, 0.4, material(CORES.escuro), 4.5, 1.0, -8.9));
  colisores.push({ x0: 4.2, x1: 4.8, z0: -9.5, z1: -8.9 });
  quadro(g, quadros, sala.quadros[0], { x: 8.2, y: 1.85, z: -10.86 }, 0, 3.4);
  quadro(g, quadros, sala.quadros[1], { x: 11.86, y: 1.85, z: -6.4 }, -Math.PI / 2, 3.2);
  mesa(g, colisores, 6.5, -7.2, 2.4, 0.9);
  monitor(g, 6.5, -7.4, 0, "#b79af0");
}

// Enxame da populacao. Raio segue a diversidade real da execucao servida, geracao a
// geracao; a altura sobe com o fitness. Nao e ilustracao: a forma vem do historico.
function criarEnxame(g, dados, centro) {
  const n = dados.populacao;
  const esferas = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.08, 12, 10),
    new THREE.MeshStandardMaterial({ color: "#b79af0", emissive: "#7a4fd0", emissiveIntensity: 0.9, roughness: 0.3 }),
    n
  );
  g.add(esferas);
  const sementes = [];
  for (let i = 0; i < n; i++) {
    sementes.push({
      a: (i * 2.399963) % (Math.PI * 2),
      r: 0.35 + ((i * 37) % 100) / 150,
      y: ((i * 53) % 100) / 100,
      v: 0.3 + ((i * 17) % 10) / 20,
    });
  }
  const historico = dados.historico;
  const divMax = Math.max.apply(null, historico.map(function (h) { return h.diversidade; }));
  const fitMin = historico[0].fitness;
  const fitMax = historico[historico.length - 1].fitness;
  const m = new THREE.Matrix4();
  const estado = { geracao: 0 };

  const base = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.7, 0.2, 40), material("#3a3550", { roughness: 0.4 }));
  base.position.set(centro.x, 0.1, centro.z);
  base.castShadow = true;
  g.add(base);
  const anel = new THREE.Mesh(new THREE.TorusGeometry(1.5, 0.03, 8, 60), new THREE.MeshBasicMaterial({ color: "#b79af0" }));
  anel.rotation.x = Math.PI / 2;
  anel.position.set(centro.x, 0.22, centro.z);
  g.add(anel);

  return {
    colisor: { x0: centro.x - 1.6, x1: centro.x + 1.6, z0: centro.z - 1.6, z1: centro.z + 1.6 },
    estado: estado,
    avancar: function (t) {
      // Uma geracao a cada 0,4 s e pausa no fim, para ler o resultado.
      const ciclo = historico.length + 8;
      estado.geracao = Math.min(Math.floor(t / 0.4) % ciclo, historico.length - 1);
      const h = historico[estado.geracao];
      const raio = 0.18 + 1.25 * (h.diversidade / divMax);
      const subida = (h.fitness - fitMin) / Math.max(fitMax - fitMin, 1e-9);
      for (let i = 0; i < n; i++) {
        const s = sementes[i];
        const a = s.a + t * s.v;
        m.makeTranslation(
          centro.x + Math.cos(a) * raio * s.r,
          0.9 + subida * 0.7 + (s.y - 0.5) * raio + Math.sin(t * 1.3 + i) * 0.04,
          centro.z + Math.sin(a) * raio * s.r
        );
        esferas.setMatrixAt(i, m);
      }
      esferas.instanceMatrix.needsUpdate = true;
    },
  };
}

function moveisDaEvolucao(g, colisores, quadros, sala, animados) {
  const enxame = criarEnxame(g, sala.enxame, { x: 7.4, z: 6.8 });
  colisores.push(enxame.colisor);
  animados.push(enxame);
  mesa(g, colisores, 3, 7.6, 2, 0.9, true);
  monitor(g, 2.8, 7.6, Math.PI / 2, "#b79af0");
  quadro(g, quadros, sala.quadros[0], { x: 4.2, y: 1.85, z: 10.86 }, Math.PI, 3);
  quadro(g, quadros, sala.quadros[1], { x: 11.86, y: 1.85, z: 7.2 }, -Math.PI / 2, 2.7);
}

function moveisDosLaudos(g, colisores) {
  mesa(g, colisores, 17.5, -7.4, 3.8, 1);
  monitor(g, 16.5, -7.6, 0, "#f2a6c6");
  monitor(g, 18.5, -7.6, 0, "#7fe3cf");
  // Pilhas de laudos aprovados e o selo das quatro verificacoes.
  for (let i = 0; i < 3; i++) {
    bloco(g, null, 13 + i * 0.7, 13.5 + i * 0.7, -10.5, -10, 0.1 + i * 0.1, "#f7f5ef", 0.8);
  }
  mesa(g, colisores, 14, -10.2, 3, 0.8);
  // Painel das quatro verificacoes, na parede cega: a janela atrapalharia a leitura.
  g.add(caixa(0.05, 2.5, 2.7, material(CORES.tela), 22.86, 1.75, -6.2, false));
  ["Fidelidade", "Segurança", "Cobertura", "Formato"].forEach(function (nome, i) {
    const selo = new THREE.Mesh(
      new THREE.PlaneGeometry(2.3, 0.58),
      new THREE.MeshBasicMaterial({ map: texturaRotulo("✓ " + nome, "#7fe3cf"), transparent: true })
    );
    selo.position.set(22.82, 2.6 - i * 0.56, -6.2);
    selo.rotation.y = -Math.PI / 2;
    g.add(selo);
  });
  planta(g, colisores, 22.2, -2.8);
}

function moveisDaCentral(g, colisores, sala, animados) {
  const painel = criarPainelDeLog(sala.log);
  const tela = new THREE.Mesh(new THREE.PlaneGeometry(6.4, 3.2 * 0.8), new THREE.MeshBasicMaterial({ map: painel.textura }));
  tela.position.set(17.5, 1.9, 10.86);
  tela.rotation.y = Math.PI;
  g.add(caixa(6.6, 2.76, 0.08, material(CORES.escuro), 17.5, 1.9, 10.9, false));
  g.add(tela);
  animados.push({ avancar: function (t, dt) { painel.avancar(dt); } });
  mesa(g, colisores, 17.5, 8, 4.2, 1);
  monitor(g, 16.3, 8.1, Math.PI, "#7fe3cf");
  monitor(g, 18.7, 8.1, Math.PI, "#ffb26b");
  // Rack do servidor com luzes piscando.
  bloco(g, colisores, 21.6, 22.7, 8.6, 10.6, 2.1, CORES.escuro);
  const luzes = [];
  for (let i = 0; i < 12; i++) {
    const luz = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.05, 0.14), new THREE.MeshBasicMaterial({ color: "#7fe3cf" }));
    luz.position.set(21.58, 0.5 + (i % 6) * 0.26, 9.1 + Math.floor(i / 6) * 0.9);
    g.add(luz);
    luzes.push(luz);
  }
  animados.push({ avancar: function (t) {
    luzes.forEach(function (luz, i) { luz.visible = Math.sin(t * (2 + i * 0.7) + i) > -0.3; });
  } });
}

function cone(g, x, z) {
  const m = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.55, 14), material("#ee7a2b"));
  m.position.set(x, 0.3, z);
  m.castShadow = true;
  g.add(m);
  g.add(caixa(0.42, 0.04, 0.42, material(CORES.escuro), x, 0.02, z));
}

function faixa(g, x0, z0, x1, z1) {
  const comprimento = Math.hypot(x1 - x0, z1 - z0);
  const m = new THREE.Mesh(new THREE.BoxGeometry(comprimento, 0.12, 0.01), new THREE.MeshStandardMaterial({ map: texturaFaixaDeObra() }));
  m.position.set((x0 + x1) / 2, 0.95, (z0 + z1) / 2);
  m.rotation.y = -Math.atan2(z1 - z0, x1 - x0);
  g.add(m);
}

function moveisDaObra(g, colisores, sala) {
  [[23.6, -1.9], [23.6, 1.9], [25.5, -6.6], [33.4, 6.6], [33.4, -6.6]].forEach(function (p) { cone(g, p[0], p[1]); });
  faixa(g, 23.6, -1.9, 23.6, -0.7);
  faixa(g, 23.6, 0.7, 23.6, 1.9);
  // Andaime.
  const matTubo = material(CORES.metal, { metalness: 0.5, roughness: 0.4 });
  [[33.8, -7.4], [33.8, -4.6], [31.8, -7.4], [31.8, -4.6]].forEach(function (p) {
    g.add(caixa(0.07, 3.2, 0.07, matTubo, p[0], 1.6, p[1]));
  });
  [1.2, 2.4].forEach(function (y) {
    g.add(caixa(2.2, 0.06, 3, material(CORES.madeira), 32.8, y, -6));
  });
  colisores.push({ x0: 31.7, x1: 33.9, z0: -7.5, z1: -4.5 });
  // Pedestais do fluxo pedido no enunciado: ainda vazios, e ligados por uma linha no chao.
  const n = sala.fluxo.length;
  sala.fluxo.forEach(function (nome, i) {
    const x = 26 + i * 1.9;
    const z = 4.4;
    const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.5, 0.7, 20), material("#d6cfbf"));
    pedestal.position.set(x, 0.35, z);
    pedestal.castShadow = true;
    g.add(pedestal);
    colisores.push({ x0: x - 0.5, x1: x + 0.5, z0: z - 0.5, z1: z + 0.5 });
    const rotulo = new THREE.Sprite(new THREE.SpriteMaterial({ map: texturaRotulo(nome, "#f2c230") }));
    rotulo.scale.set(2.2, 0.55, 1);
    rotulo.position.set(x, 1.2 + (i % 2) * 0.5, z);
    g.add(rotulo);
    if (i < n - 1) g.add(caixa(1.5, 0.02, 0.1, material("#e0a91f"), x + 0.95, 0.03, z, false));
  });
  // Sacos e caixas de obra, sem teto: esta ala ainda nao tem laje.
  bloco(g, colisores, 25.2, 26.6, -7.5, -6.5, 0.7, "#c9b48f");
  bloco(g, colisores, 26.8, 27.6, -7.5, -6.8, 0.45, "#b7a27f");
}

function construirMobilia(cena, colisores, salas, dataset) {
  const grupo = new THREE.Group();
  const quadros = [];
  const animados = [];
  moveisDaRecepcao(grupo, colisores);
  moveisDosDados(grupo, colisores, quadros, salas.dados, dataset);
  moveisDosModelos(grupo, colisores, quadros, salas.modelos);
  moveisDaImagem(grupo, colisores, quadros, salas.imagem);
  moveisDaEvolucao(grupo, colisores, quadros, salas.evolucao, animados);
  moveisDosLaudos(grupo, colisores);
  moveisDaCentral(grupo, colisores, salas.central, animados);
  moveisDaObra(grupo, colisores, salas.obras);
  cena.add(grupo);
  return { quadros: quadros, animados: animados };
}
