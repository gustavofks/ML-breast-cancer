// Robos funcionarios, montados com primitivas: nenhum modelo 3D externo.
//
// `criarRobo` monta o corpo; `Agente` cuida da rotina (trabalhar, andar, parar para
// falar) e chama a animacao do corpo, que so conhece o estado atual.

function capsula(raio, comprimento, mat) {
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(raio, comprimento, 6, 14), mat);
  m.castShadow = true;
  return m;
}

function esfera(raio, mat, escala) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(raio, 24, 18), mat);
  if (escala) m.scale.set(escala[0], escala[1], escala[2]);
  m.castShadow = true;
  return m;
}

function membro(cor, comprimento, raio) {
  const pivo = new THREE.Group();
  const parte = capsula(raio, comprimento, cor);
  parte.position.y = -comprimento / 2 - raio * 0.4;
  pivo.add(parte);
  return pivo;
}

const ACESSORIOS = {
  jaleco: function (r, matCor) {
    const casaco = new THREE.Mesh(
      new THREE.CylinderGeometry(0.26, 0.33, 0.56, 20, 1, true),
      new THREE.MeshStandardMaterial({ color: "#fbfaf6", roughness: 0.7, side: THREE.DoubleSide })
    );
    casaco.position.y = 0.72;
    casaco.castShadow = true;
    r.corpo.add(casaco);
    const lapela = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.022, 8, 24, Math.PI), matCor);
    lapela.rotation.x = Math.PI / 2 + 0.5;
    lapela.position.set(0, 0.94, 0.04);
    r.corpo.add(lapela);
  },
  oculos: function (r) {
    const mat = new THREE.MeshStandardMaterial({ color: "#20272b", roughness: 0.3 });
    [-0.1, 0.1].forEach(function (x) {
      const aro = new THREE.Mesh(new THREE.TorusGeometry(0.075, 0.012, 8, 20), mat);
      aro.position.set(x, 0.02, 0.222);
      r.cabeca.add(aro);
    });
  },
  touca: function (r, matCor) {
    const touca = new THREE.Mesh(new THREE.SphereGeometry(0.235, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2.1), matCor);
    touca.position.y = 0.05;
    touca.scale.set(1.08, 0.95, 1);
    r.cabeca.add(touca);
  },
  fone: function (r, matCor) {
    const arco = new THREE.Mesh(new THREE.TorusGeometry(0.235, 0.016, 8, 30, Math.PI), new THREE.MeshStandardMaterial({ color: "#2c3539" }));
    arco.position.y = 0.02;
    r.cabeca.add(arco);
    const microfone = new THREE.Mesh(new THREE.SphereGeometry(0.024, 10, 8), matCor);
    microfone.position.set(0.15, -0.14, 0.19);
    r.cabeca.add(microfone);
  },
  capacete: function (r) {
    const mat = new THREE.MeshStandardMaterial({ color: "#f2c230", roughness: 0.4 });
    const topo = new THREE.Mesh(new THREE.SphereGeometry(0.245, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2), mat);
    topo.position.y = 0.06;
    topo.castShadow = true;
    r.cabeca.add(topo);
    const aba = new THREE.Mesh(new THREE.CylinderGeometry(0.29, 0.29, 0.024, 24), mat);
    aba.position.y = 0.065;
    r.cabeca.add(aba);
  },
  prancheta: function (r) {
    const placa = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.27, 0.015), new THREE.MeshStandardMaterial({ color: "#c9a37a" }));
    const folha = new THREE.Mesh(new THREE.PlaneGeometry(0.17, 0.23), new THREE.MeshStandardMaterial({ color: "#fbfaf6" }));
    folha.position.z = 0.009;
    placa.add(folha);
    placa.position.set(0, -0.36, 0.11);
    placa.rotation.x = -0.4;
    r.bracoE.antebraco.add(placa);
  },
  malote: function (r, matCor) {
    const caixaCarga = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.24, 0.26), matCor);
    caixaCarga.position.set(0, 0.86, -0.3);
    caixaCarga.castShadow = true;
    r.corpo.add(caixaCarga);
  },
};

function criarRobo(spec) {
  const raiz = new THREE.Group();
  const corpo = new THREE.Group();
  raiz.add(corpo);
  const cor = new THREE.Color(spec.cor);
  const matBranco = new THREE.MeshStandardMaterial({ color: "#f6f4ee", roughness: 0.34, metalness: 0.05 });
  const matCor = new THREE.MeshStandardMaterial({ color: cor, roughness: 0.38, metalness: 0.08 });
  const matEscuro = new THREE.MeshStandardMaterial({ color: "#232c31", roughness: 0.42 });
  const matVisor = new THREE.MeshPhysicalMaterial({ color: "#0e171b", roughness: 0.16, clearcoat: 1 });
  const matLuz = new THREE.MeshBasicMaterial({ color: "#8ff2e4" });

  const tronco = capsula(0.235, 0.3, matBranco);
  tronco.position.y = 0.74;
  corpo.add(tronco);
  const cinto = new THREE.Mesh(new THREE.CylinderGeometry(0.243, 0.243, 0.1, 24), matCor);
  cinto.position.y = 0.6;
  corpo.add(cinto);
  const peito = esfera(0.05, new THREE.MeshBasicMaterial({ color: spec.cor }));
  peito.position.set(0, 0.86, 0.226);
  corpo.add(peito);

  const pescoco = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 0.1, 12), matEscuro);
  pescoco.position.y = 1.08;
  corpo.add(pescoco);

  const cabeca = new THREE.Group();
  cabeca.position.y = 1.26;
  corpo.add(cabeca);
  cabeca.add(esfera(0.235, matBranco, [1.1, 0.92, 1]));
  // Fatia frontal da esfera: no Three.js, +z fica em phi = pi/2.
  const visor = new THREE.Mesh(new THREE.SphereGeometry(0.21, 28, 16, Math.PI / 2 - 1, 2, 1.08, 0.98), matVisor);
  visor.scale.set(1.12, 0.96, 1.08);
  cabeca.add(visor);
  const olhos = [-0.082, 0.082].map(function (x) {
    const olho = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 10), matLuz);
    olho.scale.set(0.8, 1.3, 0.3);
    olho.position.set(x, 0.015, 0.222);
    cabeca.add(olho);
    return olho;
  });
  const boca = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.014, 0.01), matLuz);
  boca.position.set(0, -0.058, 0.222);
  cabeca.add(boca);
  [-1, 1].forEach(function (s) {
    const orelha = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.05, 16), matCor);
    orelha.rotation.z = Math.PI / 2;
    orelha.position.set(s * 0.258, 0, 0);
    cabeca.add(orelha);
  });
  const antena = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.12, 6), matEscuro);
  antena.position.set(0, 0.26, 0);
  cabeca.add(antena);
  const ponta = esfera(0.028, new THREE.MeshBasicMaterial({ color: spec.cor }));
  ponta.position.set(0, 0.33, 0);
  cabeca.add(ponta);

  function braco(lado) {
    const ombro = membro(matBranco, 0.17, 0.05);
    ombro.position.set(lado * 0.285, 0.96, 0);
    const cotovelo = new THREE.Group();
    cotovelo.position.y = -0.25;
    ombro.add(cotovelo);
    const antebraco = membro(matCor, 0.14, 0.046);
    cotovelo.add(antebraco);
    const mao = esfera(0.058, matEscuro);
    mao.position.y = -0.24;
    antebraco.add(mao);
    corpo.add(ombro);
    return { ombro: ombro, cotovelo: cotovelo, antebraco: antebraco };
  }

  function perna(lado) {
    const quadril = membro(matEscuro, 0.2, 0.062);
    quadril.position.set(lado * 0.115, 0.46, 0);
    const pe = esfera(0.085, matCor, [0.92, 0.55, 1.45]);
    pe.position.set(0, -0.385, 0.04);
    quadril.add(pe);
    corpo.add(quadril);
    return quadril;
  }

  const robo = {
    raiz: raiz, corpo: corpo, cabeca: cabeca, olhos: olhos, boca: boca,
    bracoE: braco(-1), bracoD: braco(1), pernaE: perna(-1), pernaD: perna(1),
    piscar: 2 + Math.random() * 3,
  };
  if (ACESSORIOS[spec.acessorio]) ACESSORIOS[spec.acessorio](robo, matCor);

  // Sombra de contato: da peso ao robo no chao sem depender so do mapa de sombras.
  const [canvas, c] = criarCanvas(128, 128);
  const gradiente = c.createRadialGradient(64, 64, 8, 64, 64, 62);
  gradiente.addColorStop(0, "rgba(30,35,38,.42)");
  gradiente.addColorStop(1, "rgba(30,35,38,0)");
  c.fillStyle = gradiente;
  c.fillRect(0, 0, 128, 128);
  const sombra = new THREE.Mesh(
    new THREE.PlaneGeometry(0.95, 0.95),
    new THREE.MeshBasicMaterial({ map: texturaDe(canvas), transparent: true, depthWrite: false })
  );
  sombra.rotation.x = -Math.PI / 2;
  sombra.position.y = 0.03;
  raiz.add(sombra);
  return robo;
}

function animarRobo(r, estado, t, dt, fase) {
  const ease = 1 - Math.exp(-dt * 10);
  const alvo = { oE: 0, oD: 0, cE: 0, cD: 0, pE: 0, pD: 0, cabX: 0, cabY: 0, bob: 0, abrir: 0.05 };
  const tt = t + fase;

  if (estado === "andando") {
    const passo = Math.sin(tt * 8.5);
    alvo.pE = passo * 0.62;
    alvo.pD = -passo * 0.62;
    alvo.oE = -passo * 0.5;
    alvo.oD = passo * 0.5;
    alvo.cE = -0.3;
    alvo.cD = -0.3;
    alvo.bob = Math.abs(Math.cos(tt * 8.5)) * 0.035;
  } else if (estado === "trabalhando") {
    alvo.oE = -0.85 + Math.sin(tt * 13) * 0.07;
    alvo.oD = -0.85 + Math.sin(tt * 13 + 2) * 0.07;
    alvo.cE = -0.9;
    alvo.cD = -0.9;
    alvo.cabX = 0.26 + Math.sin(tt * 0.6) * 0.06;
    alvo.cabY = Math.sin(tt * 0.33) * 0.3;
    alvo.bob = Math.sin(tt * 1.8) * 0.006;
  } else if (estado === "falando") {
    alvo.oD = -1.1 + Math.sin(tt * 2.3) * 0.34;
    alvo.cD = -0.8 + Math.sin(tt * 3.1) * 0.3;
    alvo.oE = -0.08;
    alvo.cE = -0.2;
    alvo.cabX = Math.sin(tt * 3.4) * 0.05;
    alvo.cabY = Math.sin(tt * 1.3) * 0.08;
    alvo.bob = Math.sin(tt * 3.4) * 0.008;
    alvo.abrir = 0.3 + Math.abs(Math.sin(tt * 15)) * 0.9;
  } else {
    alvo.cabY = Math.sin(tt * 0.5) * 0.35;
    alvo.bob = Math.sin(tt * 1.7) * 0.008;
    alvo.oE = Math.sin(tt * 1.7) * 0.03;
    alvo.oD = -Math.sin(tt * 1.7) * 0.03;
  }

  r.pernaE.rotation.x += (alvo.pE - r.pernaE.rotation.x) * ease;
  r.pernaD.rotation.x += (alvo.pD - r.pernaD.rotation.x) * ease;
  r.bracoE.ombro.rotation.x += (alvo.oE - r.bracoE.ombro.rotation.x) * ease;
  r.bracoD.ombro.rotation.x += (alvo.oD - r.bracoD.ombro.rotation.x) * ease;
  r.bracoE.cotovelo.rotation.x += (alvo.cE - r.bracoE.cotovelo.rotation.x) * ease;
  r.bracoD.cotovelo.rotation.x += (alvo.cD - r.bracoD.cotovelo.rotation.x) * ease;
  r.cabeca.rotation.x += (alvo.cabX - r.cabeca.rotation.x) * ease;
  r.cabeca.rotation.y += (alvo.cabY - r.cabeca.rotation.y) * ease;
  r.corpo.position.y += (alvo.bob - r.corpo.position.y) * ease;
  r.boca.scale.y += ((alvo.abrir * 4 + 1) - r.boca.scale.y) * ease;

  r.piscar -= dt;
  const fechado = r.piscar < 0.12;
  r.olhos.forEach(function (olho) { olho.scale.y = fechado ? 0.12 : 1.3; });
  if (r.piscar < 0) r.piscar = 1.8 + Math.random() * 4;
}

function anguloEntre(a, b) {
  let d = (b - a) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return d;
}

class Agente {
  // `rotina`: pontos com `acao` e `dur`, repetidos em ciclo. Sem rotina, fica parado.
  constructor(robo, x, z, rotina, opcoes) {
    this.robo = robo;
    this.rotina = rotina || [];
    this.indice = 0;
    this.espera = 0;
    this.caminho = [];
    this.estado = "parado";
    this.velocidade = (opcoes && opcoes.velocidade) || 1.25;
    this.fase = Math.random() * 10;
    this.olhar = null;
    this.emConversa = false;
    this.aoChegar = null;
    robo.raiz.position.set(x, 0, z);
    if (this.rotina.length) this.proximoDaRotina();
  }

  get posicao() { return this.robo.raiz.position; }

  proximoDaRotina() {
    const ponto = this.rotina[this.indice];
    this.indice = (this.indice + 1) % this.rotina.length;
    this.caminho = ponto.caminho ? ponto.caminho.slice() : [ponto];
    this.chegada = ponto;
  }

  // Interrompe a rotina: usado pela visita e pelo caso, que levam o robo a um ponto.
  irPara(caminho, aoChegar) {
    this.caminho = caminho.slice();
    this.chegada = null;
    this.aoChegar = aoChegar || null;
    this.espera = 0;
    this.emConversa = false;
  }

  // Coloca o robo no ponto sem andar: a camera ainda esta voando, e o caminho a pe
  // cortaria a mobilia. Quando ela chega, ele ja esta no lugar.
  posicionar(ponto) {
    this.caminho = [];
    this.chegada = null;
    this.aoChegar = null;
    this.espera = 0;
    this.posicao.x = ponto.x;
    this.posicao.z = ponto.z;
  }

  conversar(alvo) {
    this.emConversa = true;
    this.olhar = alvo;
    this.estado = "falando";
  }

  encerrarConversa() {
    this.emConversa = false;
    this.olhar = null;
    this.estado = "parado";
    this.espera = 0.6;
  }

  virarPara(x, z, dt) {
    const p = this.posicao;
    const desejado = Math.atan2(x - p.x, z - p.z);
    const giro = anguloEntre(this.robo.raiz.rotation.y, desejado);
    this.robo.raiz.rotation.y += giro * (1 - Math.exp(-dt * 7));
  }

  atualizar(t, dt) {
    if (this.emConversa) {
      if (this.olhar) this.virarPara(this.olhar.x, this.olhar.z, dt);
    } else if (this.caminho.length) {
      const destino = this.caminho[0];
      const p = this.posicao;
      const dx = destino.x - p.x;
      const dz = destino.z - p.z;
      const d = Math.hypot(dx, dz);
      if (d < 0.08) {
        this.caminho.shift();
        if (!this.caminho.length) this.chegou();
      } else {
        this.estado = "andando";
        const passo = Math.min(this.velocidade * dt, d);
        p.x += (dx / d) * passo;
        p.z += (dz / d) * passo;
        this.virarPara(destino.x, destino.z, dt);
      }
    } else if (this.espera > 0) {
      this.espera -= dt;
      if (this.chegada && this.chegada.olhar) this.virarPara(this.chegada.olhar.x, this.chegada.olhar.z, dt);
    } else if (this.rotina.length) {
      this.proximoDaRotina();
    }
    animarRobo(this.robo, this.estado, t, dt, this.fase);
  }

  chegou() {
    if (this.aoChegar) {
      const aviso = this.aoChegar;
      this.aoChegar = null;
      this.estado = "parado";
      aviso();
      return;
    }
    if (this.chegada) {
      this.estado = this.chegada.acao || "parado";
      this.espera = this.chegada.dur || 3;
    } else {
      this.estado = "parado";
    }
  }
}
