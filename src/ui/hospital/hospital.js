// Cena, modos e laco principal. Os demais arquivos deste diretorio vem antes, no mesmo modulo.

const SALAS_POR_ID = Object.fromEntries(DADOS.salas.map(function (s) { return [s.id, s]; }));

// Rotina de cada robo: pontos dentro da propria sala, sem cruzar a mobilia. O primeiro
// ponto e o posto de trabalho, de onde ele sai para circular.
const ROTINAS = {
  recepcao: [
    { x: -17.6, z: -6, acao: "trabalhando", dur: 7, olhar: { x: -17.6, z: -5 } },
    { x: -14.4, z: -6, acao: "parado", dur: 3, olhar: { x: -14.4, z: 0 } },
  ],
  dados: [
    { x: -4.5, z: -6.3, acao: "trabalhando", dur: 6, olhar: { x: -4.5, z: -7.2 } },
    { x: -7.6, z: -9.6, acao: "trabalhando", dur: 4, olhar: { x: -7.6, z: -11 } },
    { x: -2, z: -9.6, acao: "trabalhando", dur: 4, olhar: { x: -2, z: -11 } },
  ],
  modelos: [
    { x: -6.9, z: 5, acao: "trabalhando", dur: 6, olhar: { x: -8, z: 5 } },
    { x: -2.6, z: 7.6, acao: "parado", dur: 4, olhar: { x: 1, z: 6.4 } },
    { x: -6, z: 7.6, acao: "parado", dur: 4, olhar: { x: -6, z: 9.4 } },
  ],
  imagem: [
    { x: 6.5, z: -6.3, acao: "trabalhando", dur: 6, olhar: { x: 6.5, z: -7.2 } },
    { x: 4.4, z: -7.9, acao: "trabalhando", dur: 5, olhar: { x: 4.5, z: -9.2 } },
    { x: 9.4, z: -8.4, acao: "parado", dur: 4, olhar: { x: 8.2, z: -11 } },
  ],
  evolucao: [
    { x: 9.7, z: 3.4, acao: "trabalhando", dur: 6, olhar: { x: 10.7, z: 3.4 } },
    { x: 5.2, z: 4.2, acao: "parado", dur: 5, olhar: { x: 7.4, z: 6.8 } },
    { x: 9.8, z: 3.8, acao: "parado", dur: 5, olhar: { x: 7.4, z: 6.8 } },
  ],
  laudos: [
    { x: 17.5, z: -6.5, acao: "trabalhando", dur: 6, olhar: { x: 17.5, z: -7.4 } },
    { x: 14, z: -9.2, acao: "trabalhando", dur: 5, olhar: { x: 14, z: -10.2 } },
  ],
  central: [
    { x: 17.5, z: 7.1, acao: "trabalhando", dur: 7, olhar: { x: 17.5, z: 8 } },
    { x: 20.6, z: 9.4, acao: "trabalhando", dur: 4, olhar: { x: 21.6, z: 9.4 } },
    { x: 14.6, z: 9.6, acao: "parado", dur: 4, olhar: { x: 17.5, z: 11 } },
  ],
  obras: [
    { x: 26.4, z: -3.6, acao: "parado", dur: 5, olhar: { x: 29.8, z: 4.4 } },
    { x: 30.6, z: -5, acao: "trabalhando", dur: 5, olhar: { x: 32, z: -6 } },
  ],
};

function poseDaSala(id) {
  const c = L.SALAS[id].camera;
  return { posicao: new THREE.Vector3(...c.de), olhar: new THREE.Vector3(...c.para) };
}

function criarRenderizador(tela) {
  const renderizador = new THREE.WebGLRenderer({ canvas: tela, antialias: false, powerPreference: "high-performance" });
  renderizador.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderizador.shadowMap.enabled = true;
  renderizador.shadowMap.type = THREE.PCFSoftShadowMap;
  renderizador.toneMapping = THREE.NeutralToneMapping;
  renderizador.toneMappingExposure = 1;
  return renderizador;
}

function rotuloFlutuante(texto, cor) {
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texturaRotulo(texto, cor), depthTest: false, transparent: true }));
  sprite.scale.set(1.9, 0.475, 1);
  sprite.renderOrder = 5;
  return sprite;
}

// Nomes das salas sobre a maquete, clicaveis: so aparecem sem o teto.
function criarRotulosDasSalas(cena) {
  const grupo = new THREE.Group();
  DADOS.salas.forEach(function (sala) {
    const s = L.SALAS[sala.id];
    const rotulo = new THREE.Sprite(new THREE.SpriteMaterial({ map: texturaRotulo(sala.nome), depthTest: false, transparent: true }));
    rotulo.scale.set(9, 2.25, 1);
    rotulo.position.set((s.x0 + s.x1) / 2, 4.6, (s.z0 + s.z1) / 2);
    rotulo.renderOrder = 6;
    rotulo.userData.sala = sala.id;
    grupo.add(rotulo);
  });
  cena.add(grupo);
  return grupo;
}

function criarEquipe(cena) {
  const equipe = {};
  DADOS.salas.forEach(function (sala) {
    const robo = criarRobo(sala.robo);
    cena.add(robo.raiz);
    const inicio = ROTINAS[sala.id][0];
    const agente = new Agente(robo, inicio.x, inicio.z, ROTINAS[sala.id]);
    agente.sala = sala;
    const rotulo = rotuloFlutuante(sala.robo.nome);
    rotulo.position.y = 1.95;
    robo.raiz.add(rotulo);
    agente.rotulo = rotulo;
    equipe[sala.id] = agente;
  });
  return equipe;
}

// Maqueiros que cruzam o corredor: dao vida ao predio e nao falam.
function criarCirculantes(cena) {
  const passeios = [
    { spec: { cor: "#4da89b", acessorio: "prancheta" }, z: -1.1, de: -8, ate: 21 },
    { spec: { cor: "#9a78e0", acessorio: "malote" }, z: 1.1, de: 21, ate: -8 },
  ];
  return passeios.map(function (p) {
    const robo = criarRobo(p.spec);
    cena.add(robo.raiz);
    return new Agente(robo, p.de, p.z, [
      { x: p.ate, z: p.z, acao: "parado", dur: 2 },
      { x: p.de, z: p.z, acao: "parado", dur: 2 },
    ], { velocidade: 1.5 });
  });
}

// Portador do caso: anda o predio inteiro levando a pasta.
function criarPortador(cena) {
  const robo = criarRobo({ cor: "#e0528c", acessorio: "malote" });
  robo.raiz.visible = false;
  cena.add(robo.raiz);
  const rotulo = rotuloFlutuante("Caso em andamento", "#ffd0e2");
  rotulo.position.y = 1.95;
  robo.raiz.add(rotulo);
  return new Agente(robo, -19, 0, []);
}

class Hospital {
  constructor() {
    const tela = $("cena");
    this.tela = tela;
    this.renderizador = criarRenderizador(tela);
    this.cena = new THREE.Scene();
    this.camera3d = new THREE.PerspectiveCamera(52, 1, 0.1, 700);
    const predio = construirPredio(this.cena, SALAS_POR_ID);
    this.predio = predio;
    const mobilia = construirMobilia(this.cena, predio.colisores, SALAS_POR_ID, DADOS, predio.pecas);
    this.quadros = mobilia.quadros;
    this.animados = mobilia.animados;
    this.palco = mobilia.palco;
    this.ceu = new Ceu(this.cena, this.renderizador);
    this.pos = new Pos(this.renderizador);
    this.obra = new Obra(predio.pecas);
    this.equipe = criarEquipe(this.cena);
    this.rotulosDasSalas = criarRotulosDasSalas(this.cena);
    this.circulantes = criarCirculantes(this.cena);
    this.portador = criarPortador(this.cena);
    this.camera = new Camera(this.camera3d, tela, predio.colisores);
    this.raio = new THREE.Raycaster();
    this.relogio = new THREE.Clock();
    this.tempo = 0;
    this.modo = null;
    this.conversa = null;
    this.visita = null;
    this.caso = null;
    this.alvoProximo = null;
    this.diretor = new Diretor(this);
    this.estudio = new Estudio(this);
    this.ligarEventos();
    this.ajustarTamanho();
    addEventListener("resize", () => this.ajustarTamanho());
  }

  ajustarTamanho() {
    if (this.estudio.aberto) return;
    const w = innerWidth;
    const h = innerHeight;
    this.renderizador.setSize(w, h, false);
    const escala = this.renderizador.getPixelRatio();
    this.pos.redimensionar(Math.round(w * escala), Math.round(h * escala));
    this.camera3d.aspect = w / h;
    this.camera3d.updateProjectionMatrix();
  }

  ligarEventos() {
    document.querySelectorAll(".modos [data-modo]").forEach((botao) => {
      botao.addEventListener("click", () => this.trocarModo(botao.dataset.modo));
    });
    $("fechar-dialogo").addEventListener("click", () => this.encerrarConversa());
    $("hora").addEventListener("click", () => this.proximaHora());
    $("visita-anterior").addEventListener("click", () => this.passoDaVisita(-1));
    $("visita-proxima").addEventListener("click", () => this.passoDaVisita(1));
    $("visita-auto").addEventListener("click", () => this.alternarAutomatico());
    $("figura-fechar").addEventListener("click", () => $("figura").close());
    $("caso-fechar").addEventListener("click", () => $("escolha-caso").close());
    this.tela.addEventListener("click", (e) => this.aoClicar(e));
    addEventListener("keydown", (e) => this.aoTeclar(e));
  }

  // Manha, fim de tarde e noite, fora dos filmes: o hospital muda bastante.
  proximaHora() {
    const horas = [[10, "Manhã"], [17.6, "Fim de tarde"], [21, "Noite"]];
    this.indiceDaHora = ((this.indiceDaHora || 0) + 1) % horas.length;
    this.ceu.definirHora(horas[this.indiceDaHora][0]);
    $("hora").textContent = horas[this.indiceDaHora][1];
  }

  // Robos e lajes so existem nas alas prontas: num terreno vazio, ficariam flutuando.
  sincronizarObra() {
    const obra = this.obra;
    Object.values(this.equipe).forEach(function (a) {
      a.robo.raiz.visible = obra.progresso[L.SALAS[a.sala.id].ala] >= 0.97;
    });
    const corredorPronto = obra.pronta("fase1") && obra.pronta("fase2");
    this.circulantes.forEach(function (a) { a.robo.raiz.visible = corredorPronto; });
    Object.keys(this.predio.tetos).forEach((ala) => {
      this.predio.tetos[ala].visible = obra.pronta(ala);
    });
  }

  definirTetoDoFilme(semTeto) {
    this.predio.teto.visible = !semTeto;
    this.rotulosDasSalas.visible = false;
  }

  // Depois do estudio, o hospital volta inteiro, como na visita.
  restaurarMundo() {
    this.obra.terminarTudo();
    this.sincronizarObra();
    const palco = this.palco;
    palco.pastas.posicionar(palco.pastas.total, 1);
    palco.cofre.abrir(0);
    palco.podio.subir(1);
    palco.selos.acender(1);
    palco.regua.definir(DADOS.modelo.limiar);
    palco.enxame.estado.relogio = null;
    palco.fluxo.acender(0);
    this.ceu.definirHora(10);
    this.indiceDaHora = 0;
    $("hora").textContent = "Manhã";
    this.camera3d.updateProjectionMatrix();
  }

  aoTeclar(e) {
    if (e.code === "KeyE" && this.modo === "andar") {
      if (this.conversa) this.encerrarConversa();
      else if (this.alvoProximo) this.interagir(this.alvoProximo);
    }
    if (e.code === "Escape" && this.conversa) this.encerrarConversa();
    if (this.modo === "visita") {
      if (e.code === "ArrowRight") this.passoDaVisita(1);
      if (e.code === "ArrowLeft") this.passoDaVisita(-1);
    }
  }

  aoClicar(e) {
    if (this.camera.moveu > 6) return;
    if (this.modo === "andar" && document.pointerLockElement !== this.tela && !this.conversa) {
      this.tela.requestPointerLock?.();
      return;
    }
    const ponto = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    this.raio.setFromCamera(ponto, this.camera3d);
    const faces = this.quadros.map(function (q) { return q.malha; });
    const corpos = Object.values(this.equipe).map(function (a) { return a.robo.raiz; });
    const rotulos = this.rotulosDasSalas.visible ? this.rotulosDasSalas.children : [];
    const acertos = this.raio.intersectObjects(rotulos.concat(faces, corpos), true);
    if (!acertos.length) return;
    const objeto = acertos[0].object;
    if (objeto.userData.sala) { this.trocarModo("visita", objeto.userData.sala); return; }
    if (objeto.userData.quadro) { ampliarFigura(objeto.userData.quadro); return; }
    const agente = Object.values(this.equipe).find(function (a) {
      let no = objeto;
      while (no) { if (no === a.robo.raiz) return true; no = no.parent; }
      return false;
    });
    if (agente) this.falarCom(agente);
  }

  interagir(alvo) {
    if (alvo.agente) this.falarCom(alvo.agente);
    else ampliarFigura(alvo.quadro.dados);
  }

  // Conversa livre: a apresentacao do robo e os botoes de pergunta.
  falarCom(agente) {
    if (document.pointerLockElement) document.exitPointerLock();
    this.encerrarConversa();
    const sala = agente.sala;
    const visitante = this.camera3d.position;
    agente.conversar({ x: visitante.x, z: visitante.z });
    this.conversa = agente;
    mostrarFala(sala.robo, sala.nome, { texto: sala.fala, fonte: null });
    mostrarPerguntas(sala.perguntas, (p) => {
      mostrarFala(sala.robo, sala.nome, {
        texto: p.resposta, evidencias: p.evidencias, fonte: p.fonte, figura: p.figura,
      });
    });
    mostrarDica(null);
  }

  encerrarConversa() {
    if (this.conversa) this.conversa.encerrarConversa();
    this.conversa = null;
    fecharDialogo();
  }

  limparModo() {
    this.encerrarConversa();
    this.pararAutomatico();
    this.visita = null;
    this.caso = null;
    this.portador.robo.raiz.visible = false;
    this.camera.voo = null;
    this.estudio.fechar();
    mostrarProgresso(null);
    mostrarDica(null);
    $("controle-visita").hidden = true;
    if (document.pointerLockElement) document.exitPointerLock();
    this.mostrarTeto(true);
  }

  trocarModo(modo, sala) {
    if (modo === "caso") {
      abrirEscolhaDeCaso(DADOS.casos, (caso) => this.seguirCaso(caso));
      return;
    }
    this.limparModo();
    this.modo = modo;
    marcarModo(modo);
    if (modo === "planta") this.abrirPlanta();
    if (modo === "andar") this.comecarAPe();
    if (modo === "visita") this.comecarVisita(sala);
    if (modo === "estudio") this.estudio.abrir();
  }

  // Sem teto e com a camera alta: a maquete inteira de uma vez.
  abrirPlanta() {
    this.mostrarTeto(false);
    this.camera.raioMin = 8;
    const posicao = new THREE.Vector3(-7, 30, 31);
    const alvo = new THREE.Vector3(6.5, 0, 1);
    this.camera.voarEOrbitar(posicao, alvo, 1.8);
    mostrarDica("Arraste para girar, role para aproximar. Clique em um robô para conversar, ou em um quadro para ampliar.");
  }

  mostrarTeto(visivel) {
    this.predio.teto.visible = visivel;
    this.rotulosDasSalas.visible = !visivel;
    Object.values(this.equipe).forEach(function (a) { a.rotulo.visible = false; });
  }

  comecarAPe() {
    this.camera.entrarEmPrimeiraPessoa(-19, 0, -Math.PI / 2);
    mostrarDica("Clique na tela para olhar com o mouse. WASD ou setas para andar, Shift para correr, E para interagir.");
  }

  comecarVisita(sala) {
    const inicio = sala ? DADOS.salas.findIndex(function (s) { return s.id === sala; }) : 0;
    this.visita = { indice: inicio - 1, automatico: false };
    $("controle-visita").hidden = false;
    this.passoDaVisita(1);
  }

  passoDaVisita(direcao) {
    if (!this.visita) return;
    const proximo = this.visita.indice + direcao;
    if (proximo < 0 || proximo >= DADOS.salas.length) {
      if (proximo >= DADOS.salas.length) this.pararAutomatico();
      return;
    }
    this.encerrarConversa();
    this.visita.indice = proximo;
    const sala = DADOS.salas[proximo];
    const agente = this.equipe[sala.id];
    const pose = poseDaSala(sala.id);
    const posto = L.SALAS[sala.id].posto;
    mostrarProgresso(sala.nome + " · " + sala.fase, proximo + 1, DADOS.salas.length);
    agente.posicionar(posto);
    agente.conversar({ x: pose.posicao.x, z: pose.posicao.z });
    clearTimeout(this.relogioDaVisita);
    this.camera.voarEOrbitar(pose.posicao, pose.olhar, 2, () => {
      if (!this.visita || this.visita.indice !== proximo) return;
      this.conversa = agente;
      mostrarFala(sala.robo, sala.nome, { texto: sala.fala, fonte: null });
      mostrarPerguntas(sala.perguntas, (p) => {
        this.pararAutomatico();
        mostrarFala(sala.robo, sala.nome, { texto: p.resposta, evidencias: p.evidencias, fonte: p.fonte, figura: p.figura });
      });
      if (this.visita.automatico) {
        this.relogioDaVisita = setTimeout(() => this.passoDaVisita(1), L.tempoDeLeitura(sala.fala) * 1000);
      }
    });
  }

  alternarAutomatico() {
    if (!this.visita) return;
    this.visita.automatico = !this.visita.automatico;
    $("visita-auto").setAttribute("aria-pressed", String(this.visita.automatico));
    $("visita-auto").textContent = this.visita.automatico ? "Pausar" : "Reproduzir";
    clearTimeout(this.relogioDaVisita);
    if (this.visita.automatico) {
      // Deixa o publico terminar a sala atual antes de seguir.
      const sala = DADOS.salas[this.visita.indice];
      this.relogioDaVisita = setTimeout(() => this.passoDaVisita(1), L.tempoDeLeitura(sala.fala) * 1000);
    }
  }

  pararAutomatico() {
    clearTimeout(this.relogioDaVisita);
    if (this.visita) this.visita.automatico = false;
    $("visita-auto").setAttribute("aria-pressed", "false");
    $("visita-auto").textContent = "Reproduzir";
  }

  // Um caso real do teste percorre o hospital, parada a parada.
  seguirCaso(caso) {
    this.limparModo();
    this.modo = "caso";
    marcarModo("caso");
    this.mostrarTeto(false);
    this.rotulosDasSalas.visible = false;
    Object.values(this.equipe).forEach(function (a) { a.rotulo.visible = true; });
    this.camera.raioMin = 8;
    const etapas = L.etapasDoCaso(caso, DADOS.modelo);
    this.caso = { etapas: etapas, indice: -1, sala: "recepcao" };
    const portador = this.portador;
    portador.posicao.set(-20, 0, 0);
    portador.robo.raiz.visible = true;
    portador.rotina = [];
    $("controle-visita").hidden = true;
    this.proximaEtapa();
  }

  proximaEtapa() {
    const c = this.caso;
    if (!c) return;
    c.indice += 1;
    const etapa = c.etapas[c.indice];
    this.encerrarConversa();
    if (!etapa) {
      mostrarProgresso(null);
      this.trocarModo("planta");
      return;
    }
    mostrarProgresso("Caso: " + etapa.titulo, c.indice + 1, c.etapas.length);
    const parada = L.SALAS[etapa.sala].parada;
    const agente = this.equipe[etapa.sala];
    const caminho = L.rota(c.sala, etapa.sala, parada);
    c.sala = etapa.sala;
    agente.posicionar(L.SALAS[etapa.sala].posto);
    agente.conversar({ x: parada.x, z: parada.z });
    this.portador.irPara(caminho, () => {
      const sala = agente.sala;
      this.conversa = agente;
      agente.conversar({ x: parada.x, z: parada.z });
      mostrarFala(sala.robo, sala.nome, { texto: etapa.fala, evidencias: etapa.evidencias, fonte: etapa.fonte });
      const ultima = c.indice === c.etapas.length - 1;
      mostrarPerguntas([{ pergunta: ultima ? "Encerrar o caso" : "Seguir para a próxima sala" }], () => this.proximaEtapa());
    });
  }

  // Camera do caso: acompanha o portador de cima, sem brigar com o arraste do mouse.
  seguirPortador(dt) {
    if (!this.caso || this.camera.voo) return;
    const p = this.portador.posicao;
    const cam = this.camera;
    cam.modo = "orbita";
    const k = 1 - Math.exp(-dt * 3);
    cam.alvo.x += (p.x - cam.alvo.x) * k;
    cam.alvo.z += (p.z - cam.alvo.z) * k;
    cam.alvo.y = 0.6;
    cam.raio += (15 - cam.raio) * k * 0.4;
    cam.fi += (0.82 - cam.fi) * k * 0.4;
    cam.teta += anguloEntre(cam.teta, 0.5) * k * 0.2;
  }

  // Modo a pe: o alvo mais proximo na frente do visitante, para o aviso "E".
  procurarAlvo() {
    if (this.modo !== "andar" || this.conversa) {
      this.alvoProximo = null;
      $("mira-dica").hidden = true;
      return;
    }
    const cam = this.camera3d;
    const frente = new THREE.Vector3();
    cam.getWorldDirection(frente);
    let melhor = null;
    let distancia = 3.2;
    const considerar = function (posicao, alvo) {
      const v = posicao.clone().sub(cam.position);
      const d = v.length();
      if (d < distancia && v.normalize().dot(frente) > 0.55) {
        distancia = d;
        melhor = alvo;
      }
    };
    Object.values(this.equipe).forEach(function (a) {
      considerar(a.posicao.clone().setY(1.2), { agente: a, texto: "E  Falar com " + a.sala.robo.nome });
    });
    this.quadros.forEach(function (q) {
      considerar(q.posicao, { quadro: q, texto: "E  Ver " + q.dados.titulo });
    });
    this.alvoProximo = melhor;
    $("mira-dica").hidden = !melhor;
    if (melhor) $("mira-dica").textContent = melhor.texto;
  }

  quadroDoLaco() {
    this.avancar(Math.min(this.relogio.getDelta(), 0.05));
    this.desenhar();
  }

  desenhar() {
    if (this.estudio.aberto) this.estudio.desenhar();
    else this.pos.render(this.cena, this.camera3d);
  }

  avancar(dt) {
    this.tempo += dt;
    const t = this.tempo;
    Object.values(this.equipe).forEach(function (a) { a.atualizar(t, dt); });
    this.circulantes.forEach(function (a) { a.atualizar(t, dt); });
    if (this.portador.robo.raiz.visible) this.portador.atualizar(t, dt);
    this.animados.forEach(function (a) { a.avancar(t, dt); });
    this.ceu.avancar(t, dt);
    if (this.estudio.aberto) {
      this.diretor.avancar(dt);
    } else {
      this.seguirPortador(dt);
      this.camera.atualizar(dt, Boolean(this.conversa));
      this.procurarAlvo();
    }
    const pontos = Object.values(this.equipe).map(function (a) { return a.posicao; });
    this.circulantes.forEach(function (a) { pontos.push(a.posicao); });
    pontos.push(this.portador.posicao, this.camera3d.position);
    atualizarPortas(this.predio.portas, pontos, dt);
  }

  iniciar() {
    this.renderizador.setAnimationLoop(() => this.quadroDoLaco());
    $("carregando").hidden = true;
    const pedida = new URLSearchParams(location.search).get("modo");
    this.trocarModo(["andar", "visita", "estudio"].includes(pedida) ? pedida : "planta");
  }
}

function suportaWebGL() {
  try {
    return Boolean(document.createElement("canvas").getContext("webgl2"));
  } catch (erro) {
    return false;
  }
}

if (suportaWebGL()) {
  const hospital = new Hospital();
  window.hospital = hospital;
  hospital.iniciar();
} else {
  $("carregando").textContent = "Este navegador não oferece WebGL 2, necessário para o hospital 3D. A vitrine do projeto mostra o mesmo conteúdo sem 3D.";
}
