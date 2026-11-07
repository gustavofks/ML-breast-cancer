// Diretor: executa um roteiro no tempo, com camera, robos e cenario.
//
// O filme so anda para a frente. Para comecar de um plano no meio, as acoes dos planos
// anteriores sao aplicadas ja concluidas, e o cenario fica como estaria naquele ponto.

const ROTA_DO_PORTADOR = /^portador:(\w+):(\w+)$/;

// Cada acao recebe o hospital, o plano e a fracao `p` dele, de 0 a 1.
const ACOES = {
  pastas: function (h, plano, p) {
    const palco = h.palco;
    palco.pastas.posicionar(Math.round(palco.pastas.total * R.suave(R.janela(p, 0.08, 0.72))), 0);
  },
  trancar: function (h, plano, p) {
    const palco = h.palco;
    palco.pastas.posicionar(palco.pastas.total, R.janela(p, 0.08, 0.8));
    palco.cofre.abrir(1 - R.suave(R.janela(p, 0.82, 0.96)));
  },
  podio: function (h, plano, p) {
    h.palco.podio.subir(R.janela(p, 0.1, 0.75));
  },
  enxame: function (h, plano, p) {
    h.palco.enxame.estado.relogio = R.janela(p, 0.04, 0.9);
  },
  regua: function (h, plano, p) {
    const inicial = 0.5;
    const servido = DADOS.modelo.limiar;
    // Em escala de log-chance, o corte anda em velocidade constante na tela.
    const logito = function (v) { return Math.log(v / (1 - v)); };
    const s = R.suave(R.janela(p, 0.5, 0.78));
    const l = logito(inicial) + (logito(servido) - logito(inicial)) * s;
    h.palco.regua.definir(1 / (1 + Math.exp(-l)));
  },
  selos: function (h, plano, p) {
    h.palco.selos.acender(R.janela(p, 0.25, 0.7));
  },
  fluxo: function (h, plano, p) {
    h.palco.fluxo.acender(Math.ceil(R.janela(p, 0.1, 0.8) * h.palco.fluxo.total));
  },
  fluxo_completo: function (h) {
    h.palco.fluxo.acender(h.palco.fluxo.total);
  },
  guindaste: function () {},
};

function executarAcao(h, nome, plano, p) {
  if (nome.startsWith("construir:")) {
    h.obra.definir(nome.split(":")[1], R.janela(p, 0.05, 0.95));
    h.sincronizarObra();
  } else if (!ROTA_DO_PORTADOR.test(nome)) {
    ACOES[nome](h, plano, p);
  }
}

class Diretor {
  constructor(hospital) {
    this.h = hospital;
    this.filme = null;
    this.linha = null;
    this.t = 0;
    this.indice = -2;
    this.rodando = false;
    this.legendas = true;
    this.video = null;
    this.aoTerminar = null;
    this.aoMudarPlano = null;
  }

  carregar(filme, medidas) {
    this.filme = filme;
    this.linha = R.linhaDoTempo(filme, medidas);
    this.rodando = false;
    this.t = 0;
    this.indice = -2;
    this.prepararMundo();
    this.definirCamera("terreno", 0);
  }

  // O mundo como ele estava antes deste filme: so as alas das fases anteriores.
  prepararMundo() {
    const h = this.h;
    const primeira = this.filme.id === "fase1";
    Object.keys(h.obra.pecas).forEach((ala) => {
      h.obra.definir(ala, this.filme.alas_prontas.includes(ala) ? 1 : 0);
    });
    h.sincronizarObra();
    h.palco.pastas.posicionar(primeira ? 0 : h.palco.pastas.total, primeira ? 0 : 1);
    h.palco.cofre.abrir(primeira ? 1 : 0);
    h.palco.podio.subir(primeira ? 0 : 1);
    h.palco.selos.acender(this.filme.id === "fase2" ? 0 : 1);
    h.palco.regua.definir(this.filme.id === "fase2" ? 0.5 : DADOS.modelo.limiar);
    h.palco.enxame.estado.relogio = this.filme.id === "fase2" ? 0 : null;
    h.palco.fluxo.acender(0);
    h.portador.robo.raiz.visible = false;
    h.ceu.definirHora(this.filme.hora[0]);
  }

  iniciar(desde) {
    const k = desde || 0;
    this.prepararMundo();
    this.linha.planos.slice(0, k).forEach((plano) => {
      plano.acoes.forEach((nome) => executarAcao(this.h, nome, plano, 1));
    });
    this.t = k > 0 ? this.linha.planos[k].inicio : 0;
    this.indice = -2;
    this.rodando = true;
  }

  parar() {
    this.rodando = false;
  }

  definirCamera(nome, p) {
    const pose = poseDaTomada(nome, p, this.h.portador);
    const camera = this.h.camera3d;
    if (camera.fov !== pose.fov) {
      camera.fov = pose.fov;
      camera.updateProjectionMatrix();
    }
    camera.position.copy(pose.posicao);
    camera.lookAt(pose.olhar);
    const tomada = TOMADAS[nome];
    this.h.pos.tilt = tomada.tilt || 0;
    this.h.definirTetoDoFilme(Boolean(tomada.semTeto));
  }

  entrarNoPlano(indice) {
    const h = this.h;
    if (this.salaFalando) {
      this.salaFalando.encerrarConversa();
      this.salaFalando = null;
    }
    const plano = this.linha.planos[indice];
    if (!plano) return;
    if (plano.robo) {
      const agente = h.equipe[plano.robo];
      const pose = poseDaTomada(plano.tomada, 0.5, h.portador);
      agente.posicionar(L.SALAS[plano.robo].posto);
      agente.conversar({ x: pose.posicao.x, z: pose.posicao.z });
      this.salaFalando = agente;
    }
    plano.acoes.forEach((nome) => {
      const rota = nome.match(ROTA_DO_PORTADOR);
      if (rota) this.soltarPortador(rota[1], rota[2], plano);
    });
    if (plano.telao && this.video) {
      this.video.currentTime = 0;
      this.video.play();
    }
    if (this.aoMudarPlano) this.aoMudarPlano(indice);
  }

  // Velocidade ajustada para o caso chegar antes de o plano acabar.
  soltarPortador(de, para, plano) {
    const portador = this.h.portador;
    const inicio = L.SALAS[de].parada;
    const fim = L.SALAS[para].parada;
    const caminho = L.rota(de, para, fim);
    let distancia = 0;
    let anterior = inicio;
    caminho.forEach(function (ponto) {
      distancia += Math.hypot(ponto.x - anterior.x, ponto.z - anterior.z);
      anterior = ponto;
    });
    portador.posicionar(inicio);
    portador.robo.raiz.visible = true;
    portador.velocidade = distancia / (plano.duracao * 0.82);
    portador.irPara(caminho, function () { portador.estado = "parado"; });
  }

  avancar(dt) {
    if (!this.rodando) return;
    this.t += dt;
    const linha = this.linha;
    if (this.t >= linha.total) {
      this.rodando = false;
      if (this.aoTerminar) this.aoTerminar();
      return;
    }
    const indice = R.indiceEm(linha, this.t);
    if (indice !== this.indice) {
      this.indice = indice;
      this.entrarNoPlano(indice);
    }
    const proporcao = this.t / linha.total;
    this.h.ceu.definirHora(this.filme.hora[0] + (this.filme.hora[1] - this.filme.hora[0]) * proporcao);
    const plano = linha.planos[indice];
    if (plano) {
      const p = (this.t - plano.inicio) / plano.duracao;
      plano.acoes.forEach((nome) => executarAcao(this.h, nome, plano, p));
      this.definirCamera(plano.tomada, p);
    } else if (indice >= linha.planos.length) {
      const ultimo = linha.planos[linha.planos.length - 1];
      this.definirCamera(ultimo.tomada, 1 + (this.t - linha.creditos) / 40);
    }
  }

  // Estado de um quadro, para os letreiros. `escuro` faz o mergulho entre planos.
  quadro() {
    const linha = this.linha;
    const t = this.t;
    const indice = this.rodando || t > 0 ? R.indiceEm(linha, t) : -1;
    const plano = linha.planos[indice] || null;
    const estado = {
      t: t, indice: indice, plano: plano, linha: linha, filme: this.filme, legendas: this.legendas,
      video: this.video, escuro: 0, robo: null, p: 0, tPlano: 0, legenda: null,
    };
    if (plano) {
      estado.tPlano = t - plano.inicio;
      estado.p = estado.tPlano / plano.duracao;
      estado.legenda = R.legendaEm(plano, t);
      if (plano.robo) estado.robo = this.h.equipe[plano.robo].sala.robo;
      const entrada = 1 - R.janela(estado.tPlano, 0, 0.28);
      const saida = R.janela(estado.tPlano, plano.duracao - 0.22, plano.duracao);
      estado.escuro = Math.max(entrada, saida);
    }
    if (indice < 0) estado.escuro = R.janela(t, R.CLAQUETE - 0.4, R.CLAQUETE);
    estado.escuro = Math.max(estado.escuro, R.janela(t, linha.total - 1.6, linha.total));
    return estado;
  }
}
