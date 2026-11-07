// Estudio: ensaiar, gravar e exportar os filmes, tudo no navegador.
//
// O que aparece no quadro e exatamente o que vai para o arquivo. Teleprompter e botoes
// ficam fora do quadro, entao quem grava le o texto sem que ele saia no video.

const TIPOS_DE_VIDEO = [
  "video/mp4;codecs=avc1.640028,mp4a.40.2",
  "video/webm;codecs=vp9,opus",
  "video/webm",
];
const BITS_DE_VIDEO = 16000000;
const VOLUME_COM_VOZ = 0.3;

function baixarArquivo(nome, conteudo, tipo) {
  const url = URL.createObjectURL(conteudo instanceof Blob ? conteudo : new Blob([conteudo], { type: tipo }));
  const link = document.createElement("a");
  link.href = url;
  link.download = nome;
  link.click();
  setTimeout(function () { URL.revokeObjectURL(url); }, 20000);
}

class Estudio {
  constructor(hospital) {
    this.h = hospital;
    this.tela = $("quadro");
    this.c = this.tela.getContext("2d");
    this.diretor = hospital.diretor;
    this.aberto = false;
    this.filme = DADOS.filmes[0];
    this.medidas = {};
    this.inicio = 0;
    this.gravador = null;
    this.audio = null;
    this.microfone = null;
    this.mestre = null;
    this.ligar();
  }

  ligar() {
    const d = this.diretor;
    d.aoMudarPlano = (i) => this.mostrarPlano(i);
    d.aoTerminar = () => this.terminar();
    document.querySelectorAll("[data-filme]").forEach((botao) => {
      botao.addEventListener("click", () => this.escolher(botao.dataset.filme));
    });
    $("estudio-ensaiar").addEventListener("click", () => this.ensaiar());
    $("estudio-gravar").addEventListener("click", () => this.gravar());
    $("estudio-parar").addEventListener("click", () => this.interromper());
    $("estudio-legendas").addEventListener("change", (e) => { d.legendas = e.target.checked; });
    $("estudio-foto").addEventListener("click", () => this.tela.toBlob((blob) => baixarArquivo("hospital-" + this.filme.id + ".png", blob)));
    $("estudio-srt").addEventListener("click", () => baixarArquivo("hospital-" + this.filme.id + ".srt", R.srt(d.linha), "text/plain"));
    $("estudio-roteiro").addEventListener("click", () => {
      const texto = R.roteiroEmTexto(this.filme, d.linha) + "\nCapítulos\n" + R.capitulos(d.linha) + "\n";
      baixarArquivo("hospital-" + this.filme.id + "-roteiro.txt", texto, "text/plain");
    });
    $("estudio-video").addEventListener("change", (e) => this.carregarVideo(e.target.files[0]));
  }

  abrir() {
    this.aberto = true;
    this.h.renderizador.setSize(LARGURA, ALTURA, false);
    this.h.pos.redimensionar(LARGURA, ALTURA);
    this.h.camera3d.aspect = LARGURA / ALTURA;
    this.h.camera3d.updateProjectionMatrix();
    $("estudio").hidden = false;
    this.escolher(this.filme.id);
  }

  fechar() {
    if (!this.aberto) return;
    this.interromper();
    this.aberto = false;
    $("estudio").hidden = true;
    this.h.pos.tilt = 0;
    this.h.camera3d.fov = 52;
    this.h.restaurarMundo();
    this.h.ajustarTamanho();
  }

  escolher(id) {
    this.filme = DADOS.filmes.find(function (f) { return f.id === id; });
    this.filme.planos.forEach(function (p) { if (p.figura) imagemDaFigura(p.figura); });
    this.inicio = 0;
    this.medidas = {};
    this.diretor.carregar(this.filme, this.medidas);
    document.querySelectorAll("[data-filme]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.filme === id)));
    $("estudio-telao").hidden = !this.filme.planos.some(function (p) { return p.telao; });
    this.desenharLinha();
    this.mostrarPlano(0);
  }

  // Fita do filme: um trecho por plano, com largura proporcional ao tempo.
  desenharLinha() {
    const fita = $("fita");
    const linha = this.diretor.linha;
    fita.replaceChildren();
    linha.planos.forEach((plano) => {
      const botao = document.createElement("button");
      botao.type = "button";
      botao.className = "trecho";
      botao.style.flexGrow = String(plano.duracao);
      botao.title = "Começar deste plano: " + plano.capitulo;
      botao.textContent = plano.capitulo;
      botao.addEventListener("click", () => {
        this.inicio = plano.indice;
        this.mostrarPlano(plano.indice);
      });
      fita.append(botao);
    });
    $("estudio-duracao").textContent = R.tempoCurto(linha.total);
  }

  mostrarPlano(indice) {
    const planos = this.diretor.linha.planos;
    const plano = planos[indice];
    if (!plano) return;
    $("prompter-agora").textContent = plano.narracao;
    $("prompter-depois").textContent = planos[indice + 1] ? "Depois: " + planos[indice + 1].narracao : "Depois: créditos.";
    $("fita").querySelectorAll(".trecho").forEach(function (b, i) { b.classList.toggle("ativo", i === indice); });
  }

  async prepararAudio() {
    if (!this.audio) {
      this.audio = new AudioContext();
      this.saida = this.audio.createMediaStreamDestination();
      this.alto = this.audio.createGain();
      this.alto.connect(this.audio.destination);
      this.alto.connect(this.saida);
    }
    await this.audio.resume();
    const querVoz = $("estudio-microfone").checked;
    if (querVoz && !this.microfone) {
      try {
        const fluxo = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
        this.microfone = this.audio.createMediaStreamSource(fluxo);
      } catch (erro) {
        $("estudio-estado").textContent = "Microfone indisponível: gravando sem voz.";
        $("estudio-microfone").checked = false;
      }
    }
    // A voz vai so para o arquivo: nas caixas de som, ela voltaria ao microfone.
    if (this.microfone) {
      this.microfone.disconnect();
      if ($("estudio-microfone").checked) this.microfone.connect(this.saida);
    }
  }

  tocarTrilha() {
    this.pararTrilha();
    if (!$("estudio-musica").checked) return;
    const d = this.diretor;
    const desde = this.inicio > 0 ? d.linha.planos[this.inicio].inicio : 0;
    this.alto.gain.value = $("estudio-microfone").checked ? VOLUME_COM_VOZ : 1;
    this.mestre = agendarTrilha(this.audio, this.alto, d.linha, this.filme, desde);
  }

  pararTrilha() {
    if (!this.mestre) return;
    this.mestre.disconnect();
    this.mestre = null;
  }

  async ensaiar() {
    await this.prepararAudio();
    this.tocarTrilha();
    this.diretor.iniciar(this.inicio);
    this.marcarEstado("Ensaiando. Nada está sendo gravado.");
  }

  async gravar() {
    await this.prepararAudio();
    const tipo = TIPOS_DE_VIDEO.find(function (t) { return MediaRecorder.isTypeSupported(t); });
    const fluxo = this.tela.captureStream(30);
    this.saida.stream.getAudioTracks().forEach(function (trilha) { fluxo.addTrack(trilha); });
    const pedacos = [];
    this.gravador = new MediaRecorder(fluxo, { mimeType: tipo, videoBitsPerSecond: BITS_DE_VIDEO, audioBitsPerSecond: 192000 });
    this.gravador.ondataavailable = function (e) { if (e.data.size) pedacos.push(e.data); };
    const plano = this.diretor.linha.planos[this.inicio];
    this.gravador.onstop = () => {
      const extensao = tipo.startsWith("video/mp4") ? "mp4" : "webm";
      const sufixo = this.inicio > 0 ? "-desde-" + plano.id : "";
      baixarArquivo("hospital-" + this.filme.id + sufixo + "." + extensao, new Blob(pedacos, { type: tipo }));
      $("estudio-estado").textContent = "Gravação salva na pasta de downloads.";
    };
    this.gravador.start(1000);
    this.tocarTrilha();
    this.diretor.iniciar(this.inicio);
    this.marcarEstado("Gravando");
    document.body.classList.add("gravando");
  }

  interromper() {
    this.diretor.parar();
    this.pararTrilha();
    if (this.diretor.video) this.diretor.video.pause();
    this.pararGravador();
    this.marcarEstado("Parado.");
    if (this.aberto) this.diretor.carregar(this.filme, this.medidas);
  }

  terminar() {
    this.pararTrilha();
    this.pararGravador();
    this.marcarEstado("Filme concluído.");
  }

  pararGravador() {
    if (this.gravador && this.gravador.state !== "inactive") this.gravador.stop();
    this.gravador = null;
    document.body.classList.remove("gravando");
  }

  marcarEstado(texto) {
    $("estudio-estado").textContent = texto;
    const ativo = this.diretor.rodando;
    $("estudio-ensaiar").disabled = ativo;
    $("estudio-gravar").disabled = ativo;
    $("estudio-parar").disabled = !ativo;
  }

  // Gravacao de tela real: o plano do telao passa a durar o video inteiro.
  carregarVideo(arquivo) {
    if (!arquivo) return;
    const video = document.createElement("video");
    video.src = URL.createObjectURL(arquivo);
    video.muted = true;
    video.playsInline = true;
    video.addEventListener("loadedmetadata", () => {
      const plano = this.filme.planos.find(function (p) { return p.telao; });
      this.medidas[plano.id] = video.duration / 0.62;
      this.diretor.video = video;
      this.diretor.carregar(this.filme, this.medidas);
      this.desenharLinha();
      $("estudio-estado").textContent = "Vídeo real carregado: " + R.tempoCurto(video.duration) + ".";
    });
  }

  // Um quadro completo: cena 3D e letreiros, juntos no canvas que vira video.
  desenhar() {
    this.h.pos.render(this.h.cena, this.h.camera3d);
    const c = this.c;
    c.drawImage(this.h.renderizador.domElement, 0, 0, LARGURA, ALTURA);
    desenharLetreiros(c, this.diretor.quadro());
    if (this.diretor.rodando) {
      $("estudio-relogio").textContent = R.tempoCurto(this.diretor.t) + " / " + R.tempoCurto(this.diretor.linha.total);
      const trecho = this.diretor.quadro().legenda;
      if (trecho) $("prompter-frase").textContent = trecho.texto;
    }
  }

  // Trilha completa em WAV, para quem monta o video fora do navegador.
  async trilhaEmWav() {
    const linha = this.diretor.linha;
    const contexto = new OfflineAudioContext(2, Math.ceil(linha.total * 44100), 44100);
    agendarTrilha(contexto, contexto.destination, linha, this.filme, 0);
    return audioEmWav(await contexto.startRendering());
  }
}
