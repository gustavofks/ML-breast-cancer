// Texturas desenhadas em canvas: nenhum arquivo de imagem alem das figuras do projeto.

function criarCanvas(largura, altura) {
  const canvas = document.createElement("canvas");
  canvas.width = largura;
  canvas.height = altura;
  return [canvas, canvas.getContext("2d")];
}

function texturaDe(canvas, repetir) {
  const textura = new THREE.CanvasTexture(canvas);
  textura.colorSpace = THREE.SRGBColorSpace;
  textura.anisotropy = 8;
  if (repetir) {
    textura.wrapS = textura.wrapT = THREE.RepeatWrapping;
    textura.repeat.set(repetir[0], repetir[1]);
  }
  return textura;
}

// Granilite: base clara com lascas. Pseudoaleatorio fixo para o chao nao mudar a cada carga.
function texturaGranilite(base, repetir) {
  const [canvas, c] = criarCanvas(512, 512);
  c.fillStyle = base;
  c.fillRect(0, 0, 512, 512);
  let semente = 42;
  const sorte = function () { semente = (semente * 16807) % 2147483647; return semente / 2147483647; };
  const lascas = ["#c9c2b4", "#d9d3c6", "#b8b1a2", "#efeae0", "#a9c4bd"];
  for (let i = 0; i < 2600; i++) {
    c.fillStyle = lascas[Math.floor(sorte() * lascas.length)];
    c.globalAlpha = 0.25 + sorte() * 0.4;
    const r = 0.7 + sorte() * 2.6;
    c.beginPath();
    c.ellipse(sorte() * 512, sorte() * 512, r, r * (0.5 + sorte() * 0.5), sorte() * 3, 0, 7);
    c.fill();
  }
  c.globalAlpha = 0.16;
  c.strokeStyle = "#8d8778";
  c.lineWidth = 2;
  c.strokeRect(0, 0, 512, 512);
  c.globalAlpha = 1;
  return texturaDe(canvas, repetir);
}

function texturaGrama(repetir) {
  const [canvas, c] = criarCanvas(256, 256);
  c.fillStyle = "#9fbf8c";
  c.fillRect(0, 0, 256, 256);
  let semente = 7;
  const sorte = function () { semente = (semente * 16807) % 2147483647; return semente / 2147483647; };
  for (let i = 0; i < 1800; i++) {
    c.fillStyle = sorte() > 0.5 ? "#92b57f" : "#abc998";
    c.fillRect(sorte() * 256, sorte() * 256, 2, 3);
  }
  return texturaDe(canvas, repetir);
}

// Placa de sala: faixa da cor da ala e texto nitido, desenhado em 2x.
function texturaPlaca(texto, subtitulo, cor) {
  const [canvas, c] = criarCanvas(1024, 256);
  c.fillStyle = "#fbfaf6";
  c.fillRect(0, 0, 1024, 256);
  c.fillStyle = cor;
  c.fillRect(0, 0, 26, 256);
  c.fillStyle = "#1e2a2e";
  c.font = "600 84px 'Segoe UI', system-ui, sans-serif";
  c.textBaseline = "middle";
  c.fillText(texto, 62, subtitulo ? 96 : 128, 930);
  if (subtitulo) {
    c.fillStyle = "#5b6a6e";
    c.font = "500 46px 'Segoe UI', system-ui, sans-serif";
    c.fillText(subtitulo, 62, 186, 930);
  }
  return texturaDe(canvas);
}

function texturaRotulo(texto, cor) {
  const [canvas, c] = criarCanvas(512, 128);
  c.font = "600 54px 'Segoe UI', system-ui, sans-serif";
  const largura = Math.min(c.measureText(texto).width + 56, 500);
  const x = (512 - largura) / 2;
  c.fillStyle = "rgba(20, 28, 32, .82)";
  c.beginPath();
  c.roundRect(x, 18, largura, 92, 46);
  c.fill();
  c.fillStyle = cor || "#ffffff";
  c.textAlign = "center";
  c.textBaseline = "middle";
  c.fillText(texto, 256, 66, 460);
  return texturaDe(canvas);
}

// Tela de monitor: barras e linhas na cor do posto, em vez de um retangulo liso.
function texturaTela(cor) {
  if (texturaTela.cache[cor]) return texturaTela.cache[cor];
  const [canvas, c] = criarCanvas(256, 148);
  c.fillStyle = "#0f1b20";
  c.fillRect(0, 0, 256, 148);
  c.fillStyle = cor;
  c.globalAlpha = 0.9;
  c.fillRect(0, 0, 256, 14);
  const alturas = [34, 58, 46, 82, 70, 96, 64];
  alturas.forEach(function (h, i) { c.fillRect(14 + i * 19, 136 - h, 12, h); });
  c.globalAlpha = 0.55;
  for (let i = 0; i < 7; i++) c.fillRect(158, 30 + i * 15, 40 + ((i * 29) % 50), 5);
  c.globalAlpha = 1;
  texturaTela.cache[cor] = texturaDe(canvas);
  return texturaTela.cache[cor];
}
texturaTela.cache = {};

// Mural da recepcao: as tres fases, na cor da linha do chao que leva a cada ala.
function texturaMural(fases) {
  const [canvas, c] = criarCanvas(2048, 640);
  c.fillStyle = "#fbfaf6";
  c.fillRect(0, 0, 2048, 640);
  c.fillStyle = "#1e2a2e";
  c.font = "600 74px 'Segoe UI', system-ui, sans-serif";
  c.textBaseline = "middle";
  c.fillText("Um hospital, três fases", 64, 92);
  fases.forEach(function (fase, i) {
    const x = 64 + i * 650;
    c.fillStyle = fase.cor;
    c.fillRect(x, 190, 600, 16);
    c.fillStyle = "#1e2a2e";
    c.font = "600 60px 'Segoe UI', system-ui, sans-serif";
    c.fillText(fase.titulo, x, 270);
    c.fillStyle = "#5b6a6e";
    c.font = "500 42px 'Segoe UI', system-ui, sans-serif";
    fase.linhas.forEach(function (linha, j) { c.fillText(linha, x, 350 + j * 62); });
  });
  return texturaDe(canvas);
}

function texturaFaixaDeObra() {
  const [canvas, c] = criarCanvas(256, 32);
  for (let i = -1; i < 9; i++) {
    c.fillStyle = i % 2 ? "#f2c230" : "#23272a";
    c.beginPath();
    c.moveTo(i * 32, 32);
    c.lineTo(i * 32 + 32, 32);
    c.lineTo(i * 32 + 48, 0);
    c.lineTo(i * 32 + 16, 0);
    c.fill();
  }
  return texturaDe(canvas, [4, 1]);
}

// Paredao de log da central: a sessao gravada da API, rolando sem parar.
function criarPainelDeLog(log) {
  const [canvas, c] = criarCanvas(1024, 512);
  const textura = texturaDe(canvas);
  const linhas = log.filter(function (e) { return e.rota; });
  const altura = 40;
  let deslocamento = 0;

  function desenhar() {
    c.fillStyle = "#0d1a1f";
    c.fillRect(0, 0, 1024, 512);
    c.fillStyle = "#17313a";
    c.fillRect(0, 0, 1024, 52);
    c.fillStyle = "#7fe3cf";
    c.font = "600 26px Consolas, monospace";
    c.textBaseline = "middle";
    c.fillText("SESSÃO GRAVADA DA API", 24, 27);
    c.font = "24px Consolas, monospace";
    const primeira = Math.floor(deslocamento);
    for (let i = 0; i < 12; i++) {
      const e = linhas[(primeira + i) % linhas.length];
      const y = 78 + i * altura - (deslocamento - primeira) * altura;
      if (y < 60) continue;
      c.fillStyle = e.status >= 400 ? "#ffb26b" : "#7fe3cf";
      c.fillText(String(e.status), 24, y);
      c.fillStyle = "#cfe5ea";
      c.fillText((e.metodo || "").padEnd(5) + (e.rota || ""), 100, y);
      c.fillStyle = "#7f9aa3";
      c.fillText(e.latencia_ms + " ms", 800, y);
    }
    textura.needsUpdate = true;
  }

  desenhar();
  return {
    textura: textura,
    avancar: function (dt) { deslocamento += dt * 0.7; desenhar(); },
  };
}
