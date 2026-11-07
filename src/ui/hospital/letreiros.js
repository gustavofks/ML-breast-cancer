// Tudo que aparece escrito no filme, desenhado no proprio quadro de 1920 x 1080.
//
// Fica no canvas, nao no DOM, para sair no arquivo gravado. O teleprompter e os botoes do
// estudio ficam de fora do quadro: quem grava le, e quem assiste nao ve.

const FONTE = "'Segoe UI', system-ui, -apple-system, Roboto, sans-serif";
const LARGURA = 1920;
const ALTURA = 1080;
const IMAGENS = {};

function imagemDaFigura(nome) {
  if (!IMAGENS[nome]) {
    IMAGENS[nome] = new Image();
    IMAGENS[nome].src = FIGURAS[nome];
  }
  return IMAGENS[nome];
}

function caixaRedonda(c, x, y, w, h, r, cor) {
  c.fillStyle = cor;
  c.beginPath();
  c.roundRect(x, y, w, h, r);
  c.fill();
}

function quebrar(c, texto, largura) {
  const linhas = [];
  let linha = "";
  texto.split(/\s+/).forEach(function (palavra) {
    const tentativa = linha ? linha + " " + palavra : palavra;
    if (c.measureText(tentativa).width > largura && linha) {
      linhas.push(linha);
      linha = palavra;
    } else {
      linha = tentativa;
    }
  });
  if (linha) linhas.push(linha);
  return linhas;
}

function comOpacidade(c, a, desenho) {
  if (a <= 0) return;
  c.save();
  c.globalAlpha = a;
  desenho();
  c.restore();
}

function desenharClaquete(c, q) {
  c.fillStyle = "#0c1216";
  c.fillRect(0, 0, LARGURA, ALTURA);
  const x = 560;
  const y = 400;
  const w = 800;
  const h = 330;
  caixaRedonda(c, x, y, w, h, 14, "#171f24");
  c.fillStyle = "#f2f5f4";
  c.textAlign = "left";
  c.textBaseline = "middle";
  c.font = "700 76px " + FONTE;
  c.fillText(q.filme.selo, x + 44, y + 92);
  c.fillStyle = "#b5c2c3";
  c.font = "500 44px " + FONTE;
  c.fillText(q.filme.titulo, x + 44, y + 170);
  c.font = "500 30px " + FONTE;
  c.fillStyle = "#7f9095";
  c.fillText("Hospital do Tech Challenge", x + 44, y + 262);
  c.textAlign = "right";
  c.fillText("Tomada 1", x + w - 44, y + 262);

  // A haste fecha aos 1,7 s, junto com o estalo da trilha.
  const fechando = R.janela(q.t, 1.05, 1.7);
  const angulo = -0.42 * (1 - fechando * fechando);
  c.save();
  c.translate(x + 8, y - 8);
  c.rotate(angulo);
  c.fillStyle = "#f2f5f4";
  c.fillRect(0, -66, w - 8, 66);
  c.fillStyle = "#17202a";
  for (let i = -1; i < 12; i++) {
    c.beginPath();
    c.moveTo(i * 76, 0);
    c.lineTo(i * 76 + 38, 0);
    c.lineTo(i * 76 + 76, -66);
    c.lineTo(i * 76 + 38, -66);
    c.fill();
  }
  c.restore();
}

function desenharTitulo(c, q) {
  const a = R.janela(q.tPlano, 0.6, 1.6) * (1 - R.janela(q.tPlano, 6.2, 7.2));
  comOpacidade(c, a, function () {
    const sobe = (1 - R.suave(R.janela(q.tPlano, 0.6, 1.8))) * 26;
    c.textAlign = "left";
    c.textBaseline = "alphabetic";
    c.shadowColor = "rgba(0, 0, 0, .55)";
    c.shadowBlur = 26;
    c.fillStyle = "#ffffff";
    c.font = "500 40px " + FONTE;
    c.fillText(q.filme.selo.toUpperCase(), 110, 400 + sobe);
    c.font = "700 104px " + FONTE;
    c.fillText(q.filme.titulo, 106, 510 + sobe);
  });
}

function desenharCapitulo(c, q) {
  const coberto = q.plano.figura || q.plano.telao ? R.janela(q.p, 0.2, 0.3) : 0;
  const a = R.janela(q.tPlano, 0.25, 0.7) * (1 - R.janela(q.tPlano, 4, 4.6)) * (1 - coberto);
  comOpacidade(c, a, function () {
    c.font = "600 28px " + FONTE;
    const texto = q.plano.capitulo;
    const w = c.measureText(texto).width;
    caixaRedonda(c, 64, 104, w + 96, 56, 12, "rgba(12, 18, 22, .78)");
    c.fillStyle = "#e0528c";
    c.textAlign = "left";
    c.textBaseline = "middle";
    c.fillText(String(q.indice + 1).padStart(2, "0"), 84, 133);
    c.fillStyle = "#f2f5f4";
    c.fillText(texto, 136, 133);
  });
}

function desenharNomeDoRobo(c, q) {
  if (!q.robo) return;
  const a = R.janela(q.tPlano, 0.5, 1) * (1 - R.janela(q.tPlano, 5.2, 5.8));
  comOpacidade(c, a, function () {
    const entra = (1 - R.suave(R.janela(q.tPlano, 0.5, 1.1))) * -40;
    c.font = "700 40px " + FONTE;
    const w = Math.max(c.measureText(q.robo.nome).width, 260) + 64;
    caixaRedonda(c, 64 + entra, 730, w, 106, 12, "rgba(12, 18, 22, .84)");
    c.fillStyle = q.robo.cor;
    c.fillRect(64 + entra, 744, 7, 78);
    c.fillStyle = "#ffffff";
    c.textAlign = "left";
    c.textBaseline = "middle";
    c.fillText(q.robo.nome, 94 + entra, 765);
    c.fillStyle = "#b5c2c3";
    c.font = "500 26px " + FONTE;
    c.fillText(q.robo.papel, 94 + entra, 806);
  });
}

function textoDaEvidencia(e, p) {
  const atual = e.numero * p;
  const corpo = e.casas ? L.numero(atual, e.casas) : String(Math.round(atual));
  return (e.prefixo || "") + corpo + (e.sufixo || "");
}

// Numeros que sobem ate o valor final: a narracao e o quadro dizem a mesma coisa.
function desenharContadores(c, q) {
  const lista = q.plano.evidencias;
  if (!lista.length) return;
  const altura = 122;
  const topo = 300 - (lista.length - 2) * 40;
  const saida = 1 - R.janela(q.p, 0.94, 1);
  lista.forEach(function (e, i) {
    const inicio = 0.8 + i * 0.55;
    const a = R.janela(q.tPlano, inicio, inicio + 0.4) * saida;
    comOpacidade(c, a, function () {
      const y = topo + i * (altura + 14);
      const p = R.suave(R.janela(q.tPlano, inicio, inicio + 1.3));
      caixaRedonda(c, 1436, y, 420, altura, 14, "rgba(12, 18, 22, .84)");
      c.textAlign = "left";
      c.textBaseline = "middle";
      c.fillStyle = "#ffffff";
      c.font = "700 52px " + FONTE;
      c.fillText(textoDaEvidencia(e, p), 1462, y + 48, 370);
      c.fillStyle = "#b5c2c3";
      c.font = "500 25px " + FONTE;
      c.fillText(e.rotulo, 1462, y + 94, 370);
    });
  });
}

// A camera entrou na tela: a figura real toma o quadro.
function desenharFigura(c, q) {
  const a = R.janela(q.p, 0.2, 0.3);
  if (!q.plano.figura || a <= 0) return;
  const imagem = imagemDaFigura(q.plano.figura);
  comOpacidade(c, a, function () {
    c.fillStyle = "#0c1216";
    c.fillRect(0, 0, LARGURA, ALTURA);
    const topo = 124;
    const altura = 700;
    // Com contadores ao lado, a figura cede a coluna da direita para eles.
    const largura = q.plano.evidencias.length ? 1270 : 1680;
    caixaRedonda(c, 120, topo, largura, altura + 28, 16, "#ffffff");
    if (imagem.complete && imagem.naturalWidth) {
      const escala = Math.min((largura - 60) / imagem.naturalWidth, altura / imagem.naturalHeight);
      const w = imagem.naturalWidth * escala;
      const h = imagem.naturalHeight * escala;
      c.drawImage(imagem, 120 + (largura - w) / 2, topo + 14 + (altura - h) / 2, w, h);
    }
    c.fillStyle = "#b5c2c3";
    c.font = "500 28px " + FONTE;
    c.textAlign = "left";
    c.textBaseline = "middle";
    c.fillText(q.plano.legenda_figura || "", 124, 96);
  });
}

// Plano reservado a gravacao real: o video do usuario, ou a lista do que falta gravar.
function desenharTelao(c, q) {
  const a = R.janela(q.p, 0.32, 0.42);
  if (!q.plano.telao || a <= 0) return;
  comOpacidade(c, a, function () {
    c.fillStyle = "#0c1216";
    c.fillRect(0, 0, LARGURA, ALTURA);
    if (q.video && q.video.readyState >= 2) {
      const escala = Math.min(LARGURA / q.video.videoWidth, ALTURA / q.video.videoHeight);
      const w = q.video.videoWidth * escala;
      const h = q.video.videoHeight * escala;
      c.drawImage(q.video, (LARGURA - w) / 2, (ALTURA - h) / 2, w, h);
      return;
    }
    c.textAlign = "left";
    c.textBaseline = "middle";
    c.fillStyle = "#f2c230";
    c.font = "700 58px " + FONTE;
    c.fillText("Gravação real entra aqui", 260, 280);
    c.font = "500 38px " + FONTE;
    c.fillStyle = "#f2f5f4";
    ["Treinamento e funcionamento da LLM", "Execução de um fluxo automatizado", "Perguntas clínicas contextualizadas", "Logs e validação das respostas"]
      .forEach(function (item, i) {
        c.fillStyle = "#f2c230";
        c.fillRect(260, 384 + i * 86, 22, 22);
        c.fillStyle = "#f2f5f4";
        c.fillText(item, 310, 396 + i * 86);
      });
  });
}

function desenharLegenda(c, q) {
  if (!q.legendas || !q.legenda) return;
  c.font = "500 39px " + FONTE;
  const linhas = quebrar(c, q.legenda.texto, 1300);
  const altura = 58;
  const topo = ALTURA - 96 - linhas.length * altura;
  let maior = 0;
  linhas.forEach(function (l) { maior = Math.max(maior, c.measureText(l).width); });
  caixaRedonda(c, (LARGURA - maior) / 2 - 30, topo - 10, maior + 60, linhas.length * altura + 20, 12, "rgba(10, 15, 18, .78)");
  c.fillStyle = "#ffffff";
  c.textAlign = "center";
  c.textBaseline = "middle";
  linhas.forEach(function (l, i) { c.fillText(l, LARGURA / 2, topo + altura * i + altura / 2); });
}

function desenharFonte(c, q) {
  if (!q.plano.fonte) return;
  const a = R.janela(q.tPlano, 1, 1.5) * (1 - R.janela(q.p, 0.94, 1));
  comOpacidade(c, a, function () {
    c.font = "24px Consolas, ui-monospace, monospace";
    const texto = "Fonte: " + q.plano.fonte;
    const w = c.measureText(texto).width;
    caixaRedonda(c, LARGURA - w - 84, 74, w + 36, 42, 8, "rgba(12, 18, 22, .78)");
    c.fillStyle = "#7fe3cf";
    c.textAlign = "left";
    c.textBaseline = "middle";
    c.fillText(texto, LARGURA - w - 66, 95);
  });
}

function desenharRascunho(c, q) {
  if (!q.filme.rascunho) return;
  c.save();
  c.translate(1700, 996);
  c.rotate(-0.06);
  c.strokeStyle = "rgba(242, 194, 48, .85)";
  c.lineWidth = 4;
  c.strokeRect(-150, -32, 300, 64);
  c.fillStyle = "rgba(242, 194, 48, .9)";
  c.font = "700 34px " + FONTE;
  c.textAlign = "center";
  c.textBaseline = "middle";
  c.fillText("RASCUNHO", 0, 2);
  c.restore();
}

function desenharCreditos(c, q) {
  const t = q.t - q.linha.creditos;
  const escuro = R.janela(t, 0, 1.2);
  c.fillStyle = "rgba(9, 13, 16, " + 0.82 * escuro + ")";
  c.fillRect(0, 0, LARGURA, ALTURA);
  comOpacidade(c, R.janela(t, 0.6, 1.4), function () {
    const y = 1080 - t * 92;
    c.textBaseline = "middle";
    c.textAlign = "center";
    c.fillStyle = "#ffffff";
    c.font = "700 58px " + FONTE;
    c.fillText(q.filme.selo + ": " + q.filme.titulo, LARGURA / 2, y);
    c.fillStyle = "#e0528c";
    c.font = "600 30px " + FONTE;
    c.fillText("ELENCO", LARGURA / 2, y + 110);
    DADOS.salas.forEach(function (sala, i) {
      c.fillStyle = "#f2f5f4";
      c.textAlign = "right";
      c.font = "600 32px " + FONTE;
      c.fillText(sala.robo.nome, LARGURA / 2 - 20, y + 170 + i * 50);
      c.fillStyle = "#b5c2c3";
      c.textAlign = "left";
      c.font = "500 32px " + FONTE;
      c.fillText(sala.robo.papel, LARGURA / 2 + 20, y + 170 + i * 50);
    });
    const base = y + 170 + DADOS.salas.length * 50 + 60;
    c.textAlign = "center";
    c.fillStyle = "#e0528c";
    c.font = "600 30px " + FONTE;
    c.fillText("NÚMEROS", LARGURA / 2, base);
    c.fillStyle = "#f2f5f4";
    c.font = "500 30px " + FONTE;
    c.fillText("Todos lidos de results/ na hora de montar este filme.", LARGURA / 2, base + 56);
    c.fillStyle = "#b5c2c3";
    c.font = "500 28px " + FONTE;
    quebrar(c, DADOS.aviso, 1100).forEach(function (l, i) { c.fillText(l, LARGURA / 2, base + 130 + i * 44); });
  });
}

function barrasDeCinema(c, a) {
  const h = 64 * a;
  c.fillStyle = "#000";
  c.fillRect(0, 0, LARGURA, h);
  c.fillRect(0, ALTURA - h, LARGURA, h);
}

function desenharLetreiros(c, q) {
  if (q.indice < 0) {
    desenharClaquete(c, q);
  } else if (q.plano) {
    desenharFigura(c, q);
    desenharTelao(c, q);
    barrasDeCinema(c, 1);
    if (q.indice === 0) desenharTitulo(c, q);
    else desenharCapitulo(c, q);
    desenharNomeDoRobo(c, q);
    desenharContadores(c, q);
    desenharFonte(c, q);
    desenharLegenda(c, q);
  } else {
    barrasDeCinema(c, 1);
    desenharCreditos(c, q);
  }
  if (q.indice >= 0) desenharRascunho(c, q);
  if (q.escuro > 0) {
    c.fillStyle = "rgba(0, 0, 0, " + q.escuro + ")";
    c.fillRect(0, 0, LARGURA, ALTURA);
  }
}
