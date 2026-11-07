// Trilha sonora sintetizada na hora: sem arquivo de audio e sem licenca de musica.
//
// Tudo e agendado de uma vez num `BaseAudioContext`: a mesma funcao toca ao vivo e
// renderiza offline. Cada fase tem um clima: manha, tarde e noite.

const ACORDES = {
  fase1: [[48, 55, 64, 71], [45, 52, 60, 67], [41, 48, 57, 64], [43, 50, 59, 62]],
  fase2: [[50, 57, 65, 72], [46, 53, 62, 69], [43, 50, 62, 65], [45, 52, 60, 67]],
  fase3: [[45, 52, 60, 64], [41, 48, 57, 60], [38, 45, 53, 60], [40, 47, 55, 59]],
};
const COMPASSO = 8;

function frequencia(nota) {
  return 440 * Math.pow(2, (nota - 69) / 12);
}

function tocarAcorde(ctx, saida, notas, inicio, duracao) {
  notas.forEach(function (nota) {
    [-5, 5].forEach(function (desafino) {
      const osc = ctx.createOscillator();
      const ganho = ctx.createGain();
      osc.type = "triangle";
      osc.frequency.value = frequencia(nota);
      osc.detune.value = desafino;
      ganho.gain.setValueAtTime(0, inicio);
      ganho.gain.linearRampToValueAtTime(0.04, inicio + 2.2);
      ganho.gain.setValueAtTime(0.04, inicio + duracao - 1);
      ganho.gain.linearRampToValueAtTime(0, inicio + duracao + 2);
      osc.connect(ganho).connect(saida);
      osc.start(inicio);
      osc.stop(inicio + duracao + 2.1);
    });
  });
}

function tocarNota(ctx, saida, nota, inicio, volume) {
  const osc = ctx.createOscillator();
  const ganho = ctx.createGain();
  osc.type = "sine";
  osc.frequency.value = frequencia(nota);
  ganho.gain.setValueAtTime(0, inicio);
  ganho.gain.linearRampToValueAtTime(volume, inicio + 0.012);
  ganho.gain.exponentialRampToValueAtTime(0.0008, inicio + 1.5);
  osc.connect(ganho).connect(saida);
  osc.start(inicio);
  osc.stop(inicio + 1.6);
}

function tocarClaquete(ctx, saida, inicio) {
  const buffer = ctx.createBuffer(1, ctx.sampleRate * 0.09, ctx.sampleRate);
  const dados = buffer.getChannelData(0);
  let semente = 42;
  for (let i = 0; i < dados.length; i++) {
    semente = (semente * 16807) % 2147483647;
    dados[i] = (semente / 1073741823 - 1) * (1 - i / dados.length);
  }
  const fonte = ctx.createBufferSource();
  const ganho = ctx.createGain();
  fonte.buffer = buffer;
  ganho.gain.value = 0.7;
  fonte.connect(ganho).connect(saida);
  fonte.start(inicio);
}

// `desde`: tempo do filme em que a trilha entra, para regravar a partir de um plano.
function agendarTrilha(ctx, destino, linha, filme, desde) {
  const partida = desde || 0;
  const zero = ctx.currentTime - partida;
  const mestre = ctx.createGain();
  const filtro = ctx.createBiquadFilter();
  filtro.type = "lowpass";
  filtro.frequency.value = filme.id === "fase3" ? 760 : 1250;
  const eco = ctx.createDelay(1);
  const retorno = ctx.createGain();
  eco.delayTime.value = 0.42;
  retorno.gain.value = 0.34;
  filtro.connect(mestre);
  filtro.connect(eco);
  eco.connect(retorno).connect(eco);
  retorno.connect(mestre);
  mestre.connect(destino);

  const inicio = Math.max(ctx.currentTime, zero);
  mestre.gain.setValueAtTime(0, inicio);
  mestre.gain.linearRampToValueAtTime(0.8, inicio + 1.5);
  mestre.gain.setValueAtTime(0.8, zero + linha.total - 5);
  mestre.gain.linearRampToValueAtTime(0, zero + linha.total - 0.3);

  const acordes = ACORDES[filme.id];
  const agora = function (t) { return t >= partida; };
  for (let t = 0, i = 0; t < linha.total - 3; t += COMPASSO, i++) {
    const notas = acordes[i % acordes.length];
    if (t + COMPASSO >= partida) tocarAcorde(ctx, filtro, notas, Math.max(inicio, zero + t), COMPASSO);
    [0, 1.5, 3, 4.5, 6].forEach(function (d, j) {
      const tempo = t + d;
      if (agora(tempo) && (i + j) % 3 !== 2) tocarNota(ctx, filtro, notas[(j + i) % notas.length] + 24, zero + tempo, 0.032);
    });
  }
  // Bipe de monitor a cada plano: o hospital marca o tempo do filme.
  linha.planos.forEach(function (plano) {
    if (agora(plano.inicio)) tocarNota(ctx, mestre, 93, zero + plano.inicio, 0.05);
  });
  if (partida < R.CLAQUETE) tocarClaquete(ctx, mestre, zero + 1.7);
  return mestre;
}

// PCM de 16 bits: o formato mais simples que o ffmpeg e qualquer editor aceitam.
function audioEmWav(buffer) {
  const canais = buffer.numberOfChannels;
  const n = buffer.length;
  const bytes = 44 + n * canais * 2;
  const v = new DataView(new ArrayBuffer(bytes));
  const texto = function (pos, s) { for (let i = 0; i < s.length; i++) v.setUint8(pos + i, s.charCodeAt(i)); };
  texto(0, "RIFF");
  v.setUint32(4, bytes - 8, true);
  texto(8, "WAVEfmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, canais, true);
  v.setUint32(24, buffer.sampleRate, true);
  v.setUint32(28, buffer.sampleRate * canais * 2, true);
  v.setUint16(32, canais * 2, true);
  v.setUint16(34, 16, true);
  texto(36, "data");
  v.setUint32(40, n * canais * 2, true);
  const dados = [];
  for (let c = 0; c < canais; c++) dados.push(buffer.getChannelData(c));
  let pos = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < canais; c++) {
      v.setInt16(pos, Math.max(-1, Math.min(1, dados[c][i])) * 32767, true);
      pos += 2;
    }
  }
  return new Uint8Array(v.buffer);
}
