// Logica pura dos filmes: linha do tempo, legendas e arquivos de apoio.
//
// Sem DOM e sem Three.js, para ser testada no Node, como `logica.js`.
const HospitalRoteiro = (function () {
  "use strict";

  const PALAVRAS_POR_SEGUNDO = 2.35;
  const PAUSA = 0.9;
  const PLANO_MINIMO = 4.5;
  const CLAQUETE = 2.8;
  const CREDITOS = 12;

  function palavras(texto) {
    return texto.trim().split(/\s+/).length;
  }

  // Tempo do plano: pela voz gravada, quando existe, ou pelo ritmo de leitura.
  function duracaoDoPlano(plano, medidas) {
    const medida = medidas && medidas[plano.id];
    const base = medida ? medida + 0.7 : palavras(plano.narracao) / PALAVRAS_POR_SEGUNDO + PAUSA;
    return Math.max(PLANO_MINIMO, base);
  }

  function linhaDoTempo(filme, medidas) {
    let t = CLAQUETE;
    const planos = filme.planos.map(function (plano, indice) {
      const duracao = duracaoDoPlano(plano, medidas);
      const item = Object.assign({}, plano, { indice: indice, inicio: t, fim: t + duracao, duracao: duracao });
      t += duracao;
      return item;
    });
    return { planos: planos, creditos: t, total: t + CREDITOS };
  }

  // -1 durante a claquete; `planos.length` nos creditos.
  function indiceEm(linha, t) {
    if (t < CLAQUETE) return -1;
    for (let i = 0; i < linha.planos.length; i++) {
      if (t < linha.planos[i].fim) return i;
    }
    return linha.planos.length;
  }

  function frases(texto) {
    return texto.match(/[^.?!:]+[.?!:]*(?:\s+|$)/g).map(function (f) { return f.trim(); }).filter(Boolean);
  }

  // Quebra virgula fina: "0,964" nao pode partir a frase, e a regex acima so olha . ? ! :.
  function trechos(plano) {
    const lista = frases(plano.narracao);
    const total = lista.reduce(function (soma, f) { return soma + palavras(f); }, 0);
    const util = plano.duracao - Math.min(PAUSA, plano.duracao * 0.15);
    let t = plano.inicio;
    return lista.map(function (texto) {
      const duracao = util * palavras(texto) / total;
      const item = { texto: texto, inicio: t, fim: t + duracao };
      t += duracao;
      return item;
    });
  }

  function legendaEm(plano, t) {
    return trechos(plano).find(function (trecho) { return t >= trecho.inicio && t < trecho.fim; }) || null;
  }

  function doisDigitos(n) {
    return String(n).padStart(2, "0");
  }

  function tempoSrt(s) {
    const ms = Math.round(s * 1000);
    const h = Math.floor(ms / 3600000);
    const m = Math.floor(ms / 60000) % 60;
    const seg = Math.floor(ms / 1000) % 60;
    return doisDigitos(h) + ":" + doisDigitos(m) + ":" + doisDigitos(seg) + "," + String(ms % 1000).padStart(3, "0");
  }

  function srt(linha) {
    const blocos = [];
    linha.planos.forEach(function (plano) {
      trechos(plano).forEach(function (trecho) {
        blocos.push((blocos.length + 1) + "\n" + tempoSrt(trecho.inicio) + " --> " + tempoSrt(trecho.fim) + "\n" + trecho.texto + "\n");
      });
    });
    return blocos.join("\n");
  }

  function tempoCurto(s) {
    const total = Math.floor(s);
    return doisDigitos(Math.floor(total / 60)) + ":" + doisDigitos(total % 60);
  }

  // Lista no formato que o YouTube transforma em capitulos: comeca em 00:00.
  function capitulos(linha) {
    return linha.planos.map(function (plano, i) {
      return tempoCurto(i === 0 ? 0 : plano.inicio) + " " + plano.capitulo;
    }).join("\n");
  }

  function roteiroEmTexto(filme, linha) {
    const cabeca = filme.selo + ": " + filme.titulo + "\nDuração: " + tempoCurto(linha.total) + "\n";
    return cabeca + linha.planos.map(function (plano) {
      return "\n[" + tempoCurto(plano.inicio) + "] " + plano.capitulo + "\n" + plano.narracao + "\n";
    }).join("");
  }

  function janela(p, a, b) {
    return Math.min(1, Math.max(0, (p - a) / (b - a)));
  }

  function suave(p) {
    return p * p * (3 - 2 * p);
  }

  const api = {
    CLAQUETE, CREDITOS, palavras, duracaoDoPlano, linhaDoTempo, indiceEm, frases, trechos, legendaEm,
    srt, capitulos, roteiroEmTexto, tempoCurto, janela, suave,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  return api;
})();
const R = HospitalRoteiro;
