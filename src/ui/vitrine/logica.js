// Logica pura da vitrine: nenhuma linha aqui toca o DOM.
//
// Fica separada da tela para ser testada no Node (tests/js/), sem navegador e
// sem pandas. A tela (vitrine.js) so chama estas funcoes e desenha o resultado.
(function (raiz) {
  "use strict";

  // A conta do modelo servido, a mesma de src/ui/simulador.py#calcular. O teste
  // do Node compara o resultado com as respostas congeladas de /explain.
  function calcular(parametros, medidas) {
    var logito = parametros.intercepto;
    var contribuicoes = parametros.features.map(function (nome, i) {
      var z = (Number(medidas[nome]) - parametros.media[i]) / parametros.escala[i];
      logito += parametros.coeficientes[i] * z;
      return { feature: nome, contribuicao: parametros.coeficientes[i] * (z - parametros.media_do_fundo[i]) };
    });
    var probabilidade = 1 / (1 + Math.exp(-logito));
    contribuicoes.sort(function (a, b) { return Math.abs(b.contribuicao) - Math.abs(a.contribuicao); });
    return {
      probabilidade: probabilidade,
      classe: probabilidade >= parametros.limiar ? "Maligno" : "Benigno",
      contribuicoes: contribuicoes,
    };
  }

  // A pagina esta servida pela API? So se /health responder que o modelo
  // carregou. No Space estatico a rota nao existe (404), aberta como arquivo o
  // fetch falha; nos dois casos a vitrine usa os dados congelados.
  function detectarModo(buscar, limiteMs) {
    var cancelar = typeof AbortController === "function" ? new AbortController() : null;
    var relogio = setTimeout(function () { if (cancelar) cancelar.abort(); }, limiteMs || 2500);
    return Promise.resolve()
      .then(function () { return buscar("/health", { cache: "no-store", signal: cancelar && cancelar.signal }); })
      .then(function (resposta) { return resposta.ok ? resposta.json() : null; })
      .then(function (corpo) { return Boolean(corpo && corpo.modelo_carregado === true) ? "ao-vivo" : "congelado"; })
      .catch(function () { return "congelado"; })
      .then(function (modo) { clearTimeout(relogio); return modo; });
  }

  // Texto do Prometheus em amostras { nome, rotulos, valor }. So o que /metrics
  // escreve: comentarios, nome, rotulos entre chaves e um numero.
  function lerPrometheus(texto) {
    var amostras = [];
    String(texto).split("\n").forEach(function (linha) {
      linha = linha.trim();
      if (!linha || linha.charAt(0) === "#") return;
      var m = linha.match(/^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+(\S+)$/);
      if (!m) return;
      var rotulos = {};
      (m[2] || "").replace(/(\w+)="((?:[^"\\]|\\.)*)"/g, function (_, chave, valor) {
        rotulos[chave] = valor.replace(/\\(.)/g, "$1");
        return "";
      });
      amostras.push({ nome: m[1], rotulos: rotulos, valor: Number(m[3]) });
    });
    return amostras;
  }

  // Requisicoes, latencia media e alertas, agregados para os graficos.
  function resumirMetricas(amostras) {
    var rotas = {};
    var alertas = {};
    amostras.forEach(function (a) {
      var rota = a.rotulos.rota;
      if (a.nome === "api_requisicoes_total") {
        rotas[rota] = rotas[rota] || { requisicoes: 0, erros: 0, soma: 0, contagem: 0 };
        rotas[rota].requisicoes += a.valor;
        if (Number(a.rotulos.status) >= 400) rotas[rota].erros += a.valor;
      } else if (a.nome === "api_latencia_segundos_sum" || a.nome === "api_latencia_segundos_count") {
        rotas[rota] = rotas[rota] || { requisicoes: 0, erros: 0, soma: 0, contagem: 0 };
        rotas[rota][a.nome.endsWith("_sum") ? "soma" : "contagem"] = a.valor;
      } else if (a.nome === "api_alertas_de_desvio_total") {
        alertas[a.rotulos.feature] = (alertas[a.rotulos.feature] || 0) + a.valor;
      }
    });
    return {
      rotas: Object.keys(rotas).sort().map(function (rota) {
        var r = rotas[rota];
        return {
          rota: rota,
          requisicoes: r.requisicoes,
          erros: r.erros,
          latencia_media_ms: r.contagem ? (1000 * r.soma) / r.contagem : null,
        };
      }),
      alertas: Object.keys(alertas).sort().map(function (feature) {
        return { feature: feature, alertas: alertas[feature] };
      }),
    };
  }

  // Markdown minimo dos textos gerados: titulos "## ", itens "- " e negrito.
  // Devolve blocos e trechos, nunca HTML: a tela cria os elementos e preenche
  // com textContent, entao nada do texto gerado vira marcacao.
  function blocosDoTexto(texto) {
    var blocos = [];
    String(texto || "").split("\n").forEach(function (linha) {
      linha = linha.trim();
      if (!linha) return;
      var tipo = "paragrafo";
      if (/^#{1,6}\s+/.test(linha)) { tipo = "titulo"; linha = linha.replace(/^#{1,6}\s+/, ""); }
      else if (/^[-*]\s+/.test(linha)) { tipo = "item"; linha = linha.replace(/^[-*]\s+/, ""); }
      blocos.push({ tipo: tipo, trechos: trechos(linha) });
    });
    return blocos;
  }

  function trechos(linha) {
    var saida = [];
    var partes = linha.split(/\*\*(.+?)\*\*/);
    partes.forEach(function (parte, i) {
      if (parte) saida.push({ texto: parte, negrito: i % 2 === 1 });
    });
    return saida;
  }

  // Indicadores da abertura e o achado central, calculados dos artefatos. Nada
  // aqui e numero digitado: muda o resultado, muda a tela.
  function achados(dados) {
    var teste = (dados.processo && dados.processo.teste) || [];
    var porBraco = {};
    teste.forEach(function (linha) { porBraco[linha.braco] = linha; });
    var servido = porBraco["AG, completo"] || null;
    var original = porBraco["Original"] || null;
    var menorFn = teste.length ? Math.min.apply(null, teste.map(function (l) { return l.falsos_negativos; })) : null;
    var n = teste.length ? teste[0].n_amostras : null;
    var peso = dados.processo && dados.processo.pesos ? dados.processo.pesos.falso_negativo : null;
    return {
      servido: servido,
      original: original,
      trava_disparou: Boolean(servido && original && servido.falsos_negativos > original.falsos_negativos),
      menor_fn: menorFn,
      bracos_com_menor_fn: teste.filter(function (l) { return l.falsos_negativos === menorFn; }).map(function (l) { return l.braco; }),
      // Custo clinico de um falso negativo a mais no teste: o peso dele sobre n.
      custo_de_um_fn: n && peso ? peso / n : null,
      malignos_no_teste: servido ? servido.falsos_negativos + servido.verdadeiros_positivos : null,
    };
  }

  // Casos em que a previsao do modelo difere do diagnostico registrado.
  function casosComErro(casos) {
    return casos
      .map(function (caso, indice) { return { caso: caso, indice: indice }; })
      .filter(function (par) { return par.caso.classe !== par.caso.diagnostico_registrado; });
  }

  var api = {
    calcular: calcular,
    detectarModo: detectarModo,
    lerPrometheus: lerPrometheus,
    resumirMetricas: resumirMetricas,
    blocosDoTexto: blocosDoTexto,
    achados: achados,
    casosComErro: casosComErro,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  raiz.VitrineLogica = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
