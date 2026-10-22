// Tela da vitrine. Toda conta vem de logica.js; aqui so se desenha.
//
// Regra de seguranca: nenhum texto entra como HTML. Todo conteudo (inclusive o
// texto gerado pelo modelo de linguagem e as respostas da API) e inserido por
// textContent, em elementos criados aqui.
(function () {
  "use strict";

  var L = window.VitrineLogica;
  var dados = JSON.parse(document.getElementById("dados").textContent);
  var modo = "congelado";
  var graficos = {};

  var pct = new Intl.NumberFormat("pt-BR", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
  var num = function (casas) { return new Intl.NumberFormat("pt-BR", { maximumFractionDigits: casas, minimumFractionDigits: 0 }); };
  var n0 = num(0), n2 = num(2), n3 = num(3), n4 = num(4);
  var usd = new Intl.NumberFormat("pt-BR", { style: "currency", currency: "USD", minimumFractionDigits: 3 });

  // ------------------------------------------------------------------ utilidades
  function $(id) { return document.getElementById(id); }

  function plural(n, singular, plural_) { return n + " " + (n === 1 ? singular : plural_); }

  function el(tag, opcoes, filhos) {
    var no = document.createElement(tag);
    opcoes = opcoes || {};
    if (opcoes.classe) no.className = opcoes.classe;
    if (opcoes.texto !== undefined && opcoes.texto !== null) no.textContent = String(opcoes.texto);
    if (opcoes.atributos) Object.keys(opcoes.atributos).forEach(function (k) { no.setAttribute(k, opcoes.atributos[k]); });
    (filhos || []).forEach(function (f) { if (f) no.appendChild(f); });
    return no;
  }

  function limpar(no) { while (no.firstChild) no.removeChild(no.firstChild); return no; }

  function cor(nome) { return getComputedStyle(document.documentElement).getPropertyValue(nome).trim(); }

  function tabela(destino, cabecalho, linhas, opcoes) {
    opcoes = opcoes || {};
    var t = destino.tagName === "TABLE" ? limpar(destino) : limpar(destino).appendChild(el("table", { classe: "tabela" }));
    t.appendChild(el("thead", {}, [el("tr", {}, cabecalho.map(function (c) { return el("th", { texto: c }); }))]));
    var corpo = el("tbody");
    linhas.forEach(function (linha, i) {
      var tr = el("tr", { classe: opcoes.destacar && opcoes.destacar(i) ? "servido" : "" });
      linha.forEach(function (valor) {
        tr.appendChild(el("td", { texto: valor, classe: typeof valor === "number" || /^[\d.,%\s−-]+$/.test(String(valor)) ? "num" : "" }));
      });
      corpo.appendChild(tr);
    });
    t.appendChild(corpo);
  }

  function indicador(rotulo, valor, detalhe) {
    return el("div", { classe: "indicador" }, [
      el("div", { classe: "rotulo", texto: rotulo }),
      el("div", { classe: "valor", texto: valor }),
      detalhe ? el("div", { classe: "detalhe", texto: detalhe }) : null,
    ]);
  }

  function textoGerado(destino, texto, motivo) {
    limpar(destino);
    if (!texto) {
      destino.appendChild(el("p", { classe: "indisponivel", texto: "Texto em linguagem natural indisponível: " + (motivo || "não gerado") + "." }));
      return;
    }
    var lista = null;
    L.blocosDoTexto(texto).forEach(function (bloco) {
      var alvo;
      if (bloco.tipo === "item") {
        if (!lista) { lista = el("ul"); destino.appendChild(lista); }
        alvo = lista.appendChild(el("li"));
      } else {
        lista = null;
        alvo = destino.appendChild(el(bloco.tipo === "titulo" ? "h4" : "p"));
      }
      bloco.trechos.forEach(function (t) { alvo.appendChild(el(t.negrito ? "strong" : "span", { texto: t.texto })); });
    });
  }

  // ------------------------------------------------------------------ graficos
  function temaDosGraficos() {
    if (!window.Chart) return;
    Chart.defaults.color = cor("--tinta-2");
    Chart.defaults.borderColor = cor("--grade");
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.plugins.legend.labels.boxWidth = 10;
    Chart.defaults.plugins.legend.labels.boxHeight = 10;
    Chart.defaults.plugins.tooltip.backgroundColor = cor("--tinta");
    Chart.defaults.plugins.tooltip.titleColor = cor("--fundo");
    Chart.defaults.plugins.tooltip.bodyColor = cor("--fundo");
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.animation.duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 400;
  }

  function grafico(id, configuracao) {
    if (!window.Chart) return null;
    if (graficos[id]) graficos[id].destroy();
    graficos[id] = new Chart($(id), configuracao);
    return graficos[id];
  }

  function barrasDeContribuicao(id, contribuicoes) {
    var maiores = contribuicoes.slice(0, 10);
    var maligno = cor("--maligno"), benigno = cor("--benigno");
    grafico(id, {
      type: "bar",
      data: {
        labels: maiores.map(function (c) { return c.feature; }),
        datasets: [{
          data: maiores.map(function (c) { return c.contribuicao; }),
          backgroundColor: maiores.map(function (c) { return c.contribuicao > 0 ? maligno : benigno; }),
          borderRadius: 4, borderSkipped: false, barPercentage: 0.7,
        }],
      },
      options: {
        indexAxis: "y",
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: function (c) {
            return n3.format(c.parsed.x) + (c.parsed.x > 0 ? " (empurra para maligno)" : " (empurra para benigno)");
          } } },
        },
        scales: {
          x: { title: { display: true, text: "Contribuição (log-odds)" }, grid: { color: cor("--grade") } },
          y: { grid: { display: false } },
        },
      },
    });
  }

  // ------------------------------------------------------------------ abertura
  function abertura() {
    var a = L.achados(dados);
    var alvo = limpar($("indicadores"));
    alvo.appendChild(indicador(
      "Modelo servido",
      dados.modelo.familia || "—",
      dados.modelo.features.length + " medidas · limiar " + pct.format(dados.modelo.limiar)
    ));
    if (a.servido) {
      alvo.appendChild(indicador(
        "Falsos negativos no teste",
        a.servido.falsos_negativos + " de " + a.malignos_no_teste,
        "tumores malignos não sinalizados; referência da Fase 1: " + (a.original ? a.original.falsos_negativos : "—")
      ));
    }
    if (dados.textos) {
      alvo.appendChild(indicador(
        "Textos aprovados",
        dados.textos.aprovados + " de " + dados.textos.total,
        "nas quatro verificações automáticas"
      ));
      if (dados.textos.custo) {
        alvo.appendChild(indicador(
          "Custo de gerar todos os textos",
          usd.format(dados.textos.custo.custo_usd),
          dados.textos.custo.modelo + ", em lote, uma única vez"
        ));
      }
    }
  }

  // ------------------------------------------------------------------ capitulo 1
  function processo() {
    var p = dados.processo;
    if (!p) {
      $("jornada").appendChild(el("li", {}, [el("p", { texto: "Resultados do algoritmo genético não encontrados em results/genetic/." })]));
      return;
    }
    var a = L.achados(dados);
    var experimentos = Object.keys(p.convergencia);
    var familias = {};
    experimentos.forEach(function (e) { Object.keys(p.convergencia[e]).forEach(function (f) { familias[f] = true; }); });
    var bracos = p.bracos.slice().sort(function (x, y) { return x.custo_medio - y.custo_medio; });
    var cfg = p.configuracoes[experimentos[0]] || {};

    var passos = [
      ["Referência da Fase 1", a.original
        ? "Regressão logística, 30 medidas, limiar 0,5: " + plural(a.original.falsos_negativos, "falso negativo", "falsos negativos") + " e " +
          plural(a.original.falsos_positivos, "falso positivo", "falsos positivos") + " no teste."
        : "Regressão logística com as 30 medidas."],
      ["Busca genética", p.execucoes + " buscas: " + experimentos.length + " configurações × " + Object.keys(familias).length +
        " famílias × sementes. Cada indivíduo carrega medidas, hiperparâmetros e limiar; população de " + (cfg.populacao || "—") +
        " por " + (cfg.geracoes || "—") + " gerações."],
      ["Custo clínico como fitness", "Falso negativo pesa " + p.pesos.falso_negativo + "×, falso positivo " + p.pesos.falso_positivo +
        "×, mais uma penalidade por medida usada. Avaliado por validação cruzada no treino."],
      ["Braços de ablação", "Otimizar só o limiar, só as medidas ou só os hiperparâmetros isola de onde vem o ganho. Menor custo: " +
        bracos[0].braco.replace("AG, ", "") + "."],
      ["Regra registrada antes do teste", p.regra
        ? "Em " + p.regra.decidida_em.split("-").reverse().join("/") + ": servir o melhor fitness em validação cruzada, com trava se piorar os falsos negativos da referência."
        : "Critério de escolha fixado antes de olhar o teste."],
      ["Modelo em produção", (a.trava_disparou ? "A trava disparou: servida a referência. " : "A trava não disparou. ") +
        dados.modelo.familia + " com " + dados.modelo.features.length + " medidas e limiar " + pct.format(dados.modelo.limiar) + "."],
    ];
    var jornada = limpar($("jornada"));
    passos.forEach(function (passo, i) {
      jornada.appendChild(el("li", { classe: i === passos.length - 1 ? "final" : "" }, [el("h3", { texto: passo[0] }), el("p", { texto: passo[1] })]));
    });

    var achado = limpar($("achado"));
    achado.appendChild(el("strong", { texto: "O achado central" }));
    achado.appendChild(el("p", { texto:
      "O ganho em falsos negativos veio de mover o ponto de operação (limiar ou peso das classes), não de escolher medidas. " +
      "No teste, o menor número de falsos negativos (" + a.menor_fn + ") veio dos braços " +
      a.bracos_com_menor_fn.map(function (b) { return "“" + b.replace("AG, ", "") + "”"; }).join(" e ") + ". " +
      "Com " + p.teste[0].n_amostras + " casos, cada falso negativo vale " + n3.format(a.custo_de_um_fn) +
      " de custo clínico: diferenças de um ou dois casos entre braços estão dentro do ruído." }));

    var seletor = $("f-experimento");
    if (!seletor.options.length) {
      experimentos.forEach(function (e) { seletor.appendChild(el("option", { texto: e, atributos: { value: e } })); });
      seletor.addEventListener("change", function () { convergencia(experimentos); });
    }
    convergencia(experimentos);

    var s1 = cor("--serie-1");
    grafico("g-bracos", {
      type: "bar",
      data: { labels: bracos.map(function (b) { return b.braco.replace("AG, ", ""); }),
        datasets: [{ label: "Custo clínico médio", data: bracos.map(function (b) { return b.custo_medio; }),
          backgroundColor: s1, borderRadius: 4, barPercentage: 0.6 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: {
        label: function (c) { var b = bracos[c.dataIndex];
          return ["custo " + n4.format(b.custo_medio) + " ± " + n4.format(b.custo_desvio),
            "FN médio " + n2.format(b.falsos_negativos_medio) + " · FP médio " + n2.format(b.falsos_positivos_medio),
            n2.format(b.features_medio) + " medidas"]; } } } },
        scales: { x: { beginAtZero: true, title: { display: true, text: "Custo clínico (validação cruzada)" } }, y: { grid: { display: false } } } },
    });
    tabela($("d-bracos"), ["Braço", "Custo médio", "Desvio", "FN médio", "FP médio", "Medidas"], bracos.map(function (b) {
      return [b.braco, n4.format(b.custo_medio), n4.format(b.custo_desvio), n2.format(b.falsos_negativos_medio), n2.format(b.falsos_positivos_medio), n2.format(b.features_medio)];
    }));

    if (p.teste) {
      var teste = p.teste.slice().sort(function (x, y) { return x.falsos_negativos - y.falsos_negativos || x.falsos_positivos - y.falsos_positivos; });
      var rotulo = function (l) { return l.braco.replace("AG, ", "") + (l.braco === "AG, completo" ? " (servido)" : ""); };
      grafico("g-teste", {
        type: "bar",
        data: { labels: teste.map(rotulo), datasets: [
          { label: "Falsos negativos", data: teste.map(function (l) { return l.falsos_negativos; }), backgroundColor: cor("--maligno"), borderRadius: 4 },
          { label: "Falsos positivos", data: teste.map(function (l) { return l.falsos_positivos; }), backgroundColor: cor("--benigno"), borderRadius: 4 },
        ] },
        options: { indexAxis: "y", plugins: { tooltip: { callbacks: { afterBody: function (itens) { var l = teste[itens[0].dataIndex];
          return ["AUC " + n4.format(l.roc_auc) + " · recall " + n4.format(l.recall), l.n_features + " medidas · limiar " + n4.format(l.limiar), l.experimento + " · " + l.familia]; } } } },
          scales: { x: { beginAtZero: true, ticks: { precision: 0 }, title: { display: true, text: "Casos (de " + teste[0].n_amostras + ")" } }, y: { grid: { display: false } } } },
      });
      tabela($("d-teste"), ["Abordagem", "FN", "FP", "AUC", "Recall", "Medidas", "Limiar", "Custo clínico"], teste.map(function (l) {
        return [rotulo(l), l.falsos_negativos, l.falsos_positivos, n4.format(l.roc_auc), n4.format(l.recall), l.n_features, n4.format(l.limiar), n4.format(l.custo_clinico)];
      }), { destacar: function (i) { return teste[i].braco === "AG, completo"; } });
    }

    var razoes = [], penalidades = [];
    p.varredura.forEach(function (v) {
      if (razoes.indexOf(v.razao_custo) < 0) razoes.push(v.razao_custo);
      if (penalidades.indexOf(v.parcimonia) < 0) penalidades.push(v.parcimonia);
    });
    razoes.sort(function (x, y) { return x - y; });
    var series = ["--serie-1", "--serie-2", "--serie-3"];
    grafico("g-varredura", {
      type: "line",
      data: { labels: razoes.map(function (r) { return r + ":1"; }), datasets: penalidades.map(function (pen, i) {
        var linhas = razoes.map(function (r) { return p.varredura.filter(function (v) { return v.razao_custo === r && v.parcimonia === pen; })[0]; });
        return { label: "penalidade " + n2.format(pen), data: linhas.map(function (v) { return v ? v.limiar_medio : null; }), _linhas: linhas,
          borderColor: cor(series[i % 3]), backgroundColor: cor(series[i % 3]), borderWidth: 2, pointRadius: 4, pointHoverRadius: 6 };
      }) },
      options: { interaction: { mode: "index", intersect: false }, plugins: { tooltip: { callbacks: { label: function (c) {
        var v = c.dataset._linhas[c.dataIndex];
        return c.dataset.label + ": limiar " + n3.format(v.limiar_medio) + " · " + n2.format(v.features_medio) + " medidas · FN " + n2.format(v.falsos_negativos_medio) + " · FP " + n2.format(v.falsos_positivos_medio);
      } } } },
        scales: { x: { title: { display: true, text: "Peso do falso negativo sobre o falso positivo" } }, y: { beginAtZero: true, title: { display: true, text: "Limiar médio" } } } },
    });
    tabela($("d-varredura"), ["Razão", "Penalidade", "Limiar médio", "Medidas", "FN médio", "FP médio"], p.varredura.map(function (v) {
      return [v.razao_custo + ":1", n2.format(v.parcimonia), n4.format(v.limiar_medio), n2.format(v.features_medio), n2.format(v.falsos_negativos_medio), n2.format(v.falsos_positivos_medio)];
    }));

    var features = p.features.slice().sort(function (x, y) { return y.frequencia - x.frequencia; }).slice(0, 12);
    grafico("g-features", {
      type: "bar",
      data: { labels: features.map(function (f) { return f.feature; }), datasets: [{ label: "Frequência", data: features.map(function (f) { return f.frequencia; }),
        backgroundColor: features.map(function (f) { return dados.modelo.features.indexOf(f.feature) >= 0 ? cor("--destaque") : cor("--serie-1"); }),
        borderRadius: 4, barPercentage: 0.7 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: function (c) { var f = features[c.dataIndex];
        return [pct.format(f.frequencia) + " das execuções (" + f.selecoes + " de " + f.execucoes + ")", "posição no SHAP da Fase 1: " + f.posicao_shap,
          dados.modelo.features.indexOf(f.feature) >= 0 ? "usada pelo modelo servido" : ""]; } } } },
        scales: { x: { beginAtZero: true, max: 1, ticks: { callback: function (v) { return pct.format(v); } } }, y: { grid: { display: false } } } },
    });
    tabela($("d-features"), ["Medida", "Frequência", "Seleções", "Posição no SHAP", "No modelo servido"], p.features.map(function (f) {
      return [f.feature, pct.format(f.frequencia), f.selecoes + " de " + f.execucoes, f.posicao_shap, dados.modelo.features.indexOf(f.feature) >= 0 ? "sim" : ""];
    }));
  }

  function convergencia(experimentos) {
    var p = dados.processo;
    var escolhido = $("f-experimento").value || experimentos[0];
    var curvas = p.convergencia[escolhido];
    var familias = Object.keys(curvas);
    var series = ["--serie-1", "--serie-2", "--serie-3"];
    var geracoes = Math.max.apply(null, familias.map(function (f) { return curvas[f].length; }));
    grafico("g-convergencia", {
      type: "line",
      data: { labels: Array.from({ length: geracoes }, function (_, i) { return i; }), datasets: familias.map(function (f, i) {
        return { label: f, data: curvas[f], borderColor: cor(series[i % 3]), backgroundColor: cor(series[i % 3]), borderWidth: 2, pointRadius: 0, pointHoverRadius: 5, tension: 0.15 };
      }) },
      options: { interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { callbacks: { title: function (i) { return "Geração " + i[0].label; }, label: function (c) { return c.dataset.label + ": " + n4.format(c.parsed.y); } } } },
        scales: { x: { title: { display: true, text: "Geração" }, ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: "Melhor fitness" } } } },
    });
    tabela($("d-convergencia"), ["Geração"].concat(familias), Array.from({ length: geracoes }, function (_, g) {
      return [g].concat(familias.map(function (f) { return curvas[f][g] === undefined ? "" : n4.format(curvas[f][g]); }));
    }));
  }

  // ------------------------------------------------------------------ capitulo 2
  function casos() {
    var seletor = $("f-caso");
    var soErros = $("f-erros");
    function preencher() {
      var atual = seletor.value;
      limpar(seletor);
      var lista = soErros.checked ? L.casosComErro(dados.casos) : dados.casos.map(function (c, i) { return { caso: c, indice: i }; });
      lista.forEach(function (par) {
        var erro = par.caso.classe !== par.caso.diagnostico_registrado ? " · previsto " + par.caso.classe : "";
        seletor.appendChild(el("option", { texto: par.caso.rotulo + erro, atributos: { value: String(par.indice) } }));
      });
      if (lista.some(function (par) { return String(par.indice) === atual; })) seletor.value = atual;
      mostrarCaso(Number(seletor.value));
    }
    seletor.addEventListener("change", function () { mostrarCaso(Number(seletor.value)); });
    soErros.addEventListener("change", preencher);
    $("abrir-no-simulador").addEventListener("click", function () {
      var caso = dados.casos[Number(seletor.value)];
      definirSimulador(caso.medidas);
      $("simulador").scrollIntoView();
    });
    preencher();
  }

  function cartao(rotulo, valor, classe) {
    return el("div", { classe: "cartao" + (classe === "alerta" ? " alerta" : "") }, [
      el("div", { classe: "rotulo", texto: rotulo }),
      el("div", { classe: "valor " + (classe && classe !== "alerta" ? classe : ""), texto: valor }),
    ]);
  }

  function classeCss(classe) { return classe === "Maligno" ? "maligno" : "benigno"; }

  function mostrarCaso(indice) {
    var caso = dados.casos[indice];
    if (!caso) return;
    var acertou = caso.classe === caso.diagnostico_registrado;
    var alvo = limpar($("c-caso"));
    alvo.appendChild(cartao("Previsão do modelo", caso.classe, classeCss(caso.classe)));
    alvo.appendChild(cartao("Probabilidade de malignidade", pct.format(caso.probabilidade)));
    alvo.appendChild(cartao("Limiar de decisão", pct.format(caso.limiar)));
    alvo.appendChild(cartao("Diagnóstico registrado", caso.diagnostico_registrado, classeCss(caso.diagnostico_registrado)));
    alvo.appendChild(cartao("Resultado", acertou ? "Acerto" : (caso.diagnostico_registrado === "Maligno" ? "Falso negativo" : "Falso positivo"), acertou ? null : "alerta"));
    barrasDeContribuicao("g-caso", caso.contribuicoes);
    tabela($("d-caso"), ["Medida", "Contribuição"], caso.contribuicoes.map(function (c) { return [c.feature, n4.format(c.contribuicao)]; }));
    textoGerado($("texto-caso"), caso.texto, caso.texto_motivo);
  }

  // ------------------------------------------------------------------ capitulo 3
  var simulacao = {};
  var esperaDaApi = null;

  function simulador() {
    var s = dados.simulador;
    var controles = limpar($("sim-controles"));
    dados.modelo.features.forEach(function (nome) {
      var f = s.faixas[nome];
      var passo = (f.maximo - f.minimo) / 400;
      var id = "sim-" + nome.replace(/\W+/g, "-");
      var entrada = el("input", { atributos: { type: "range", id: id, min: String(f.minimo), max: String(f.maximo), step: String(passo) } });
      var saida = el("output", { atributos: { for: id } });
      entrada.addEventListener("input", function () { simulacao[nome] = Number(entrada.value); atualizarSimulacao(); });
      controles.appendChild(el("div", { classe: "controle", atributos: { "data-medida": nome } }, [
        el("label", { texto: nome, atributos: { for: id } }), saida, entrada,
        el("div", { classe: "faixa" }, [el("span", { texto: n4.format(f.minimo) }), el("span", { texto: n4.format(f.maximo) })]),
      ]));
    });
    Array.prototype.forEach.call(document.querySelectorAll("[data-perfil]"), function (botao) {
      botao.addEventListener("click", function () {
        var perfil = {};
        dados.modelo.features.forEach(function (n) { perfil[n] = s.faixas[n][botao.getAttribute("data-perfil")]; });
        definirSimulador(perfil);
      });
    });
    var media = {};
    dados.modelo.features.forEach(function (n) { media[n] = s.faixas[n].media; });
    definirSimulador(media);
  }

  function definirSimulador(medidas) {
    dados.modelo.features.forEach(function (nome) {
      simulacao[nome] = Number(medidas[nome]);
      var entrada = document.querySelector('[data-medida="' + nome + '"] input');
      if (entrada) entrada.value = String(simulacao[nome]);
    });
    atualizarSimulacao();
  }

  function atualizarSimulacao() {
    dados.modelo.features.forEach(function (nome) {
      var saida = document.querySelector('[data-medida="' + nome + '"] output');
      if (saida) saida.textContent = n4.format(simulacao[nome]);
    });
    if (modo === "ao-vivo") {
      clearTimeout(esperaDaApi);
      esperaDaApi = setTimeout(function () {
        fetch("/explain", { method: "POST", headers: { "Content-Type": "application/json", "X-Request-ID": "vitrine-simulador" },
          body: JSON.stringify({ medidas: simulacao }) })
          .then(function (r) { return r.json(); })
          .then(mostrarSimulacao)
          .catch(function () { $("sim-classe").textContent = "API indisponível"; });
      }, 250);
    } else if (dados.simulador.parametros) {
      mostrarSimulacao(L.calcular(dados.simulador.parametros, simulacao));
    }
  }

  function mostrarSimulacao(resultado) {
    var limiar = dados.modelo.limiar;
    $("sim-prob").textContent = pct.format(resultado.probabilidade);
    var barra = $("sim-barra");
    barra.style.width = (100 * resultado.probabilidade).toFixed(2) + "%";
    barra.style.background = resultado.probabilidade >= limiar ? cor("--maligno") : cor("--benigno");
    $("sim-limiar").style.left = (100 * limiar).toFixed(2) + "%";
    $("sim-limiar-texto").textContent = "limiar " + pct.format(limiar);
    var classe = $("sim-classe");
    classe.textContent = "Previsão: " + resultado.classe;
    classe.className = "classe " + classeCss(resultado.classe);
    barrasDeContribuicao("g-sim", resultado.contribuicoes);
  }

  function origemDoSimulador() {
    var nota = $("sim-origem");
    if (modo === "ao-vivo") {
      nota.textContent = "Ao vivo: cada mudança é enviada a POST /explain, e a tela mostra a resposta da API.";
    } else if (dados.simulador.parametros) {
      nota.textContent = "Versão estática: a conta é feita no navegador, com os coeficientes do modelo servido. " +
        "É a mesma conta da API, conferida nos " + dados.casos.length + " casos do teste.";
    } else {
      nota.textContent = "O modelo servido não é linear, então a conta não cabe no navegador: o simulador funciona na versão ao vivo, servida pelo container.";
      Array.prototype.forEach.call(document.querySelectorAll("#simulador input, #simulador button"), function (c) { c.disabled = true; });
    }
  }

  // ------------------------------------------------------------------ capitulo 4
  var cenarioAtual = null;
  var ultimoEvento = 0;
  var eventosVistos = [];

  function consoleDaApi() {
    var lista = limpar($("console-lista"));
    dados.console.cenarios.forEach(function (c) {
      var botao = el("button", { classe: "console-item", atributos: { type: "button", "aria-pressed": "false", "data-cenario": c.id, role: "listitem" } }, [
        el("span", { classe: "linha" }, [el("span", { classe: "metodo " + c.metodo.toLowerCase(), texto: c.metodo }), el("strong", { texto: c.titulo })]),
        el("code", { texto: c.rota }),
      ]);
      botao.addEventListener("click", function () { escolherCenario(c); if (modo !== "ao-vivo") enviar(); });
      lista.appendChild(botao);
    });
    $("req-enviar").addEventListener("click", enviar);
    $("log-metrics").addEventListener("change", desenharLog);

    var origem = $("console-origem");
    if (modo === "ao-vivo") {
      origem.textContent = "Ao vivo: as chamadas vão para a API deste servidor. Edite o corpo antes de enviar, se quiser. O log e as métricas se atualizam sozinhos.";
      lerLog(); setInterval(lerLog, 1500);
      lerMetricas(); setInterval(lerMetricas, 3000);
    } else {
      origem.textContent = "Versão estática: mostra uma sessão gravada na montagem desta página, com as respostas, o log e as métricas que a API produziu. " +
        "Para chamar a API de verdade, rode o container do projeto.";
      $("req-corpo").readOnly = true;
      $("req-enviar").textContent = "Ver resposta gravada";
      eventosVistos = dados.console.gravacao.eventos.slice();
      desenharLog();
      desenharMetricas(dados.console.gravacao.metricas);
    }
    escolherCenario(dados.console.cenarios[0]);
    if (modo !== "ao-vivo") enviar();
  }

  function escolherCenario(c) {
    cenarioAtual = c;
    Array.prototype.forEach.call(document.querySelectorAll(".console-item"), function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-cenario") === c.id ? "true" : "false");
    });
    var metodo = $("req-metodo");
    metodo.textContent = c.metodo;
    metodo.className = "metodo " + c.metodo.toLowerCase();
    $("req-rota").textContent = c.rota;
    $("req-descricao").textContent = c.descricao;
    $("req-corpo").value = c.corpo ? JSON.stringify(c.corpo, null, 2) : "";
    $("req-corpo").hidden = !c.corpo;
    $("req-corpo-rotulo").textContent = c.corpo ? "Corpo do pedido (JSON)" : "Sem corpo: " + c.metodo + " só lê.";
  }

  function mostrarResposta(status, latencia, id, corpo) {
    var s = $("res-status");
    s.textContent = String(status);
    s.className = "status " + (status < 400 ? "ok" : "erro");
    $("res-latencia").textContent = latencia === null ? "" : n2.format(latencia) + " ms";
    $("res-id").textContent = id ? "X-Request-ID: " + id : "";
    $("res-corpo").textContent = typeof corpo === "string" ? corpo : JSON.stringify(corpo, null, 2);
  }

  function enviar() {
    var c = cenarioAtual;
    if (!c) return;
    if (modo !== "ao-vivo") {
      var g = dados.console.gravacao.respostas[c.id];
      if (g) mostrarResposta(g.status, g.latencia_ms, g.id, g.corpo);
      return;
    }
    var corpo = null;
    if (c.corpo) {
      try { corpo = JSON.parse($("req-corpo").value); }
      catch (erro) { mostrarResposta("JSON inválido", null, null, String(erro.message)); return; }
    }
    var inicio = performance.now();
    fetch(c.rota, { method: c.metodo, headers: corpo ? { "Content-Type": "application/json" } : {}, body: corpo ? JSON.stringify(corpo) : undefined, cache: "no-store" })
      .then(function (r) {
        var latencia = performance.now() - inicio;
        var tipo = r.headers.get("content-type") || "";
        return (tipo.indexOf("json") >= 0 ? r.json() : r.text()).then(function (b) {
          mostrarResposta(r.status, latencia, r.headers.get("X-Request-ID"), b);
          lerLog(); lerMetricas();
        });
      })
      .catch(function (erro) { mostrarResposta("falha", null, null, String(erro)); });
  }

  function lerLog() {
    fetch("/logs/recentes?depois=" + ultimoEvento, { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (corpo) {
        if (!corpo.eventos.length) return;
        ultimoEvento = corpo.eventos[corpo.eventos.length - 1].seq;
        eventosVistos = eventosVistos.concat(corpo.eventos).slice(-200);
        desenharLog(corpo.eventos.length);
      })
      .catch(function () {});
  }

  function desenharLog(novos) {
    var alvo = limpar($("log"));
    var mostrarMetricas = $("log-metrics").checked;
    var visiveis = eventosVistos.filter(function (e) { return mostrarMetricas || e.rota !== "/metrics"; }).slice(-60).reverse();
    if (!visiveis.length) {
      alvo.appendChild(el("li", { classe: "vazio", texto: "Nenhum evento ainda. Faça uma chamada ao lado." }));
      return;
    }
    visiveis.forEach(function (e, i) {
      var classe = e.evento === "desvio" ? "desvio" : (e.status >= 400 ? "erro" : "");
      if (typeof novos === "number" && i < novos) classe += " novo";
      var momento = new Date(e.momento);
      var hora = isNaN(momento) ? "" : momento.toLocaleTimeString("pt-BR");
      var cabeca = e.evento === "desvio"
        ? [el("strong", { texto: "DESVIO" }), el("span", { texto: e.feature + " = " + n4.format(e.valor) + " (" + n2.format(e.desvios) + " desvios padrão)" })]
        : [el("strong", { texto: e.metodo + " " + e.rota }), el("span", { texto: "status " + e.status }), el("span", { texto: n2.format(e.latencia_ms) + " ms" }),
          e.resultado ? el("span", { texto: "resultado " + e.resultado }) : null];
      alvo.appendChild(el("li", { classe: classe }, [
        el("div", { classe: "cabeca-evento" }, [el("span", { texto: hora })].concat(cabeca)),
        el("div", { texto: JSON.stringify(e) }),
      ]));
    });
  }

  function lerMetricas() {
    fetch("/metrics", { cache: "no-store" }).then(function (r) { return r.text(); }).then(desenharMetricas).catch(function () {});
  }

  function desenharMetricas(texto) {
    var resumo = L.resumirMetricas(L.lerPrometheus(texto));
    var rotas = resumo.rotas;
    var chave = JSON.stringify(resumo);
    if (desenharMetricas.ultima === chave) return;
    desenharMetricas.ultima = chave;
    grafico("g-requisicoes", {
      type: "bar",
      data: { labels: rotas.map(function (r) { return r.rota; }), datasets: [
        { label: "Respondidas", data: rotas.map(function (r) { return r.requisicoes - r.erros; }), backgroundColor: cor("--serie-1"), borderRadius: 4, stack: "s", maxBarThickness: 26 },
        { label: "Com erro (4xx/5xx)", data: rotas.map(function (r) { return r.erros; }), backgroundColor: cor("--erro"), borderRadius: 4, stack: "s", maxBarThickness: 26 },
      ] },
      options: { indexAxis: "y", animation: false, plugins: { tooltip: { callbacks: { afterBody: function (i) { var r = rotas[i[0].dataIndex];
        return r.latencia_media_ms === null ? "" : "latência média " + n2.format(r.latencia_media_ms) + " ms"; } } } },
        scales: { x: { stacked: true, beginAtZero: true, ticks: { precision: 0 }, title: { display: true, text: "Requisições" } }, y: { stacked: true, grid: { display: false } } } },
    });
    grafico("g-alertas", {
      type: "bar",
      data: { labels: resumo.alertas.length ? resumo.alertas.map(function (a) { return a.feature; }) : ["nenhum alerta"],
        datasets: [{ label: "Alertas de desvio", data: resumo.alertas.length ? resumo.alertas.map(function (a) { return a.alertas; }) : [0],
          backgroundColor: cor("--maligno"), borderRadius: 4, barPercentage: 0.6, maxBarThickness: 26 }] },
      options: { indexAxis: "y", animation: false, plugins: { legend: { display: false }, title: { display: true, text: "Alertas de desvio por medida", align: "start" } },
        scales: { x: { beginAtZero: true, ticks: { precision: 0 } }, y: { grid: { display: false } } } },
    });
    var linhas = rotas.map(function (r) { return [r.rota, r.requisicoes, r.erros, r.latencia_media_ms === null ? "" : n2.format(r.latencia_media_ms) + " ms"]; });
    resumo.alertas.forEach(function (a) { linhas.push(["alerta: " + a.feature, a.alertas, "", ""]); });
    tabela($("d-metricas"), ["Rota", "Requisições", "Com erro", "Latência média"], linhas);
  }

  // ------------------------------------------------------------------ capitulo 5
  var VERIFICACOES = {
    fidelidade: "Todo número do texto existe na evidência fornecida ao modelo.",
    seguranca: "Nenhuma afirmação diagnóstica; a ressalva de apoio à decisão está presente.",
    cobertura: "As três medidas de maior peso no caso são citadas.",
    formato: "Seções esperadas presentes e tamanho dentro do limite.",
  };
  var PRODUTOS = {
    explicacao_paciente: "Explicação por paciente",
    resumo_gestao: "Resumo para a gestão clínica",
    nota_de_mudanca: "Nota de mudança",
  };
  function nomeDoProduto(identificador) {
    var base = identificador.replace(/_v\d+$/, "");
    return (PRODUTOS[base] || base) + " (v" + identificador.split("_v").pop() + ")";
  }

  function textos() {
    var t = dados.textos;
    if (!t) { $("t-indicadores").appendChild(el("p", { texto: "Resultados da camada de texto não encontrados em results/llm/." })); return; }
    var alvo = limpar($("t-indicadores"));
    alvo.appendChild(indicador("Textos aprovados", t.aprovados + " de " + t.total, "texto reprovado nunca chega à tela"));
    if (t.custo) {
      alvo.appendChild(indicador("Custo real do lote", usd.format(t.custo.custo_usd), n0.format(t.custo.tokens_entrada) + " tokens de entrada · " + n0.format(t.custo.tokens_saida) + " de saída"));
      alvo.appendChild(indicador("Modelo de linguagem", t.custo.modelo, "chamado só na geração em lote; o servidor nunca o chama"));
    }
    tabela($("t-verificacoes"), ["Verificação", "O que confere", "Aprovados"], Object.keys(t.verificacoes).map(function (nome) {
      return [nome, VERIFICACOES[nome] || "", t.verificacoes[nome] + " de " + t.total];
    }));
    if (t.prompts) {
      tabela($("t-prompts"), ["Prompt", "Fidelidade", "Segurança", "Cobertura", "Formato", "Aprovado"], t.prompts.map(function (p) {
        var v = p.verificacoes, s = function (x) { return x ? "sim" : "não"; };
        return [nomeDoProduto(p.identificador), s(v.fidelidade), s(v.seguranca), s(v.cobertura), s(v.formato), s(p.aprovado)];
      }));
    }
    var gestao = limpar($("t-gestao"));
    t.gestao.forEach(function (g) {
      var corpo = el("div", { classe: "texto-gerado" });
      textoGerado(corpo, g.texto);
      gestao.appendChild(el("article", { classe: "cartao-texto" }, [el("h3", { texto: nomeDoProduto(g.identificador) }), corpo]));
    });
  }

  // ------------------------------------------------------------------ rodape e modo
  function rodape() {
    var alvo = limpar($("rodape"));
    alvo.appendChild(el("p", { texto: "Modelo servido: " + dados.modelo.origem + ". Commit " + dados.modelo.commit + "." }));
    var links = el("p");
    links.appendChild(el("a", { texto: "Código e relatórios no GitHub", atributos: { href: "https://github.com/gustavofks/ML-breast-cancer" } }));
    if (modo === "ao-vivo") {
      links.appendChild(document.createTextNode(" · "));
      links.appendChild(el("a", { texto: "Documentação interativa da API (/docs)", atributos: { href: "/docs" } }));
      links.appendChild(document.createTextNode(" · "));
      links.appendChild(el("a", { texto: "Painel Gradio (/painel/)", atributos: { href: "/painel/" } }));
    }
    alvo.appendChild(links);
  }

  function selo() {
    var s = $("modo");
    s.className = "selo " + modo;
    s.textContent = modo === "ao-vivo" ? "Ao vivo — conectado à API" : "Versão estática — dados congelados";
  }

  function capituloAtual() {
    if (!("IntersectionObserver" in window)) return;
    var links = {};
    Array.prototype.forEach.call(document.querySelectorAll(".capitulos a"), function (a) { links[a.getAttribute("href").slice(1)] = a; });
    var observador = new IntersectionObserver(function (entradas) {
      entradas.forEach(function (e) {
        if (e.isIntersecting && links[e.target.id]) {
          Object.keys(links).forEach(function (k) { links[k].classList.toggle("atual", k === e.target.id); });
        }
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    Object.keys(links).forEach(function (id) { if ($(id)) observador.observe($(id)); });
  }

  function desenharTudo() {
    temaDosGraficos();
    abertura();
    processo();
    casos();
    origemDoSimulador();
    simulador();
    textos();
    rodape();
  }

  // Desenha ja com os dados congelados; se a API responder, troca para o modo
  // ao vivo nas partes que falam com ela.
  selo();
  desenharTudo();
  capituloAtual();
  L.detectarModo(window.fetch ? window.fetch.bind(window) : function () { return Promise.reject(); }).then(function (m) {
    modo = m;
    selo();
    origemDoSimulador();
    rodape();
    consoleDaApi();
    atualizarSimulacao();
  });

  // Tema do sistema trocado com a pagina aberta: os graficos leem as cores na
  // hora de desenhar, entao sao redesenhados.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    temaDosGraficos();
    processo();
    mostrarCaso(Number($("f-caso").value));
    atualizarSimulacao();
    desenharMetricas.ultima = null;
    if (modo !== "ao-vivo") desenharMetricas(dados.console.gravacao.metricas); else lerMetricas();
  });
})();
