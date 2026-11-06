// Tudo que e DOM: dialogo, escolha de caso, quadro ampliado e dicas.

const $ = function (id) { return document.getElementById(id); };

function escreverDevagar(elemento, texto, aoTerminar) {
  clearInterval(escreverDevagar.relogio);
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
    elemento.textContent = texto;
    if (aoTerminar) aoTerminar();
    return;
  }
  let i = 0;
  elemento.textContent = "";
  escreverDevagar.relogio = setInterval(function () {
    i = Math.min(texto.length, i + 2);
    elemento.textContent = texto.slice(0, i);
    if (i >= texto.length) {
      clearInterval(escreverDevagar.relogio);
      if (aoTerminar) aoTerminar();
    }
  }, 18);
}

function rostoDoRobo(cor) {
  return '<svg viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="34" r="25" fill="#f6f4ee"/>' +
    '<rect x="15" y="24" width="34" height="19" rx="9.5" fill="#10191d"/>' +
    '<circle cx="25" cy="33" r="3" fill="#8ff2e4"/><circle cx="39" cy="33" r="3" fill="#8ff2e4"/>' +
    '<rect x="30.8" y="4" width="2.4" height="8" fill="#232c31"/><circle cx="32" cy="4.5" r="3.6" fill="' + cor + '"/>' +
    '<rect x="4" y="29" width="5" height="11" rx="2.5" fill="' + cor + '"/><rect x="55" y="29" width="5" height="11" rx="2.5" fill="' + cor + '"/></svg>';
}

function desenharEvidencias(lista) {
  const caixaDeEvidencias = $("evidencias");
  caixaDeEvidencias.replaceChildren();
  (lista || []).forEach(function (item) {
    const div = document.createElement("div");
    div.className = "evidencia";
    const valor = document.createElement("strong");
    valor.textContent = item.valor;
    const rotulo = document.createElement("span");
    rotulo.textContent = item.rotulo;
    div.append(valor, rotulo);
    caixaDeEvidencias.append(div);
  });
}

// O dialogo do robo. `fala` traz texto, evidencias, fonte e, se houver, figura.
function mostrarFala(robo, sala, fala) {
  $("dialogo").hidden = false;
  $("retrato").innerHTML = rostoDoRobo(robo.cor);
  $("nome-robo").textContent = robo.nome;
  $("papel-robo").textContent = robo.papel + " · " + sala;
  escreverDevagar($("texto-fala"), fala.texto);
  desenharEvidencias(fala.evidencias);
  const fonte = $("fonte");
  fonte.hidden = !fala.fonte;
  fonte.textContent = fala.fonte ? "Fonte: " + fala.fonte : "";
  const figura = $("ver-figura");
  figura.hidden = !fala.figura;
  figura.onclick = function () { ampliarFigura({ figura: fala.figura, titulo: fala.tituloFigura || "Figura citada" }); };
}

function mostrarPerguntas(perguntas, aoEscolher) {
  const caixaDePerguntas = $("perguntas");
  caixaDePerguntas.replaceChildren();
  perguntas.forEach(function (p) {
    const botao = document.createElement("button");
    botao.type = "button";
    botao.className = "pergunta";
    botao.textContent = p.pergunta;
    botao.addEventListener("click", function () {
      caixaDePerguntas.querySelectorAll(".pergunta").forEach(function (b) { b.classList.remove("ativa"); });
      botao.classList.add("ativa");
      aoEscolher(p);
    });
    caixaDePerguntas.append(botao);
  });
}

function fecharDialogo() {
  clearInterval(escreverDevagar.relogio);
  $("dialogo").hidden = true;
}

function ampliarFigura(dados) {
  const imagem = new Image();
  imagem.src = FIGURAS[dados.figura];
  imagem.alt = dados.titulo;
  $("figura-imagem").replaceChildren(imagem);
  $("figura-titulo").textContent = dados.titulo;
  $("figura").showModal();
}

function mostrarDica(texto) {
  const dica = $("dica");
  dica.hidden = !texto;
  dica.textContent = texto || "";
}

function marcarModo(modo) {
  document.querySelectorAll(".modos [data-modo]").forEach(function (b) {
    b.setAttribute("aria-pressed", String(b.dataset.modo === modo));
  });
  document.body.dataset.modo = modo;
}

// Progresso da visita e do caso, com a mesma barra.
function mostrarProgresso(titulo, atual, total) {
  const barra = $("progresso");
  barra.hidden = !titulo;
  if (!titulo) return;
  $("progresso-titulo").textContent = titulo;
  $("progresso-conta").textContent = atual + " de " + total;
  $("progresso-barra").style.transform = "scaleX(" + atual / total + ")";
}

function descricaoDoTipo(tipo) {
  return {
    "verdadeiro positivo": "Tumor maligno, detectado",
    "verdadeiro negativo": "Benigno, sem alarme",
    "falso negativo": "Maligno que passou: o erro mais grave",
    "falso positivo": "Benigno com alarme falso",
  }[tipo];
}

// Escolha do caso: um botao por tipo, porque o interessante e comparar acerto e erro.
function abrirEscolhaDeCaso(casos, aoEscolher) {
  const grupos = L.casosPorTipo(casos);
  const lista = $("lista-casos");
  lista.replaceChildren();
  ["falso negativo", "verdadeiro positivo", "falso positivo", "verdadeiro negativo"].forEach(function (tipo) {
    const grupo = grupos[tipo] || [];
    if (!grupo.length) return;
    const botao = document.createElement("button");
    botao.type = "button";
    botao.className = "opcao-caso";
    botao.dataset.tipo = tipo;
    const titulo = document.createElement("strong");
    titulo.textContent = descricaoDoTipo(tipo);
    const conta = document.createElement("span");
    conta.textContent = grupo.length + (grupo.length === 1 ? " caso" : " casos") + " no teste";
    botao.append(titulo, conta);
    botao.addEventListener("click", function () {
      $("escolha-caso").close();
      aoEscolher(grupo[0]);
    });
    lista.append(botao);
  });
  $("escolha-caso").showModal();
}
