"""Versao estatica do painel, publicada como Space estatico no Hugging Face.

O Hugging Face passou a cobrar por Spaces que executam codigo (SDK Gradio e
Docker); so o Space estatico continua gratuito para conta pessoal. Esta versao
congela, numa pagina HTML, o que a API responde para cada caso do conjunto de
teste.

**Os numeros vem da propria API**, consultada em processo na hora de montar a
pagina: probabilidade, limiar, contribuicoes e texto sao a resposta de
`/explain`, nao uma conta paralela. A pagina mostra exatamente o que a API
servida mostraria. A API completa continua no container do projeto.

A pagina nao carrega nada de fora: abre sem servidor, como o relatorio HTML da
Fase 1. Os dados entram num bloco JSON, e todo texto e inserido na tela por
`textContent`, nunca como HTML.
"""

from __future__ import annotations

import json
from typing import Any

from src.ui.app import AVISO, casos_de_teste, rotulo_do_caso


def dados_dos_casos(api) -> dict[str, Any]:
    """O que a API responde para cada caso do teste, mais a ficha do modelo."""
    casos = casos_de_teste()
    medidas = casos.drop(columns="diagnostico_registrado")
    info = api.get("/model-info").json()

    registros = []
    for indice in range(len(casos)):
        resposta = api.post("/explain", json={"medidas": medidas.iloc[indice].to_dict()})
        resposta.raise_for_status()
        corpo = resposta.json()
        registros.append(
            {
                "rotulo": rotulo_do_caso(casos, indice),
                "diagnostico_registrado": casos["diagnostico_registrado"].iloc[indice],
                "probabilidade": corpo["probabilidade"],
                "limiar": corpo["limiar"],
                "classe": corpo["classe"],
                "contribuicoes": corpo["contribuicoes"],
                "texto": corpo["texto"],
                "texto_motivo": corpo["texto_motivo"],
            }
        )

    return {
        "modelo": {
            "origem": info["origem"],
            "features": info["features"],
            "limiar": info["limiar"],
            "commit": info["commit"],
        },
        "casos": registros,
    }


def _json_seguro(dados: dict[str, Any]) -> str:
    """JSON que nao fecha o bloco `<script>` antes da hora.

    `</` vira `<\\/`, que o JSON le como a mesma barra: um texto contendo
    `</script>` deixaria de encerrar o bloco e de injetar marcacao na pagina.
    """
    return json.dumps(dados, ensure_ascii=False).replace("</", "<\\/")


def _aviso_em_texto() -> str:
    return AVISO.replace("**", "")


def pagina(dados: dict[str, Any]) -> str:
    """A pagina inteira: estilo, dados e script, sem nenhum recurso externo."""
    return (
        _MODELO.replace("__AVISO__", _aviso_em_texto())
        .replace("__DADOS__", _json_seguro(dados))
    )


_MODELO = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apoio ao diagnóstico de câncer de mama</title>
<style>
  :root {
    --fundo: #f7f7f5; --cartao: #ffffff; --tinta: #1a1a19; --suave: #6b6b68;
    --borda: #e2e2de; --maligno: #eb6834; --benigno: #2a78d6; --aviso: #fff4e5;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --fundo: #151514; --cartao: #1f1f1d; --tinta: #ececea; --suave: #a3a39f;
      --borda: #33332f; --aviso: #2e2618;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--fundo); color: var(--tinta);
    font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  main { max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 1.6rem; margin: 0 0 12px; }
  .aviso {
    background: var(--aviso); border: 1px solid var(--borda); border-radius: 10px;
    padding: 12px 16px; margin-bottom: 20px;
  }
  label { display: block; font-weight: 600; margin-bottom: 6px; }
  select {
    width: 100%; padding: 10px 12px; font: inherit; color: var(--tinta);
    background: var(--cartao); border: 1px solid var(--borda); border-radius: 8px;
  }
  .cartoes {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px; margin: 20px 0;
  }
  .cartao {
    background: var(--cartao); border: 1px solid var(--borda); border-radius: 10px;
    padding: 12px 14px;
  }
  .cartao .rotulo { color: var(--suave); font-size: .85rem; }
  .cartao .valor { font-size: 1.25rem; font-weight: 650; }
  .maligno { color: var(--maligno); }
  .benigno { color: var(--benigno); }
  section {
    background: var(--cartao); border: 1px solid var(--borda); border-radius: 10px;
    padding: 16px; margin-bottom: 16px;
  }
  section h2 { font-size: 1.05rem; margin: 0 0 10px; }
  .barra { display: grid; grid-template-columns: 170px 1fr; align-items: center; gap: 10px; margin: 6px 0; }
  .barra .nome { font-size: .9rem; overflow-wrap: anywhere; }
  .trilho { position: relative; height: 22px; }
  .trilho .zero { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--suave); }
  .trilho .preenchido { position: absolute; top: 3px; height: 16px; border-radius: 3px; }
  .legenda { color: var(--suave); font-size: .85rem; margin-top: 8px; }
  #texto h3 { font-size: 1rem; margin: 14px 0 4px; }
  #texto p { margin: 0 0 8px; }
  footer { color: var(--suave); font-size: .85rem; margin-top: 24px; }
  @media (max-width: 520px) { .barra { grid-template-columns: 1fr; gap: 2px; } }
</style>
</head>
<body>
<main>
  <h1>Apoio ao diagnóstico de câncer de mama</h1>
  <div class="aviso" role="note">__AVISO__</div>

  <label for="caso">Caso do conjunto de teste</label>
  <select id="caso"></select>

  <div class="cartoes">
    <div class="cartao"><div class="rotulo">Previsão do modelo</div><div class="valor" id="classe"></div></div>
    <div class="cartao"><div class="rotulo">Probabilidade de malignidade</div><div class="valor" id="probabilidade"></div></div>
    <div class="cartao"><div class="rotulo">Limiar de decisão</div><div class="valor" id="limiar"></div></div>
    <div class="cartao"><div class="rotulo">Diagnóstico registrado na base</div><div class="valor" id="registrado"></div></div>
  </div>

  <section>
    <h2>Medidas que mais pesaram neste caso</h2>
    <div id="barras"></div>
    <div class="legenda">Laranja empurra a previsão para maligno; azul, para benigno.</div>
  </section>

  <section>
    <h2>Explicação em linguagem natural</h2>
    <div id="texto"></div>
  </section>

  <footer id="rodape"></footer>
</main>

<script id="dados" type="application/json">__DADOS__</script>
<script>
(function () {
  var dados = JSON.parse(document.getElementById("dados").textContent);
  var pct = new Intl.NumberFormat("pt-BR", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
  var num = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 4 });
  var seletor = document.getElementById("caso");

  dados.casos.forEach(function (caso, i) {
    var opcao = document.createElement("option");
    opcao.value = String(i);
    opcao.textContent = caso.rotulo;
    seletor.appendChild(opcao);
  });

  function marcar(elemento, texto) {
    elemento.textContent = texto;
    elemento.className = "valor " + (texto === "Maligno" ? "maligno" : "benigno");
  }

  function barras(contribuicoes) {
    var alvo = document.getElementById("barras");
    alvo.textContent = "";
    var maiores = contribuicoes.slice(0, 10);
    var escala = Math.max.apply(null, maiores.map(function (c) { return Math.abs(c.contribuicao); })) || 1;
    maiores.forEach(function (c) {
      var linha = document.createElement("div");
      linha.className = "barra";
      var nome = document.createElement("div");
      nome.className = "nome";
      nome.textContent = c.feature + " (" + num.format(c.contribuicao) + ")";
      var trilho = document.createElement("div");
      trilho.className = "trilho";
      var zero = document.createElement("div");
      zero.className = "zero";
      var preenchido = document.createElement("div");
      preenchido.className = "preenchido";
      var largura = 50 * Math.abs(c.contribuicao) / escala;
      preenchido.style.width = largura + "%";
      preenchido.style.left = (c.contribuicao >= 0 ? 50 : 50 - largura) + "%";
      preenchido.style.background = c.contribuicao > 0 ? "var(--maligno)" : "var(--benigno)";
      trilho.appendChild(zero);
      trilho.appendChild(preenchido);
      linha.appendChild(nome);
      linha.appendChild(trilho);
      alvo.appendChild(linha);
    });
  }

  function texto(caso) {
    var alvo = document.getElementById("texto");
    alvo.textContent = "";
    if (!caso.texto) {
      var falta = document.createElement("p");
      falta.textContent = "Texto em linguagem natural indisponível: " + caso.texto_motivo + ".";
      alvo.appendChild(falta);
      return;
    }
    caso.texto.split("\\n").forEach(function (linha) {
      linha = linha.trim();
      if (!linha) return;
      var bloco = document.createElement(linha.indexOf("## ") === 0 ? "h3" : "p");
      bloco.textContent = linha.replace(/^##\\s+/, "");
      alvo.appendChild(bloco);
    });
  }

  function mostrar(i) {
    var caso = dados.casos[i];
    marcar(document.getElementById("classe"), caso.classe);
    document.getElementById("probabilidade").textContent = pct.format(caso.probabilidade);
    document.getElementById("limiar").textContent = pct.format(caso.limiar);
    marcar(document.getElementById("registrado"), caso.diagnostico_registrado);
    barras(caso.contribuicoes);
    texto(caso);
  }

  document.getElementById("rodape").textContent =
    "Versão estática: os números e textos foram gerados pela API do projeto e congelados " +
    "nesta página. Modelo servido: " + dados.modelo.origem + "; " +
    dados.modelo.features.length + " medidas; commit " + dados.modelo.commit +
    ". A API completa, com previsão para medidas novas, roda no container do projeto.";

  seletor.addEventListener("change", function () { mostrar(Number(seletor.value)); });
  mostrar(0);
})();
</script>
</body>
</html>
"""
