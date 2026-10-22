"""Casos do teste congelados a partir da API, para a versao estatica da vitrine.

O Hugging Face passou a cobrar por Spaces que executam codigo (SDK Gradio e
Docker); so o Space estatico continua gratuito para conta pessoal. Este modulo
congela o que a API responde para cada caso do conjunto de teste; a pagina que
mostra esses dados e a vitrine, em `src/ui/vitrine.py`.

**Os numeros vem da propria API**, consultada em processo na hora de montar a
pagina: probabilidade, limiar, contribuicoes e texto sao a resposta de
`/explain`, nao uma conta paralela. A pagina mostra exatamente o que a API
servida mostraria. A API completa continua no container do projeto.

Os dados entram na pagina num bloco JSON, escapado por `_json_seguro`.
"""

from __future__ import annotations

import json
from typing import Any

from src.ui.app import casos_de_teste, rotulo_do_caso


def dados_dos_casos(api) -> dict[str, Any]:
    """O que a API responde para cada caso do teste, mais a ficha do modelo.

    As medidas do modelo vao junto: a vitrine abre qualquer caso no simulador.
    """
    casos = casos_de_teste()
    medidas = casos.drop(columns="diagnostico_registrado")
    info = api.get("/model-info").json()
    do_modelo = list(info["features"])

    registros = []
    for indice in range(len(casos)):
        resposta = api.post("/explain", json={"medidas": medidas.iloc[indice].to_dict()})
        resposta.raise_for_status()
        corpo = resposta.json()
        registros.append(
            {
                "rotulo": rotulo_do_caso(casos, indice),
                "medidas": {n: float(medidas[n].iloc[indice]) for n in do_modelo},
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
            "familia": info["familia"],
            "hiperparametros": info["hiperparametros"],
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
