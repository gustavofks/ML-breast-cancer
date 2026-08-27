"""Executa o pipeline de diagnostico por imagem, de ponta a ponta.

Uso, a partir da raiz do projeto:

    pip install -r requirements-vision.txt
    python scripts/run_vision.py

Espera a base organizada em uma pasta por classe em `data/raw/images/`
(ver README). Caracteriza a base, treina as duas arquiteturas, avalia no
conjunto de teste, grava figuras e `results/metrics_vision.json` e regenera o
relatorio HTML — cuja segunda aba passa a ser montada a partir dessas metricas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, plotting, report  # noqa: E402
from src.vision import dataset as vdata  # noqa: E402
from src.vision import evaluate as vevaluate  # noqa: E402
from src.vision import model as vmodel  # noqa: E402
from src.vision import train as vtrain  # noqa: E402

DISCUSSAO = (
    "O modelo de imagem parte do pixel cru, sem nenhuma medida extraída por especialista — "
    "é essa a diferença em relação ao pipeline tabular, que depende de um profissional ter "
    "medido os núcleos celulares antes. Em compensação, aprende com muito menos supervisão "
    "estruturada e erra mais. Vale como demonstração de viabilidade, não como sistema clínico."
)


def main(epochs: int) -> None:
    plotting.apply_style()
    config.ensure_output_dirs()

    print("1/5  Caracterizando a base de imagens...")
    resumo = vdata.dataset_summary()
    contagens = vdata.count_images()
    print(contagens.to_string(index=False))
    print(f"      classe positiva: {resumo['classe_positiva']}")

    vevaluate.plot_class_balance(contagens)
    vevaluate.plot_samples(config.IMAGES_DIR, resumo["classes"])

    print("2/5  Montando treino, validação e teste (estratificado)...")
    particoes, class_names = vdata.stratified_split()
    composicao = vdata.split_summary(particoes, class_names)
    print(composicao.to_string(index=False))

    treino, validacao, teste, class_names = vdata.load_datasets()
    pesos = vdata.class_weights(contagens, class_names)
    print(f"      pesos de classe: {pesos}")

    print(f"3/5  Treinando ({epochs} épocas no máximo, com parada antecipada)...")
    modelos = vmodel.get_models(n_classes=len(class_names))
    modelos, historicos, tabela_treino = vtrain.train_all(
        modelos, treino, validacao, class_weight=pesos, epochs=epochs
    )
    print(tabela_treino.to_string(index=False))
    vevaluate.plot_training_curves(historicos)

    print("4/5  Avaliando no conjunto de teste...")
    resultados = vevaluate.evaluate_all(modelos, teste, class_names)
    print(resultados.to_string(index=False))

    melhor = str(resultados.iloc[0]["modelo"])
    print(f"      melhor por recall da classe maligna: {melhor}")

    vevaluate.plot_confusion(modelos[melhor], teste, class_names, melhor)
    perdidos = vevaluate.plot_misclassified_malignant(modelos[melhor], teste, class_names)

    figuras = [
        {
            "arquivo": "20_imagens_por_classe.png",
            "titulo": "Imagens por classe",
            "leitura": "A base é desbalanceada: 56% benignas, 27% malignas e 17% normais. O treino "
            "usa pesos de classe para compensar, evitando que a rede aprenda a favorecer a classe "
            "majoritária.",
        },
        {
            "arquivo": "21_amostras_imagens.png",
            "titulo": "Exemplos de ultrassom por classe",
            "leitura": "Ao contrário do dataset tabular, aqui não há medidas extraídas por um "
            "especialista: a rede parte do pixel cru e precisa aprender sozinha o que distingue "
            "as lesões.",
        },
        {
            "arquivo": "22_curvas_treino.png",
            "titulo": "Curvas de treino",
            "leitura": "A distância entre as curvas de treino e validação mede o sobreajuste. "
            "A parada antecipada devolve os pesos da melhor época de validação.",
        },
        {
            "arquivo": "23_matriz_confusao_imagens.png",
            "titulo": f"Matriz de confusão — {melhor}",
            "leitura": "Com três classes, o erro que importa é o caso maligno classificado como "
            "benigno ou normal: é o equivalente ao falso negativo do pipeline tabular.",
        },
    ]
    if perdidos is not None:
        figuras.append(
            {
                "arquivo": "24_malignos_nao_detectados.png",
                "titulo": "Casos malignos não detectados",
                "leitura": "Mostrar quais imagens escaparam é o análogo visual da análise SHAP do "
                "pipeline tabular: em vez de apenas contar erros, expõe o que o modelo não viu.",
            }
        )

    print("5/5  Gravando métricas e regenerando o relatório...")
    payload = {
        "dataset": {
            "nome": "BUSI — Breast Ultrasound Images",
            **resumo,
        },
        "particoes": composicao.to_dict(orient="records"),
        "modelo_escolhido": melhor,
        "metrica_de_selecao": "recall_maligno",
        "treino": tabela_treino.to_dict(orient="records"),
        "teste": resultados.to_dict(orient="records"),
        "figuras": figuras,
        "discussao": DISCUSSAO,
    }
    config.VISION_METRICS_FILE.write_text(
        __import__("json").dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"      {config.VISION_METRICS_FILE}")

    pagina = report.build_report()
    print(f"      {pagina}")
    print("\nConcluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de diagnóstico por imagem")
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="número máximo de épocas (a parada antecipada pode encerrar antes)",
    )
    main(parser.parse_args().epochs)
