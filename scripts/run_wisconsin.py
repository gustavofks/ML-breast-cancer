"""Executa o pipeline completo do dataset Wisconsin, de ponta a ponta.

Uso, a partir da raiz do projeto:

    python scripts/run_wisconsin.py

Carrega e limpa os dados, separa treino e teste, compara os modelos por
validacao cruzada, ajusta hiperparametros, avalia no conjunto de teste e grava
todas as figuras e metricas em `results/`. E idempotente: rodar de novo
regenera todos os artefatos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # backend sem janela: o script roda em terminal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, eda, evaluation, explain, models, plotting, report  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.preprocessing import split_data, split_summary  # noqa: E402


def main() -> None:
    plotting.apply_style()
    config.ensure_output_dirs()

    print("1/8  Carregando e limpando os dados...")
    X, y = load_dataset()
    print(f"      {X.shape[0]} amostras, {X.shape[1]} features")

    print("2/8  Análise exploratória...")
    eda.plot_class_balance(y)
    eda.plot_feature_distributions(
        X, y, eda.feature_group(X, "mean"),
        "02_distribuicoes_mean.png",
        "Distribuição das medidas médias por diagnóstico",
    )
    eda.plot_boxplots_by_class(
        X, y, eda.feature_group(X, "worst"),
        "03_boxplots_worst.png",
        'Dispersão das medidas "worst" por diagnóstico',
    )
    eda.plot_separation_ranking(X, y)
    eda.plot_correlation_heatmap(X)
    eda.plot_target_correlation(X, y)

    print("3/8  Separando treino e teste (estratificado)...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(split_summary(y_train, y_test).to_string(index=False))

    print("4/8  Validação cruzada e ajuste de hiperparâmetros...")
    cv_results = models.cross_validate_models(models.get_models(), X_train, y_train)
    modelos_ajustados, grade = models.tune_models(models.get_models(), X_train, y_train)
    melhor = models.select_best(cv_results)
    print(f"      Melhor modelo por {models.SELECTION_METRIC}: {melhor}")

    print("5/8  Avaliando no conjunto de teste...")
    resultados_teste = evaluation.evaluate_all(modelos_ajustados, X_test, y_test)
    print(resultados_teste[["modelo", "accuracy", "recall", "f1", "roc_auc", "falsos_negativos"]].to_string(index=False))

    modelo_final = modelos_ajustados[melhor]
    evaluation.plot_confusion_matrix(modelo_final, X_test, y_test, melhor)
    evaluation.plot_roc_curves(modelos_ajustados, X_test, y_test)
    evaluation.plot_metrics_comparison(resultados_teste)
    evaluation.plot_threshold_tradeoff(modelo_final, X_test, y_test, melhor)
    limiares = evaluation.threshold_analysis(modelo_final, X_test, y_test)

    print("6/8  Explicabilidade...")
    coeficientes = explain.linear_coefficients(modelo_final, list(X.columns))
    permutacao = explain.permutation_scores(modelo_final, X_test, y_test)
    explain.plot_coefficients(coeficientes)
    explain.plot_permutation_importance(permutacao)

    explicacao = explain.shap_explanation(modelo_final, X_train, X_test)
    explain.plot_shap_beeswarm(explicacao)
    shap_global = explain.shap_importance(explicacao)

    # Um falso negativo e o caso mais instrutivo para o relatorio: mostra por que
    # o modelo errou justamente onde o erro custa mais caro.
    previsoes = modelo_final.predict(X_test)
    falsos_negativos = np.where((y_test.to_numpy() == 1) & (previsoes == 0))[0]
    caso = int(falsos_negativos[0]) if len(falsos_negativos) else 0
    explain.plot_shap_waterfall(
        explicacao,
        caso,
        f"SHAP — caso {caso} do teste (tumor maligno não detectado)",
        "14_shap_caso_falso_negativo.png",
    )

    print("7/8  Gravando métricas...")
    caminho = evaluation.save_metrics(
        {
            "dataset": {
                "nome": "Breast Cancer Wisconsin (Diagnostic)",
                "amostras": int(X.shape[0]),
                "features": int(X.shape[1]),
                "benignos": int((y == 0).sum()),
                "malignos": int((y == 1).sum()),
                "treino": int(len(y_train)),
                "teste": int(len(y_test)),
            },
            "modelo_escolhido": melhor,
            "metrica_de_selecao": models.SELECTION_METRIC,
            "validacao_cruzada": cv_results.to_dict(orient="records"),
            "hiperparametros": grade.to_dict(orient="records"),
            "teste": resultados_teste.to_dict(orient="records"),
            "limiares": limiares.to_dict(orient="records"),
            "explicabilidade": {
                "coeficientes": coeficientes.head(15).to_dict(orient="records"),
                "permutacao": permutacao.head(15).to_dict(orient="records"),
                "shap_global": shap_global.head(15).to_dict(orient="records"),
                "caso_analisado": caso,
            },
        }
    )
    print(f"      {caminho}")

    print("8/8  Gerando relatório HTML...")
    pagina = report.build_report()
    print(f"      {pagina}")

    print(f"\nConcluído. Figuras em {config.FIGURES_DIR}")
    print(f"Abra o relatório no navegador: {config.REPORT_FILE}")


if __name__ == "__main__":
    main()
