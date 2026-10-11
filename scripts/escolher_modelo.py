"""Grava `results/genetic/modelo_servido.json`, o modelo que a API serve.

Uso, a partir da raiz do projeto:

    python scripts/escolher_modelo.py --braco "AG, só limiar"
    python scripts/escolher_modelo.py --referencia

**A escolha usa so o que a busca viu.** Dentro do braco, vence a semente de maior
fitness em validacao cruzada no treino, o mesmo criterio que
`scripts/evaluate_arms.py` usa para escolher qual execucao avaliar. O conjunto
de teste nunca entra aqui: escolher o modelo pelo numero de teste transformaria
o teste em criterio de selecao, e o numero publicado do modelo escolhido sairia
otimista. Por isso o braco deve ser decidido **antes** de olhar
`comparacao_final.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.api.servico import EspecificacaoDoModelo, referencia_da_fase_1  # noqa: E402


def especificacao_do_braco(payloads: list[dict], braco: str) -> dict:
    """Melhor execucao do braco, por fitness em validacao cruzada, como especificacao."""
    candidatas = [p for p in payloads if p.get("braco", "AG, completo") == braco]
    if not candidatas:
        disponiveis = sorted({p.get("braco", "AG, completo") for p in payloads})
        sys.exit(f"braco {braco!r} nao encontrado; disponiveis: {', '.join(disponiveis)}")

    escolhida = max(candidatas, key=lambda p: p["avaliacao"]["fitness"])
    melhor = escolhida["melhor"]
    return {
        "familia": melhor["familia"],
        "hiperparametros": melhor["hiperparametros"],
        "features": melhor["features"],
        "limiar": melhor["limiar"],
        "origem": (
            f"{braco}, {escolhida['experimento']}, {escolhida['familia']}, "
            f"semente {escolhida['semente']} (escolhido por validação cruzada no treino)"
        ),
    }


def especificacao_da_referencia() -> dict:
    return referencia_da_fase_1().para_dicionario()


def gravar(especificacao: dict, destino: Path = config.MODELO_SERVIDO_FILE) -> None:
    """Valida antes de gravar: arquivo invalido derrubaria a API na inicializacao."""
    EspecificacaoDoModelo.de_dicionario(especificacao)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(especificacao, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Escolha do modelo servido pela API.")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--braco", help='nome do braco, como "AG, completo"')
    grupo.add_argument("--referencia", action="store_true", help="o modelo da Fase 1")
    parser.add_argument("--execucoes", type=Path, default=config.GENETIC_DIR / "execucoes.json")
    parser.add_argument("--saida", type=Path, default=config.MODELO_SERVIDO_FILE)
    argumentos = parser.parse_args()

    if argumentos.referencia:
        especificacao = especificacao_da_referencia()
    else:
        payloads = json.loads(argumentos.execucoes.read_text(encoding="utf-8"))
        especificacao = especificacao_do_braco(payloads, argumentos.braco)

    gravar(especificacao, argumentos.saida)
    print(f"Modelo servido: {especificacao['origem']}")
    print(f"  {len(especificacao['features'])} medidas, limiar {especificacao['limiar']}")
    print(f"Gravado em {argumentos.saida}")


if __name__ == "__main__":
    main()
