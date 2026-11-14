"""Gera, com o modelo professor, os textos do hospital ficticio.

Roda no WSL com o professor servido (scripts/wsl/servir_professor.sh). Cada
pedido fica em cache pelo hash das mensagens e dos parametros: a geracao pode
ser interrompida e retomada, e rodar de novo nao gasta nada.

    python scripts/wsl/gerar_textos.py            # tudo
    python scripts/wsl/gerar_textos.py --etapa protocolos
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src import config  # noqa: E402
from src.assistente.modelo import ServidorCompativelOpenAI  # noqa: E402
from src.clinica import curadoria  # noqa: E402
from src.corpus import fontes, professor, protocolos  # noqa: E402
from src.llm.provider import GeracaoIncompleta  # noqa: E402

ORIGEM = config.FASE3_ORIGEM_PROFESSOR
GERADO = config.CLINICA_GERADO_DIR
CACHE = config.FASE3_CACHE_PROFESSOR_DIR
LOTE = 10


def pedir(cliente, mensagens, *, max_tokens, temperatura=0.0, semente=config.SEED) -> str:
    chave = hashlib.sha256(json.dumps([mensagens, max_tokens, temperatura, semente],
                                      ensure_ascii=False).encode()).hexdigest()
    arquivo = CACHE / f"{chave}.json"
    if arquivo.exists():
        return json.loads(arquivo.read_text(encoding="utf-8"))["texto"]
    texto = cliente.conversar(mensagens, max_tokens=max_tokens, temperatura=temperatura, semente=semente)
    CACHE.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps({"texto": texto}, ensure_ascii=False), encoding="utf-8")
    return texto


def etapa_fontes() -> None:
    config.FONTES_DIR.mkdir(parents=True, exist_ok=True)
    registro = {}
    for chave, fonte in fontes.FONTES.items():
        destino = config.FONTES_DIR / fonte.arquivo
        if not destino.exists():
            subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0", "-o", str(destino), fonte.url], check=True)
        registro[chave] = {"titulo": fonte.titulo, "url": fonte.url, "sha256": fontes.sha256(destino)}
    GERADO.mkdir(parents=True, exist_ok=True)
    (GERADO / "fontes.json").write_text(json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8")


def _protocolo(cliente, tema, texto_fonte, versao) -> tuple[protocolos.Protocolo, list[dict]]:
    texto = pedir(cliente, professor.pedido_protocolo(tema.titulo, texto_fonte), max_tokens=3000)
    apontadas = professor.ler_lista_json(
        pedir(cliente, professor.pedido_verificacao(texto, texto_fonte), max_tokens=2000),
        campos=("frase",))
    removidas = []
    linhas = []
    for linha in texto.splitlines():
        motivo = None
        sem_base = curadoria.numeros_sem_base(linha, texto_fonte)
        if not linha.startswith("#") and sem_base:
            motivo = f"numero sem base: {sorted(sem_base)}"
        elif (any(a["frase"].strip() and a["frase"].strip() in linha for a in apontadas)
              and curadoria.suporte_lexical(linha, texto_fonte) < config.FASE3_SUPORTE_MINIMO):
            motivo = "apontada pelo professor e sem suporte na fonte"
        elif curadoria.tem_posologia(linha):
            motivo = "posologia"
        if motivo:
            removidas.append({"protocolo": tema.id, "versao": versao, "linha": linha, "motivo": motivo})
        else:
            linhas.append(linha)
    fonte_desc = ("config.py" if tema.fonte == "interno" else
                  f"{tema.fonte} p. " + ", ".join(f"{a}-{b}" for a, b in tema.paginas))
    papel = "inedito" if (tema.papel == "inedito" or versao == 2) else "treino"
    return protocolos.Protocolo(tema.id, versao, tema.titulo, fonte_desc, papel,
                                "\n".join(linhas).strip() + "\n"), removidas


def etapa_protocolos(cliente) -> None:
    removidas = []
    for tema in fontes.TEMAS:
        versoes = (1, 2) if tema.fonte == "interno" else (1,)
        for versao in versoes:
            texto_fonte = (fontes.fatos_internos(versao) if tema.fonte == "interno"
                           else fontes.extrair(tema))
            protocolo, fora = _protocolo(cliente, tema, texto_fonte, versao)
            protocolos.salvar(protocolo, config.CLINICA_PROTOCOLOS_DIR)
            removidas += fora
            print(f"{tema.id} v{versao}: {len(fora)} linhas removidas", flush=True)
    _gravar(GERADO / "protocolos_removido.jsonl", removidas)


def _gravar(caminho: Path, itens: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="\n") as saida:
        for item in itens:
            saida.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _perguntas(cliente, protocolo, n, semente_base) -> list[dict]:
    lista = protocolos.trechos(protocolo)
    ids = [t.id for t in lista]
    bloco = protocolos.bloco_de_trechos(lista)
    itens = []
    for lote in range(0, n, LOTE):
        texto = pedir(cliente, professor.pedido_perguntas(bloco, n=min(LOTE, n - lote), ids=ids),
                      max_tokens=6000, temperatura=0.7, semente=semente_base + lote)
        for item in professor.ler_lista_json(texto, campos=("pergunta", "resposta")):
            itens.append({**item, "protocolo": protocolo.id, "versao": protocolo.versao,
                          "ids_recebidos": ids, "bloco": bloco, "origem": ORIGEM})
    return itens


def etapa_perguntas(cliente) -> None:
    todos = protocolos.carregar(config.CLINICA_PROTOCOLOS_DIR)
    treino, teste = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        tarefas_treino = [executor.submit(_perguntas, cliente, p,
                                          config.FASE3_PERGUNTAS_TREINO_POR_PROTOCOLO, 1000 + i * 100)
                          for i, p in enumerate(todos) if p.papel == "treino"]
        tarefas_teste = [executor.submit(_perguntas, cliente, p,
                                         config.FASE3_PERGUNTAS_TESTE_POR_PROTOCOLO, 9000 + i * 100)
                         for i, p in enumerate(todos)]
        for tarefa in tarefas_treino:
            treino += tarefa.result()
        for tarefa in tarefas_teste:
            teste += tarefa.result()
    _gravar(GERADO / "perguntas_treino.jsonl", treino)
    _gravar(GERADO / "perguntas_teste.jsonl", teste)
    print(f"perguntas: {len(treino)} treino, {len(teste)} teste", flush=True)


def _lotes(cliente, fabrica, total, semente_base) -> list[dict]:
    itens = []
    for lote in range(0, total, LOTE):
        texto = pedir(cliente, fabrica(min(LOTE, total - lote)), max_tokens=6000,
                      temperatura=0.7, semente=semente_base + lote)
        itens += [{**item, "protocolo": None, "origem": ORIGEM}
                  for item in professor.ler_lista_json(texto, campos=("pergunta", "resposta"))]
    return itens


def etapa_avulsos(cliente) -> None:
    for nome, fabrica, total, semente in (
        ("recusas", professor.pedido_recusas, config.FASE3_N_RECUSAS, 20000),
        ("fora_do_escopo", professor.pedido_fora_do_escopo, config.FASE3_N_FORA_DO_ESCOPO, 30000),
        ("gerais", professor.pedido_gerais, config.FASE3_N_GERAIS, 40000),
    ):
        itens = _lotes(cliente, fabrica, total, semente)
        _gravar(GERADO / f"{nome}.jsonl", itens)
        print(f"{nome}: {len(itens)}", flush=True)


def etapa_nao_encontrado(cliente) -> None:
    todos = [p for p in protocolos.carregar(config.CLINICA_PROTOCOLOS_DIR) if p.papel == "treino"]
    perguntas = [json.loads(linha) for linha in
                 (GERADO / "perguntas_treino.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(config.SEED)
    escolhidas = rng.sample(perguntas, min(config.FASE3_N_NAO_ENCONTRADO, len(perguntas)))
    itens = []
    for item in escolhidas:
        outro = rng.choice([p for p in todos if p.id != item["protocolo"]])
        lista = protocolos.trechos(outro)
        bloco = protocolos.bloco_de_trechos(lista)
        try:
            resposta = pedir(cliente, professor.pedido_nao_encontrado(item["pergunta"], bloco),
                             max_tokens=800)
        except GeracaoIncompleta:
            continue
        itens.append({"pergunta": item["pergunta"], "resposta": resposta.strip(),
                      "protocolo": item["protocolo"], "ids_recebidos": [t.id for t in lista],
                      "bloco": bloco, "origem": ORIGEM})
    _gravar(GERADO / "nao_encontrado.jsonl", itens)
    print(f"nao_encontrado: {len(itens)}", flush=True)


ETAPAS = ("fontes", "protocolos", "perguntas", "avulsos", "nao_encontrado")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--etapa", choices=ETAPAS + ("tudo",), default="tudo")
    args = parser.parse_args()
    cliente = ServidorCompativelOpenAI(config.FASE3_PROFESSOR_URL, nome="professor")
    for etapa in ETAPAS:
        if args.etapa not in (etapa, "tudo"):
            continue
        if etapa == "fontes":
            etapa_fontes()
        else:
            globals()[f"etapa_{etapa}"](cliente)


if __name__ == "__main__":
    main()
