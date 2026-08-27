"""Carregamento e caracterizacao da base de imagens.

Escrito para ser **agnostico ao dataset**: funciona com qualquer base
organizada em uma pasta por classe, convencao adotada tanto pelo BUSI
(ultrassom) quanto pelas versoes em JPEG do CBIS-DDSM (mamografia).

    data/raw/images/
        benign/     imagem_001.png ...
        malignant/  imagem_101.png ...
        normal/     imagem_201.png ...

A separacao segue a mesma logica do pipeline tabular: um conjunto de treino, um
de validacao (usado durante o ajuste) e um de teste que so e tocado na avaliacao
final.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

EXTENSOES = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

# Nomes de pasta tratados como classe positiva, qualquer que seja o dataset.
NOMES_POSITIVOS = ("malignant", "maligno", "malign", "cancer", "positive")

# Bases de segmentacao, como o BUSI, distribuem a mascara ao lado da imagem, no
# mesmo diretorio. Mascaras nao sao amostras: se ficarem na pasta, o Keras as
# carrega como se fossem exames e duplica o conjunto com imagens em preto e
# branco que nao existem na realidade clinica.
MARCADORES_DE_MASCARA = ("_mask", "_gt", "_segmentation")


class ImageDatasetError(RuntimeError):
    """Erro levantado quando a base de imagens nao esta no formato esperado."""


def discover_classes(root: Path | None = None) -> list[str]:
    """Lista as classes a partir dos nomes das subpastas, em ordem alfabetica."""
    diretorio = root or config.IMAGES_DIR
    if not diretorio.exists():
        raise ImageDatasetError(
            f"Base de imagens não encontrada em {diretorio}. "
            "Baixe o dataset e organize em uma pasta por classe."
        )

    classes = sorted(p.name for p in diretorio.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ImageDatasetError(
            f"Esperadas ao menos 2 classes em {diretorio}, encontradas: {classes}."
        )
    return classes


def is_mask(arquivo: Path) -> bool:
    """Identifica mascaras de segmentacao pelo nome do arquivo."""
    nome = arquivo.stem.lower()
    return any(marcador in nome for marcador in MARCADORES_DE_MASCARA)


def find_masks(root: Path | None = None) -> list[Path]:
    """Lista as mascaras que porventura estejam misturadas as imagens."""
    diretorio = root or config.IMAGES_DIR
    return [
        arquivo
        for arquivo in diretorio.rglob("*")
        if arquivo.suffix.lower() in EXTENSOES and is_mask(arquivo)
    ]


def count_images(root: Path | None = None) -> pd.DataFrame:
    """Contagem e proporcao de imagens por classe, ignorando mascaras."""
    diretorio = root or config.IMAGES_DIR
    linhas = []

    for classe in discover_classes(diretorio):
        arquivos = [
            arquivo
            for arquivo in (diretorio / classe).rglob("*")
            if arquivo.suffix.lower() in EXTENSOES and not is_mask(arquivo)
        ]
        if not arquivos:
            raise ImageDatasetError(f"Classe '{classe}' não contém imagens.")
        linhas.append({"classe": classe, "imagens": len(arquivos)})

    tabela = pd.DataFrame(linhas)
    tabela["proporcao"] = (tabela["imagens"] / tabela["imagens"].sum()).round(4)
    return tabela.sort_values("imagens", ascending=False).reset_index(drop=True)


def positive_index(class_names: list[str]) -> int:
    """Indice da classe positiva (maligno), pelo nome da pasta.

    Cai para a ultima classe quando nenhum nome conhecido e encontrado, e o
    chamador deve conferir — o recall so faz sentido se a classe positiva
    estiver correta.
    """
    for indice, nome in enumerate(class_names):
        if nome.strip().lower() in NOMES_POSITIVOS:
            return indice
    return len(class_names) - 1


def class_weights(counts: pd.DataFrame, class_names: list[str]) -> dict[int, float]:
    """Pesos inversamente proporcionais a frequencia de cada classe.

    Bases de imagens medicas costumam ser desbalanceadas. Sem correcao, a rede
    aprende a favorecer a classe majoritaria — justamente o oposto do que
    interessa quando a classe rara e a maligna.
    """
    por_classe = counts.set_index("classe")["imagens"]
    total = int(por_classe.sum())
    n_classes = len(class_names)

    return {
        indice: round(total / (n_classes * int(por_classe[nome])), 4)
        for indice, nome in enumerate(class_names)
    }


def list_images(root: Path | None = None) -> tuple[list[str], list[int], list[str]]:
    """Lista os caminhos das imagens com o indice da classe de cada uma.

    Returns:
        `(caminhos, rotulos, class_names)`, com os caminhos em ordem estavel
        para que a particao dependa apenas da semente.
    """
    diretorio = root or config.IMAGES_DIR
    class_names = discover_classes(diretorio)

    caminhos, rotulos = [], []
    for indice, classe in enumerate(class_names):
        arquivos = sorted(
            arquivo
            for arquivo in (diretorio / classe).rglob("*")
            if arquivo.suffix.lower() in EXTENSOES and not is_mask(arquivo)
        )
        caminhos.extend(str(arquivo) for arquivo in arquivos)
        rotulos.extend([indice] * len(arquivos))

    return caminhos, rotulos, class_names


def stratified_split(
    root: Path | None = None,
    validation_split: float = config.VALIDATION_SPLIT,
    seed: int = config.SEED,
) -> tuple[dict[str, tuple[list[str], list[int]]], list[str]]:
    """Divide as imagens em treino, validacao e teste preservando as classes.

    O `image_dataset_from_directory` do Keras divide **aleatoriamente**, sem
    estratificar. Em uma base pequena e desbalanceada como esta, isso deixa a
    proporcao de casos malignos no teste ao acaso — inconsistente com o rigor
    aplicado ao pipeline tabular. Aqui a particao e feita sobre a lista de
    arquivos, com `stratify`, e o restante e dividido meio a meio entre
    validacao e teste, tambem estratificado.
    """
    from sklearn.model_selection import train_test_split

    caminhos, rotulos, class_names = list_images(root)

    treino_x, restante_x, treino_y, restante_y = train_test_split(
        caminhos,
        rotulos,
        test_size=validation_split,
        random_state=seed,
        stratify=rotulos,
    )
    validacao_x, teste_x, validacao_y, teste_y = train_test_split(
        restante_x,
        restante_y,
        test_size=0.5,
        random_state=seed,
        stratify=restante_y,
    )

    particoes = {
        "treino": (treino_x, treino_y),
        "validacao": (validacao_x, validacao_y),
        "teste": (teste_x, teste_y),
    }
    return particoes, class_names


def split_summary(
    particoes: dict[str, tuple[list[str], list[int]]],
    class_names: list[str],
) -> pd.DataFrame:
    """Composicao de cada conjunto, para comprovar que a estratificacao valeu."""
    positiva = positive_index(class_names)
    linhas = []

    for nome, (_, rotulos) in particoes.items():
        linha = {"conjunto": nome, "imagens": len(rotulos)}
        for indice, classe in enumerate(class_names):
            linha[classe] = sum(1 for r in rotulos if r == indice)
        linha["proporcao_positiva"] = round(
            sum(1 for r in rotulos if r == positiva) / len(rotulos), 4
        )
        linhas.append(linha)

    return pd.DataFrame(linhas)


def _build_dataset(
    caminhos: list[str],
    rotulos: list[int],
    n_classes: int,
    image_size: tuple[int, int],
    batch_size: int,
    embaralhar: bool,
    seed: int,
):
    """Constroi um `tf.data.Dataset` a partir de caminhos e rotulos.

    As imagens saem em escala 0-255, igual ao que o
    `image_dataset_from_directory` entrega: a normalizacao acontece dentro dos
    modelos, em `src/vision/model.py`.
    """
    import tensorflow as tf

    if n_classes == 2:
        alvos = tf.constant(rotulos, dtype=tf.float32)[:, None]
    else:
        alvos = tf.one_hot(rotulos, depth=n_classes)

    conjunto = tf.data.Dataset.from_tensor_slices((tf.constant(caminhos), alvos))

    def carregar(caminho, alvo):
        imagem = tf.io.read_file(caminho)
        imagem = tf.image.decode_image(imagem, channels=3, expand_animations=False)
        imagem = tf.image.resize(imagem, image_size)
        return tf.cast(imagem, tf.float32), alvo

    if embaralhar:
        conjunto = conjunto.shuffle(len(caminhos), seed=seed, reshuffle_each_iteration=True)

    autotune = tf.data.AUTOTUNE
    return conjunto.map(carregar, num_parallel_calls=autotune).batch(batch_size).cache().prefetch(autotune)


def load_datasets(
    root: Path | None = None,
    image_size: tuple[int, int] = config.IMAGE_SIZE,
    batch_size: int = config.BATCH_SIZE,
    validation_split: float = config.VALIDATION_SPLIT,
    seed: int = config.SEED,
):
    """Monta os conjuntos de treino, validacao e teste, estratificados.

    Returns:
        `(treino, validacao, teste, class_names)`.
    """
    diretorio = root or config.IMAGES_DIR

    # O Keras carrega tudo o que encontra na pasta: uma mascara esquecida vira
    # amostra de treino. Melhor falhar aqui, com instrucao clara, do que treinar
    # com um conjunto silenciosamente contaminado.
    mascaras = find_masks(diretorio)
    if mascaras:
        raise ImageDatasetError(
            f"{len(mascaras)} máscaras de segmentação encontradas em {diretorio} "
            f"(por exemplo, {mascaras[0].name}). Remova-as das pastas de classe: "
            "elas seriam carregadas como se fossem exames."
        )

    particoes, class_names = stratified_split(diretorio, validation_split, seed)
    n_classes = len(class_names)

    conjuntos = [
        _build_dataset(
            caminhos,
            rotulos,
            n_classes,
            image_size,
            batch_size,
            embaralhar=(nome == "treino"),
            seed=seed,
        )
        for nome, (caminhos, rotulos) in particoes.items()
    ]
    return (*conjuntos, class_names)


def dataset_summary(root: Path | None = None) -> dict:
    """Resumo da base para o relatorio: classes, contagens e classe positiva."""
    diretorio = root or config.IMAGES_DIR
    contagens = count_images(diretorio)
    classes = discover_classes(diretorio)

    return {
        "diretorio": str(diretorio),
        "classes": classes,
        "imagens": int(contagens["imagens"].sum()),
        "por_classe": contagens.to_dict(orient="records"),
        "classe_positiva": classes[positive_index(classes)],
    }
