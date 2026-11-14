"""Gerador da clinica ficticia, com identificadores falsos de proposito.

Os dados nascem com nome, CPF, telefone, e-mail e endereco no texto livre,
como num prontuario de verdade. So assim a anonimizacao pode ser demonstrada e
testada: anonimizar dados que ja nasceram anonimos nao prova nada.

Nenhum texto daqui entra no treino do modelo (secao 4.2 do spec): os textos
clinicos do treino vem do modelo professor. Os prontuarios servem ao fluxo e a
avaliacao. Por isso cada registro carrega a origem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from faker import Faker

from src import config
from src.clinica.paaf import CasoPaaf

ORIGEM = "gerador-faker"

_BIRADS = [  # categoria, peso: a maioria dos exames de rastreamento e normal
    ("BI-RADS 1", 40), ("BI-RADS 2", 30), ("BI-RADS 3", 12),
    ("BI-RADS 4", 10), ("BI-RADS 5", 3), ("BI-RADS 0", 5),
]
_MOTIVOS = [
    "consulta de rastreamento",
    "avaliacao de nodulo palpavel",
    "retorno com exame de imagem",
    "dor mamaria ciclica",
    "seguimento de achado provavelmente benigno",
]
_HISTORICOS = [
    "Sem historico familiar de cancer de mama.",
    "Mae com cancer de mama aos 52 anos.",
    "Irma com cancer de mama aos 45 anos.",
    "Nega comorbidades.",
    "Hipertensao arterial controlada.",
]


@dataclass(frozen=True)
class Paciente:
    id: str
    nome: str
    cpf: str
    nascimento: date
    telefone: str
    email: str
    endereco: str
    historico: str
    origem: str = ORIGEM


@dataclass(frozen=True)
class Exame:
    paciente_id: str
    tipo: str
    solicitado_em: date
    resultado_em: date | None
    resultado: str | None
    medidas: dict[str, float] | None = None
    caso_paaf: int | None = None
    origem: str = ORIGEM


@dataclass(frozen=True)
class Evolucao:
    paciente_id: str
    data: date
    texto: str
    origem: str = ORIGEM


@dataclass(frozen=True)
class Clinica:
    pacientes: list[Paciente] = field(default_factory=list)
    exames: list[Exame] = field(default_factory=list)
    evolucoes: list[Evolucao] = field(default_factory=list)


def _exame_de_imagem(rng: random.Random, paciente_id: str, tipo: str, hoje: date) -> Exame:
    solicitado = hoje - timedelta(days=rng.randint(1, 120))
    if rng.random() < 0.25:  # um quarto ainda sem resultado: parte vence o prazo
        return Exame(paciente_id, tipo, solicitado, None, None)
    resultado_em = min(hoje, solicitado + timedelta(days=rng.randint(3, 20)))
    categorias, pesos = zip(*_BIRADS)
    return Exame(paciente_id, tipo, solicitado, resultado_em,
                 rng.choices(categorias, weights=pesos)[0])


def gerar(*, n_pacientes: int, casos_paaf: list[CasoPaaf], semente: int) -> Clinica:
    """Clinica deterministica: mesma semente, mesmos dados, byte a byte."""
    if len(casos_paaf) > n_pacientes:
        raise ValueError("Ha mais casos PAAF do que pacientes.")
    rng = random.Random(semente)
    fake = Faker("pt_BR")
    fake.seed_instance(semente)
    hoje = date.fromisoformat(config.CLINICA_HOJE)

    clinica = Clinica()
    for indice in range(n_pacientes):
        pid = f"P{indice:04d}"
        paciente = Paciente(
            id=pid,
            nome=fake.name_female(),
            cpf=fake.cpf(),
            # Relativo ao "hoje" fixo da clinica, e nao ao relogio da maquina: o
            # date_of_birth do Faker usa a data atual e mudaria o banco a cada dia.
            nascimento=hoje - timedelta(days=rng.randint(35 * 366, 80 * 365)),
            telefone=fake.cellphone_number(),
            email=fake.free_email(),
            endereco=fake.address().replace("\n", ", "),
            historico=rng.choice(_HISTORICOS),
        )
        clinica.pacientes.append(paciente)
        clinica.exames.append(_exame_de_imagem(rng, pid, "Mamografia", hoje))
        if rng.random() < 0.5:
            clinica.exames.append(_exame_de_imagem(rng, pid, "Ultrassonografia mamaria", hoje))

        data_consulta = hoje - timedelta(days=rng.randint(0, 60))
        clinica.evolucoes.append(Evolucao(
            pid, data_consulta,
            f"Paciente {paciente.nome}, CPF {paciente.cpf}, nascida em "
            f"{paciente.nascimento:%d/%m/%Y}, comparece para {rng.choice(_MOTIVOS)}. "
            f"Contato: {paciente.telefone}, {paciente.email}. "
            f"Reside em {paciente.endereco}. {paciente.historico}",
        ))

    for caso, paciente in zip(casos_paaf, rng.sample(clinica.pacientes, len(casos_paaf))):
        clinica.exames.append(Exame(
            paciente.id, "PAAF", hoje - timedelta(days=rng.randint(1, 60)),
            None, None, medidas=dict(caso.medidas), caso_paaf=caso.caso,
        ))
    return clinica
