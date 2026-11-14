"""Testes da anonimizacao: propriedades, nao valores."""

import json
import re

from src import config
from src.clinica import anonimizacao, gerador
from src.clinica.paaf import CasoPaaf


def _clinica(n=30):
    casos = [CasoPaaf(caso=i, medidas={"area_mean": 1.0}, referencia="Benigno")
             for i in range(min(5, n))]
    return gerador.gerar(n_pacientes=n, casos_paaf=casos, semente=42)


def _identificadores(paciente):
    digitos_cpf = re.sub(r"\D", "", paciente.cpf)
    digitos_tel = re.sub(r"\D", "", paciente.telefone)
    return [
        paciente.nome, paciente.cpf, digitos_cpf, paciente.telefone, digitos_tel[-8:],
        paciente.email, paciente.endereco, f"{paciente.nascimento:%d/%m/%Y}",
        paciente.nascimento.isoformat(),
    ] + [parte for parte in paciente.nome.split() if len(parte) > 3]


def test_nenhum_identificador_sobrevive_em_nenhum_registro_anonimizado():
    """Na escala real da clinica: mais pacientes, mais chance de um formato de
    telefone ou endereco que a primeira camada nao conhece."""
    clinica = _clinica(config.CLINICA_N_PACIENTES)
    saida = json.dumps(anonimizacao.anonimizar(clinica).registros, ensure_ascii=False)

    for paciente in clinica.pacientes:
        for identificador in _identificadores(paciente):
            assert identificador not in saida, identificador


def test_pseudonimo_e_estavel_e_nao_deriva_do_nome():
    clinica = _clinica()
    a = anonimizacao.anonimizar(clinica)
    b = anonimizacao.anonimizar(clinica)

    assert a.mapa == b.mapa
    assert all(re.fullmatch(r"PACIENTE-\d{4}", p) for p in a.mapa.values())


def test_mapa_de_pseudonimos_nao_entra_nos_registros():
    resultado = anonimizacao.anonimizar(_clinica())

    assert "mapa" not in json.dumps(resultado.registros)
    assert all("P0" not in json.dumps(r) for r in resultado.registros)


def test_idade_substitui_a_data_de_nascimento():
    clinica = _clinica(3)
    registros = anonimizacao.anonimizar(clinica).registros
    pacientes = [r for r in registros if r["tipo"] == "paciente"]

    for registro in pacientes:
        assert "nascimento" not in registro
        assert 35 <= registro["idade"] <= 80


def test_medidas_da_paaf_e_datas_clinicas_sao_preservadas():
    clinica = _clinica()
    exames = [r for r in anonimizacao.anonimizar(clinica).registros if r["tipo"] == "exame"]

    paaf = [e for e in exames if e["exame"] == "PAAF"]
    assert len(paaf) == 5 and all(e["medidas"] == {"area_mean": 1.0} for e in paaf)
    assert all(e["solicitado_em"] <= config.CLINICA_HOJE for e in exames)


def test_texto_livre_ganha_marcadores_no_lugar_dos_identificadores():
    evolucoes = [r for r in anonimizacao.anonimizar(_clinica(3)).registros
                 if r["tipo"] == "evolucao"]

    for evolucao in evolucoes:
        for marcador in ("PACIENTE-", "[CPF]", "[NASCIMENTO]", "[TELEFONE]", "[EMAIL]", "[ENDERECO]"):
            assert marcador in evolucao["texto"]


def test_cpf_e_telefone_soltos_no_texto_sao_removidos_mesmo_de_quem_nao_esta_no_cadastro():
    texto = "Ligou a filha, CPF 123.456.789-09, telefone (11) 98765-4321, email ana@exemplo.com."

    limpo = anonimizacao.limpar_texto(texto)

    assert "123.456.789-09" not in limpo and "98765-4321" not in limpo
    assert "ana@exemplo.com" not in limpo
