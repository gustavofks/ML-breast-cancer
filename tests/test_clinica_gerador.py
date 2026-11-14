"""Testes do gerador da clinica ficticia."""

from datetime import date

from src import config
from src.clinica import gerador
from src.clinica.paaf import CasoPaaf


def _casos(n):
    return [CasoPaaf(caso=i, medidas={"area_mean": 500.0 + i}, referencia="Benigno")
            for i in range(n)]


def test_mesma_semente_gera_a_mesma_clinica():
    a = gerador.gerar(n_pacientes=12, casos_paaf=_casos(5), semente=42)
    b = gerador.gerar(n_pacientes=12, casos_paaf=_casos(5), semente=42)

    assert a == b


def test_cada_caso_paaf_vai_para_exatamente_uma_paciente():
    clinica = gerador.gerar(n_pacientes=12, casos_paaf=_casos(5), semente=42)

    paaf = [e for e in clinica.exames if e.tipo == "PAAF"]
    assert sorted(e.caso_paaf for e in paaf) == list(range(5))
    assert len({e.paciente_id for e in paaf}) == 5


def test_laudo_da_paaf_fica_pendente_e_sem_a_referencia():
    """O prontuario nao pode trazer a resposta que o fluxo vai triar."""
    clinica = gerador.gerar(n_pacientes=12, casos_paaf=_casos(5), semente=42)

    for exame in (e for e in clinica.exames if e.tipo == "PAAF"):
        assert exame.resultado is None
        assert exame.medidas is not None
        assert "Benigno" not in repr(exame) and "Maligno" not in repr(exame)


def test_evolucao_carrega_identificadores_no_texto_livre():
    """Os dados nascem identificaveis, para a anonimizacao ser demonstrada."""
    clinica = gerador.gerar(n_pacientes=3, casos_paaf=[], semente=42)

    for paciente in clinica.pacientes:
        textos = " ".join(ev.texto for ev in clinica.evolucoes
                          if ev.paciente_id == paciente.id)
        assert paciente.nome in textos
        assert paciente.cpf in textos
        assert paciente.telefone in textos


def test_datas_de_exame_nao_passam_de_hoje_da_clinica():
    clinica = gerador.gerar(n_pacientes=20, casos_paaf=_casos(5), semente=42)
    hoje = date.fromisoformat(config.CLINICA_HOJE)

    for exame in clinica.exames:
        assert exame.solicitado_em <= hoje
        assert exame.resultado_em is None or exame.solicitado_em <= exame.resultado_em <= hoje


def test_ha_pendencias_vencidas_e_exames_recentes_sem_resultado():
    """O fluxo precisa de casos dos dois lados do prazo para ser testado."""
    clinica = gerador.gerar(n_pacientes=40, casos_paaf=_casos(5), semente=42)
    hoje = date.fromisoformat(config.CLINICA_HOJE)
    sem_resultado = [e for e in clinica.exames if e.resultado is None]
    idades = [(hoje - e.solicitado_em).days for e in sem_resultado]

    assert any(d > config.CLINICA_PRAZO_PENDENCIA_DIAS for d in idades)
    assert any(d <= config.CLINICA_PRAZO_PENDENCIA_DIAS for d in idades)
