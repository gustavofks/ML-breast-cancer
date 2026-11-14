"""Testes da curadoria: o que barra e o que deixa passar."""

from src.clinica import curadoria


def test_numero_ausente_da_fonte_e_apontado():
    fonte = "Mamografia a cada 2 anos entre 50 e 69 anos."

    assert curadoria.numeros_sem_base("Faixa de 50 a 69 anos, a cada 2 anos.", fonte) == set()
    assert curadoria.numeros_sem_base("Crescimento acima de 20% exige biópsia.", fonte) == {"20"}


def test_numeracao_de_lista_e_ids_de_citacao_nao_contam_como_numero():
    assert curadoria.numeros_sem_base("1. Fazer mamografia [PROT-01 v1 §3].", "Fazer mamografia.") == set()


def test_numero_escrito_por_extenso_na_fonte_conta_como_base():
    assert curadoria.numeros_sem_base("controle em 6 meses", "controle em seis meses") == set()


def test_posologia_e_barrada_e_texto_clinico_comum_nao():
    assert curadoria.tem_posologia("Tomar 500 mg de 8 em 8 horas.")
    assert curadoria.tem_posologia("Prescrevo tamoxifeno 20 mg ao dia.")
    assert not curadoria.tem_posologia("A mamografia deve ser repetida em 6 meses.")


def test_portugues_e_reconhecido_e_ingles_nao():
    assert curadoria.em_portugues("A paciente deve fazer a mamografia de rastreamento.")
    assert not curadoria.em_portugues("The patient should undergo screening mammography.")


def test_quase_duplicatas_sao_reconhecidas():
    assert curadoria.quase_iguais("Qual a idade do rastreamento?", "Qual é a idade do rastreamento?")
    assert not curadoria.quase_iguais("Qual a idade do rastreamento?", "Quando fazer ultrassom?")


def test_citacao_valida_exige_ao_menos_um_id_e_so_ids_recebidos():
    recebidos = {"PROT-01 v1 §1", "PROT-01 v1 §2"}

    assert curadoria.citacoes_validas("Sim [PROT-01 v1 §2].", recebidos)
    assert not curadoria.citacoes_validas("Sim, sem citar.", recebidos)
    assert not curadoria.citacoes_validas("Sim [PROT-09 v1 §1].", recebidos)


def test_deduplicar_mantem_o_primeiro_e_registra_o_motivo():
    itens = [{"pergunta": "Qual a idade do rastreamento?"},
             {"pergunta": "Qual é a idade do rastreamento?"},
             {"pergunta": "Quando fazer ultrassom?"}]

    mantidos, descartados = curadoria.deduplicar(itens, chave="pergunta")

    assert [i["pergunta"] for i in mantidos] == [itens[0]["pergunta"], itens[2]["pergunta"]]
    assert descartados[0]["motivo"] == "quase duplicata"


def test_suporte_lexical_alto_para_parafrase_e_baixo_para_invencao():
    fonte = ("O Ministério da Saúde recomenda contra o rastreamento do câncer de mama com "
             "termografia, seja isoladamente, seja em conjunto com a mamografia.")

    parafrase = "O Ministério da Saúde recomenda contra o rastreamento com termografia isolada."
    invencao = "Pacientes com implantes devem preferir ressonância anual com contraste."

    assert curadoria.suporte_lexical(parafrase, fonte) >= 0.8
    assert curadoria.suporte_lexical(invencao, fonte) < 0.3
