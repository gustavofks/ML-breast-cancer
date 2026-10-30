"""Geracao de explicacoes em linguagem natural a partir das evidencias do modelo.

A camada inteira fala com um `Protocol` (ver `provider.py`), nunca com um cliente
especifico. Duas consequencias praticas: a suite roda sem modelo de linguagem
instalado, e trocar o modelo local e mudar uma linha de configuracao.
"""
