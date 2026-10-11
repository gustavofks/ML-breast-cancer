# Apoio ao diagnóstico de câncer de mama

Tech Challenge da pós-graduação em IA da FIAP, Fase 2. Escolha um caso do
conjunto de teste da base Breast Cancer Wisconsin e veja a probabilidade de
malignidade, o limiar de decisão, as medidas que mais pesaram e uma explicação em
linguagem natural.

**Sistema de apoio à decisão.** A previsão e a explicação ajudam a leitura do
caso, mas o diagnóstico final é sempre do médico responsável.

## Rotas

- `/painel/` — o painel (a raiz redireciona para ele)
- `GET /health` — vivacidade
- `GET /model-info` — modelo servido, medidas, limiar e commit de origem
- `POST /predict` e `POST /explain` — previsão e explicação a partir das medidas
- `GET /metrics` — contadores e latências no formato Prometheus

Os textos em linguagem natural foram gerados uma vez, na máquina de
desenvolvimento, e são lidos de um cache versionado. Este container não fala com
nenhum modelo de linguagem nem com a internet.
