---
id: PROT-13
versao: 1
titulo: Gestão de pendências de exames
fonte: config.py
papel: treino
---
# Gestão de pendências de exames
## Objetivo
Estabelecer critérios para a identificação de exames solicitados sem resultado há mais de 30 dias, considerados pendentes, e garantir o envio de alertas para a equipe médica responsável pela paciente, informando o exame, a data da solicitação e o tempo de pendência.

## Público
Equipe médica responsável pela paciente.

## Conduta
1. Classificar como pendente qualquer exame solicitado sem resultado há mais de 30 dias.
2. Gerar alerta para a equipe médica responsável sempre que houver pendência.
3. Informar no alerta o exame, a data da solicitação e o número de dias que o exame está pendente.
4. Classificar PAAF com laudo citopatológico pendente pelo modelo de triagem antes do retorno.
5. Emitir alerta de cautela caso o resultado do modelo de triagem esteja perto do limiar, mesmo sem pendência vencida.

## Observações
1. O resultado do modelo de triagem não substitui o laudo; a decisão é sempre do médico.
