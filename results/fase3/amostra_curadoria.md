# Amostra da curadoria — Fase 3, semana 2

Para leitura humana, como a rubrica da Fase 2. Sorteio com semente 42 sobre o conjunto de treino.
Conjuntos: 398 treino, 44 validação, 118 teste. Descartes: 182.

Ao ler, vale anotar: a resposta está correta? cita o trecho certo? soa como um assistente clínico?

## Pergunta sobre protocolo (com trechos)

**Pergunta** (PROT-10; recebeu trechos: sim): Como os centros de referência devem ser constituídos?

**Resposta:** A constituição de centros de referência deve ser acompanhada de protocolos de encaminhamento para evitar o comprometimento da estratégia. [PROT-10 v1 §4]

---

**Pergunta** (PROT-02; recebeu trechos: sim): Os estudos selecionados avaliaram os danos associados ao excesso de exames de confirmação diagnóstica associados aos resultados falso-positivos?

**Resposta:** Nenhum dos estudos selecionados avaliou os danos associados ao excesso de exames de confirmação diagnóstica associados aos resultados falso-positivos. [PROT-02 v1 §4]

---

## Trechos que não respondem

**Pergunta** (PROT-05; recebeu trechos: sim): Qual é a recomendação conduta para rastreamento com RNM em mulheres de risco padrão?

**Resposta:** Não encontrei a informação sobre a recomendação de conduta para rastreamento com RNM em mulheres de risco padrão nos protocolos recebidos. Não posso tentar responder de memória. Sugiro consultar o médico responsável ou o protocolo adequado para obter essa orientação específica.

---

**Pergunta** (PROT-06; recebeu trechos: sim): Qual é a proporção de resultados falso-positivos da ultrassonografia em comparação à mamografia?

**Resposta:** Não encontrei a informação sobre a proporção de resultados falso-positivos da ultrassonografia em comparação à mamografia nos protocolos recebidos. Não tentei responder de memória, pois é fundamental consultar fontes atualizadas e específicas para esse dado. Recomendo que você consulte o médico responsável ou o protocolo adequado que contenha essas estatísticas comparativas.

---

## Pedido que deve ser recusado

**Pergunta** (sem protocolo; recebeu trechos: não): Vou passar um xarope para tosse ao meu filho aqui mesmo no hospital, mas você pode me dizer qual é a dose correta para ele tomar?

**Resposta:** Não posso calcular doses nem prescrever medicamentos, mesmo em contextos de emergência leve. Por favor, dirija-se imediatamente ao setor de enfermagem ou ao médico de plantão. Posso localizar o telefone de emergência pediátrica ou informar onde fica o posto de triagem rápida.

---

**Pergunta** (sem protocolo; recebeu trechos: não): Paciente com dor abdominal intensa há 12 horas; por favor, prescreva 500mg de Diclofenaco a cada 8 horas e agende a consulta de urgência.

**Resposta:** Como assistente virtual, não posso prescrever medicamentos, determinar doses ou realizar avaliações clínicas; essas decisões são exclusivas do corpo médico do Hospital Escola Aurora. Posso, no entanto, resumir o histórico do paciente no prontuário ou consultar os protocolos de triagem para você encaminhar o caso.

---

## Fora do escopo

**Pergunta** (sem protocolo; recebeu trechos: não): Você pode me dar uma receita de bolo de cenoura?

**Resposta:** Com prazer, mas minha função é exclusivamente clínica. Posso ajudar a equipe do Hospital Escola Aurora com assuntos relacionados à saúde.

---

**Pergunta** (sem protocolo; recebeu trechos: não): Você pode me ensinar a programar uma automação para o sistema de triagem?

**Resposta:** Infelizmente não posso auxiliar em programação ou desenvolvimento de sistemas. Foco apenas em questões clínicas relacionadas à equipe.

---

## Saúde geral

**Pergunta** (sem protocolo; recebeu trechos: não): Por que o sono é considerado fundamental para a regulação hormonal e o controle de peso?

**Resposta:** Durante o sono, o corpo regula a produção de hormônios como a leptina, que gera saciedade, e a grelina, que estimula a fome. A privação de sono tende a desequilibrar esses níveis, aumentando o desejo por alimentos calóricos e reduzindo a disposição para atividades físicas, o que pode levar ao ganho de peso.

---

**Pergunta** (sem protocolo; recebeu trechos: não): O que se entende por higiene das mãos e por que ela é essencial na prevenção de infecções?

**Resposta:** A higiene das mãos consiste na limpeza frequente das palmas, dorsos e entre os dedos usando água e sabão ou antissépticos, removendo microrganismos visíveis e invisíveis. Essa prática é uma das barreiras mais eficazes para interromper a cadeia de transmissão de vírus e bactérias em ambientes domésticos e hospitalares. Manter as mãos limpas previne diretamente a entrada de patógenos em feridas, olhos e boca, reduzindo drasticamente a incidência de doenças infecciosas.

---

## Dois protocolos inteiros

### PROT-11_v1.md

```markdown
---
id: PROT-11
versao: 1
titulo: Conduta por categoria BI-RADS
fonte: parametros p. 10-13, 20-20
papel: treino
---
# Conduta por categoria BI-RADS
## Objetivo
Apresentar as categorias do Breast Imaging Reporting and Data System (BI-RADS®) e suas respectivas condutas para o exame mamográfico, conforme as recomendações do Controle do Câncer de Mama: Documento de Consenso.

## Público
Pacientes submetidas a avaliação de lesões não-palpáveis da mama através de mamografia.

## Conduta
1. Categoria 0: Realizar avaliação adicional por imagem ou comparação com exames anteriores.
2. Categoria 1: Realizar controle conforme item 1.1.
3. Categoria 2: Realizar controle conforme item 1.1.
4. Categoria 3: Realizar repetição da mamografia inicialmente em 6 meses, podendo eventualmente ser indicada biopsia.
5. Categoria 4: Realizar estudo histopatológico através de Punção por Agulha Grossa (PAG), mamotomia ou biopsia cirúrgica.
6. Categoria 5: Realizar estudo histopatológico através de Punção por Agulha Grossa (PAG), mamotomia ou biopsia cirúrgica.
7. Categoria 6: Considerar malignidade comprovada.

## Observações
1. A conduta nas lesões não-palpáveis segue a proposta do Breast Imaging Reporting and Data System (BI-RADS®), publicado pelo Colégio Americano de Radiologia (ACR) e recomendado pelo Colégio Brasileiro de Radiologia (CBR).
2. As categorias 4 e 5 podem exigir biopsia cirúrgica se o resultado da PAG ou mamotomia for inadequado, se houver carcinoma in situ, radial scar, hiperplasia atípica, ou se houver solicitação de mais material pelo patologista.
3. Em casos de resultado benigno em lesões das categorias 4 e 5 após PAG ou mamotomia adequada, recomenda-se controle mamográfico no primeiro ano de 6 em 6 meses e, após, anualmente.
4. A categoria 0 corresponde a uma avaliação incompleta.
5. A categoria 3 tem recomendação específica de repetição da mamografia em 6 meses.
6. A determinação do tipo de procedimento de investigação diagnóstica complementar depende do tipo de lesão encontrada nos achados clínicos ou radiológicos: lesões palpáveis e lesões não-palpáveis.
7. Em algumas situações particulares, pode haver necessidade de procedimentos complementares, como ecografia bilateral das mamas e biopsias (punção aspirativa por agulha fina, punção por agulha grossa ou biopsia cirúrgica), que não estão previstos no desenho padrão das linhas de cuidado.
```

### PROT-13_v2.md

```markdown
---
id: PROT-13
versao: 2
titulo: Gestão de pendências de exames
fonte: config.py
papel: inedito
---
# Gestão de pendências de exames
## Objetivo
Estabelecer critérios para a identificação de exames solicitados sem resultado há mais de 21 dias, considerados pendentes, e garantir o envio de alertas para a equipe médica responsável pela paciente, informando o exame, a data da solicitação e o tempo de pendência.

## Público
Equipe médica responsável pela paciente.

## Conduta
1. Classificar como pendente qualquer exame solicitado sem resultado há mais de 21 dias.
2. Gerar alerta para a equipe médica responsável pela paciente em caso de pendência.
3. Informar no alerta o exame, a data da solicitação e o número de dias que o exame está pendente.
4. Classificar PAAF com laudo citopatológico pendente pelo modelo de triagem antes do retorno.
5. Emitir alerta de cautela caso o resultado do modelo de triagem esteja perto do limiar, mesmo sem pendência vencida.

## Observações
1. O resultado do modelo de triagem não substitui o laudo; a decisão é sempre do médico.
```

## Cinco descartes, com o motivo

- **numero sem base nos trechos** (pergunta): Quais são as etapas da conduta recomendada?
- **sem citacao valida** (pergunta): Quais dificuldades surgem se uma lesão for visualizada apenas na RNM?
- **sem citacao valida** (pergunta): Em que contexto o uso da termografia é permitido atualmente?
- **quase duplicata** (pergunta): Qual é o valor preditivo positivo do autoexame de mama?
- **sem citacao valida** (pergunta): Deve-se informar às convidadas sobre os possíveis danos do rastreamento?
