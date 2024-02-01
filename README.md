# Calculo-Numerico

Repositório para o curso de Noções de Cálculo Numérico MAP3122

## RELATÓRIO TÉCNICO - versão 2024.01.06

**RELATÓRIO TÉCNICO (RT) (ATENÇÃO: vale sempre a versão mais recente neste tópico!)**

Essa atividade pode ser feita em dupla, mas não em tripla.

### [A] ESTRUTURA GERAL
* Título
* Autores (nome completo com NUSP, afiliação, endereço e email profissionais)
* Resumo e palavras-chave
* 1. Introdução
* 2. Modelagem Matemática
* 3. Metodologia Numérica
    3.1 Discretização do Problema
    3.2 Técnicas Numéricas
* 4. Resultados
    4.1 Verificação das implementações (use soluções manufaturadas)
    4.2 Aplicação
* 5. Conclusão
* Referências Bibliográficas
* Apêndices (pode conter explicações dos trechos mais delicados dos programas desenvolvidos)

### DETALHES E EXPLICAÇÕES

CUIDADO! Erros de português serão descontados. Use um "spellcheck" para português no Overleaf, por exemplo. Cuidado com "gerundismos" (e.g. "estaremos indo apresentar", ...) pois também haverá descontos.

Na **INTRODUÇÃO**, você deve salientar a importância do problema cuja solução será aproximada numericamente (importância prática, teórica, acadêmico-didática, o que seja). Em geral, comentários sobre abordagens já usadas para resolver o problema também são praxe (abordagens computacionais, teóricas, experimentais, etc). Inclua pequena revisão de literatura que discorra sobre esses aspectos e que ajude a entender o problema, sua relevância e abordagens para resolvê-lo. Cite trabalhos que apoiem suas afirmações. Esta seção termina explicando a organização do trabalho ao leitor ("o que vem a seguir").

Na **MODELAGEM MATEMÁTICA**, você parte de considerações, premissas, que levam à equação diferencial que modela matematicamente o problema, PASSO A PASSO. Em geral, tais considerações se traduzem matematicamente usando as leis, os princípios, que se aplicam à área: leis físicas, biológicas, dados experimentais, econômicos, etc. o que for necessário para se chegar, de maneira lógica e consistente ao Problema de Cauchy que será resolvido computacionalmente. Essa seção termina com o conjunto de EDOs com suas condições iniciais e o domínio de definição do problema. Todo símbolo matemático deve ter sua explicação, vars de estado devem ser acompanhadas de unidades métricas. A dimensionalidade de seu problema deve ser no mínimo dois, ou seja, quando reescrito na forma padrão, o Problema de Cauchy tem no mínimo duas vars de estado.

Na **METODOLOGIA NUMÉRICA**, primeiramente você parte de discretizações do domínio de definição do problema e da equação diferencial para obter o conjunto de equações algébricas que deverá ser resolvido. Para resolver e apresentar a aproximação de tal conjunto de equações você precisará usar técnicas (a serem) vistas na disciplina dentre EDOs, Zeros, Sistemas Lineares, interpolação, quadratura e MMQ. No mínimo três técnicas diferentes devem ser usadas. Subseção 3.1 termina com a apresentação da discretização do domínio de definição e do conjunto de equações algébricas (lineares ou não lineares) a serem resolvidas. Subseção 3.2 termina com um algoritmo que mostra o uso das técnicas da disciplina (onde e quando elas são usadas e como o programa segue seu fluxo rumo ao término).

Em **RESULTADOS**, primeiro mostramos na Subseção 4.1 o uso do programa para um problema similar cuja solução é conhecida (frequentemente obtida por construção, isto é, via estratégia de solução manufaturada). Em geral, gráficos com a solução exata e aproximações para diversos passos de integração são apresentados. Além disso, tabelas de convergência mostrando que as aproximações tendem à solução exata (e com que "velocidade") também fazem parte. Nesta subseção, frequentemente é aceitável mostrar apenas dois tipos de tabelas de convergência: uma usando o erro de discretização global e, outra, usando diferenças relativas entre aproximações sucessivamente mais refinadas (PERGUNTE-ME COMO). Na Subseção 4.2, você aproxima a solução do problema escolhido para um intervalo de definição, parâmetros e condição inicial conhecidos. Faz parte do conteúdo dessa subseção a inclusão de gráficos e tabelas de convergência. Análises e comentários sobre os resultados obtidos são obrigatórios! Não é dever do leitor (eventualmente seu "chefe") fazer isso - a princípio, você fez os testes, você os interpreta a quem lê. Gráficos, tabelas e LaTeX "floats" em geral devem ser enumerados, terem legendas e serem citados no texto. Não se esqueça das unidades métricas em tudo (em tabelas, figuras, etc, nomes de eixos, em colunas de tabelas).

IMPORTANTE: os programas entregues, se rodados, devem reproduzir os resultados reportados no RT!

Na **CONCLUSÃO**, você resumidamente explica o problema abordado, processo de discretização, abordagem de solução, comenta os principais resultados obtidos e possíveis melhorias para o futuro. Também costuma-se comentar o que sua estratégia tem de diferente/inovadora (ela é melhor ou não?) que outras que vc mencionou antes em sua revisão de literatura na INTRODUÇÃO. A seção CONCLUSÃO tem esse nome porque é a última. Os autores não têm necessariamente que "concluir" nada no sentido de deduzir. É o fechamento, em contraposição à INTRODUÇÃO.

Na sequência, você inclui as **REFERÊNCIAS BIBLIOGRÁFICAS**, a lista de trabalhos (artigos, liv

ros, sítios internet, ...) que vc usou para embasar seu relatório, inclui **APÊNDICES** que conterão os detalhes que de outra forma fariam o leitor perder o foco em sua leitura, trechos de programas mais sofisticados, etc.

ATENÇÃO! Use o LaTex conforme modelos a seguir (uma ou duas colunas, não importa, você escolhe). MSWord NÃO! Cuidado com erros de português.

Modelos LaTex/pdf zippados de BONS RELATÓRIOS: [R1] [R2].

[B] CRITÉRIO DE CORREÇÃO, ENTREGA FRACIONADA E DEFINIÇÃO DO COEFICIENTE "C"

**CRITÉRIO DE CORREÇÃO**

2.0 pontos: [1] ESTRUTURA GERAL (clareza, organização, completude e conteúdo, detalhamento, ...)
3.5 pontos: [2] MODELAGEM MATEMÁTICA E METODOLOGIA NUMÉRICA (apresentação clara, concisa e correta do processo de modelagem do Problema de Cauchy de interesse, do processo de discretização do domínio de definição e do sistema de EDOs no formato padrão, apresentação clara das equações algébricas a serem resolvidas e das técnicas numéricas (tópicos diferentes) abordadas na disciplina). Algoritmo implementado.
3.5 pontos: [3] RESULTADOS (verificação e aplicação, gráficos e tabelas de convergência numérica, interpretação de resultados). CUIDADO! Os programas entregues, quando rodados, devem refletir os resultados apresentados!
1.0 ponto..: [4] CONCLUSÃO, REFERÊNCIAS BIBLIOGRÁFICAS E APÊNDICES (escolha uma norma para citação e siga. Trechos de programa devem vir em apêndices)

RT = SOMA [i], i=1:4, é a nota do relatório e fica sujeita às seguintes alterações adicionais a seguir:

* ERROS CRASSOS DE ORTOGRAFIA, GRAMÁTICA (concordâncias verbal/nominal e "gerundismo" de telemarketing): -0.3 cada ocorrência com desconto máximo de -1.0;
* NÃO USOU LaTex, SEM CORREÇÃO (RT <-- 0);
* DESCONTO DE 0.5 ponto por dia de atraso com máximo de 3 dias de atraso na data final;
* BÔNUS 1.0 para casos excepcionais, fora do padrão em capricho, organização, escolha do problema, discretização e/ou seleção das técnicas numéricas: nesse caso, eventualmente, servirá de modelo de relatório a ser seguido (anonimizado).

**ENTREGA FRACIONADA E DEFINIÇÃO DO COEFICIENTE "C"**

Partes do RELATÓRIO TÉCNICO serão entregues antecipadamente com conteúdo preliminar. O objetivo é orientar e acompanhar o desenvolvimento ao longo do trabalho. O coeficiente C <-- 0. A cada parte entregue no prazo estabelecido, incluindo a entrega final, o coeficiente C <-- C+1. ATENÇÃO! SE a entrega "tiver conteúdo" ENTÃO C<--C+1 SENÃO C<--C).

Fazendo C <-- C/T, onde T é o número total de partes a serem entregues, incluindo o relatório final, a nota do relatório passa a ser

RT <-- MIN{10, 0.5*(1+C)*RT}
