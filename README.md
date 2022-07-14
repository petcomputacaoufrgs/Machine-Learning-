
# Machine Learning
Esse repositório tem o intuito de guiar interessados em aprender *Machine Learning*. 
Ele foi feito com base nos conhecimentos adquiridos pelos petianos que trabalharam no [Projeto Papagaio](https://github.com/petcomputacaoufrgs/papagaio).
### Tabela de conteúdos
**[Introdução](#introdução)**<br>
**[Redes Neurais](#redes-neurais)**<br>
**[LSTM](#lstm)**<br>

## Introdução
Na programação convencional, damos ordens para o computador fazer uma tarefa para gerar um resultado. Com Machine Learning, nós damos o resultado para o computador, e deixamos ele aprender o melhor jeito de fazer a tarefa. Machine Learning nos permite fazer coisas (...).

(videos)

E mesmo com conhecimentos básicos, também conseguimos fazer coisas incríveis:

(videos)



### Pré requisitos
Álgebra linear, conceitos de derivada, python, numpy 



### Recomendações
[Coursera - Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning?specialization=machine-learning-introduction): esse curso é uma excelente base para quem está começando, porque ele aborda tudo que é necessário aprender de uma forma bem didática, com exercícios teóricos e práticos. Entretanto, as aulas podem acabar sendo muito pesadas, pois o conteúdo é apresentado detalhadamente. <br>
[Curso de Machine Learning - PET Computação](https://petcompufrgs.github.io/ml-course/): desenvolvido pelo ex-petiano Thiago, esse curso encapsula todos os conceitos de Machine Learning sucintamente, em forma de texto. É um excelente material para complementar os estudos de Machine Learning.

## Redes Neurais
Regressão linear e regressão logística são apenas a ponta do iceberg na área de Machine Learning. Com esses modelos, temos claras limitações devido à falta de complexidade. Nesse contexto, as redes neurais foram desenvolvidas para que se pudesse construir modelos muito mais profundos, capazes de resolver uma gama muito maior de problemas. <br>
Para conseguir entender as redes neurais, é extremamente necessário que você, após entender a teoria, bote a mão na massa e tente fazer suas próprias redes, a fim de entender como todo o processo dela funciona. <br>
Abaixo temos duas abas, recomendamos que você veja os vídeos para aprender a teoria, e depois tentem fazer os códigos por conta, apenas consultando quando surgirem dúvidas. 

### Recomendações
[Redes neurais - 3b1b](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi): é uma ótima série de vídeos para se ter uma visão inicial sobre os conceitos de Redes Neurais. Recomenda-se começar o estudo de Redes Neurais com essa playlist.

[Coursera - Advanced Learning Algorithms](https://www.coursera.org/learn/advanced-learning-algorithms?specialization=machine-learning-introduction#syllabus)



### Exercícios para fixação
Redes neurais são extremamente complicadas de entender sem praticar, principalmente a parte de backpropagation, que envolve muita matemática. <br>
Segue abaixo uma lista de exercícios para praticar a construção de Redes Neurais. <br>
**Exercício 1**:
Desenvolva uma rede neural trivial para obter o valor desejado de y, conforme imagem abaixo:
![Exercicio 1](Códigos/Imagens/exercise_1.png)

O *learning rate* e o número de épocas fica a seu critério. Os valores da imagem são apenas como referência.
É recomendado tentar resolver esse exercício de maneira simples, apenas utilizando um laço *for* para iterar por múltiplas épocas do treino. O objetivo desse exercício é ter um primeiro contato com o forward e backward pass, observando como muda a loss, o peso e o valor de saída da rede.

 **Exercício 2**:
Modifique o exercício anterior, adicionando mais neurônios, conforme a imagem abaixo:
![Exercicio 2](Códigos/Imagens/exercise_2.png)
Preste atenção em como a loss cai em comparação ao exercício anterior. 

**Exercício 3**:
Construa uma rede neural que simule o portão lógico XOR. 
```python
# Tabela verdade XOR
x = [[0,0], [1, 0], [1, 1], [0, 1]]  
y = [0, 0, 1, 0]
```
Para resolver esse problema, é necessário utilizar os conhecimentos obtidos dos exercícios anteriores de forward e backward pass, e adicionar a função de ativação Sigmoide na camada de saída, já que o XOR não é linearmente separável. Diagrama da rede:<br>
![Exercicio 3](Códigos/Imagens/exercise_3.png)
Os pesos devem ser inicializados utilizando a função [randn do NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html) e os bias devem ser 1. <br>
Para esse exercício, utilize o learning rate de 0.2 e 5000 épocas como referência. <br>
É esperado que ao final do treino a loss convirja para próximo de 0, e a rede neural acerte todas combinações da XOR.

### Resoluções dos exercícios
É fundamental olhar as resoluções somente após tentar fazer os exercícios por conta. 
[Exercícios 1 e 2](dummy_neural_network.ipynb)





 



## LSTM
