# Desafio da RZK Digital

Nós, aqui na RZK Digital, somos uma equipe de ciência de dados que abraçou o desenvolvimento de software.

No seu dia a dia, você realizará tarefas que abrangem desde análises de dados até a criação e manutenção de softwares, utilizando Python e SQL.

Este desafio tem como objetivo avaliar sua capacidade de aprender, mas principalmente a manipulação e apresentação de dados.

## Descrição

### Treinamento

O arquivo [_titanic.csv_](./titanic.csv) contém dados de 891 passageiros do Titanic.

A segunda coluna deste CSV contém o valor 1 se o passageiro sobreviveu e 0 caso contrário.

No arquivo [_script.py_](./script.py), escreva o código para treinar modelos de Machine Learning capazes de prever, com base nas informações de um passageiro, se ele sobreviveu ou não.

Treine os três modelos a seguir: regressão logísica, K-Nearest Neighbors (KNN) e árvore de decisão.

### Power BI

Ao final do _script_, após o treinamento, salve os resultados dos modelos no arquivo CSV [_titanic_treinado.csv_](./titanic_treinado.csv).

Cada linha desse novo CSV deve conter:

- o nome do modelo de Machine Learning,
- as informações de um passageiro,
- o valor predito pelo respectivo modelo dizendo se o passageiro sobreviveu ou não,
- o valor real indicando se o passageiro sobreviveu.

Com [_titanic_treinado.csv_](./titanic_treinado.csv), crie um _dashboard_ de uma página no Power BI contendo, no mínimo:

- uma matriz de confusão com os resultados dos modelos;
- um _slicer_ para filtrar os resultados na matriz de acordo com um modelo selecionado.

### Bibliotecas

O arquivo [_requirements.txt_](./requirements.txt) especifica as bibliotecas necessárias para o desafio; no entanto, uma biblioteca está faltando.

Se você tentar executar o arquivo [_script.py_](./script.py), ocorrerá um erro.

Adicione a biblioteca faltante ao [_requirements.txt_](./requirements.txt) para corrigir o problema.

### _Pull Request_

Submeta um _pull request_ com a solução do desafio.

### Observações

Esta é uma tarefa introdutória em ciência de dados. Há muitos conteúdos disponíveis na Internet que podem ajudá-lo a realizá-la. Sinta-se à vontade para buscar referências e aprender.

Se tiver dúvidas ou precisar de ajuda, não hesite em nos consultar.
