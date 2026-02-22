# Exemplo de Classificação de Usuário com TensorFlow.js

Este é um projeto de exemplo que demonstra como treinar e usar um modelo de rede neural para classificar usuários em categorias (premium, medium, basic) com base em seus dados (idade, cor favorita e localização).

## Tecnologias Utilizadas

- [Node.js](https://nodejs.org/)
- [TensorFlow.js](https://www.tensorflow.org/js)

## Instalação

Para instalar as dependências do projeto, execute o seguinte comando:

```bash
npm install
```

## Como Usar

Para executar o projeto, utilize o comando:

```bash
npm start
```

Este comando irá treinar o modelo com os dados de exemplo e, em seguida, fará uma predição para um novo usuário, exibindo as probabilidades de ele pertencer a cada categoria.

## Estrutura do Projeto

- `index.js`: O arquivo principal que contém todo o código para treinamento do modelo e predição.
- `package.json`: Define as dependências e os scripts do projeto.
- `.gitignore`: Especifica os arquivos e pastas que devem ser ignorados pelo Git (como a pasta `node_modules`).
