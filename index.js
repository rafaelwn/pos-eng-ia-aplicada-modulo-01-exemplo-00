import tf from '@tensorflow/tfjs-node';


async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    // Primeira camada da rede:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizacoes)

    // 80 neuronios = aqui coloquei tudo isso, pq tem pouca base de treino
    // quanto mais neuronios, mais complexidade a rede pode aprender
    // e consequentemente, mais processamento ela vai usar

    // A ReLU age como um filtro:
    // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    /// Se a informação chegou nesse neuronio é positiva, passa para frente!
    // se for zero ou negativa, pode jogar fora, nao vai servir para nada
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    // Saída: 3 neuronios
    // um para cada categoria (premium, medium, basic)

    // activation: softmax normaliza a saida em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    // Compilando o modelo
    // optimizer Adam ( Adaptive Moment Estimation)
    // é um treinador pessoal moderno para redes neurais:
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com historico de erros e acertos

    // loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // a categoria premium será sempre [1, 0, 0]

    // quanto mais distante da previsão do modelo da resposta correta
    // maior o erro (loss)
    // Exemplo classico: classificação de imagens, recomendação, categorização de
    // usuário
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Treinamento do modelo
    // verbose: desabilita o log interno (e usa só callback)
    // epochs: quantidade de veses que vai rodar no dataset
    // shuffle: embaralha os dados, para evitar viés
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                // onEpochEnd: (epoch, log) => console.log(
                //     `Epoch: ${epoch}: loss = ${log.loss}`
                // )
            }
        }
    )

    return model
}

async function predict(model, pessoa) {
    // transformar o array js para o tensor (tfjs)
    const tfInput = tf.tensor2d(pessoa)

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}

function normalizarPessoa(pessoa) {
    // Normalizar idade (usando min=25, max=40 do treino)
    const idadeNormalizada = (pessoa.idade - 25) / (40 - 25)
    
    // One-hot encoding para cor
    const cores = { azul: [1, 0, 0], vermelho: [0, 1, 0], verde: [0, 0, 1] }
    const corEncoded = cores[pessoa.cor] || [0, 0, 0]
    
    // One-hot encoding para localização
    const localizacoes = { 
        "São Paulo": [1, 0, 0], 
        "Rio": [0, 1, 0], 
        "Curitiba": [0, 0, 1] 
    }
    const locEncoded = localizacoes[pessoa.localizacao] || [0, 0, 0]
    
    return [[idadeNormalizada, ...corEncoded, ...locEncoded]]
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
const pessoas = [
    { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
    { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
    { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" },
    // Dados sintéticos para premium (padrão similar a Erick)
    { nome: "João", idade: 32, cor: "azul", localizacao: "São Paulo" },
    { nome: "Marina", idade: 28, cor: "azul", localizacao: "São Paulo" },
    { nome: "Roberto", idade: 35, cor: "azul", localizacao: "Rio" },
    // Dados sintéticos para medium (padrão similar a Ana)
    { nome: "Sofia", idade: 26, cor: "vermelho", localizacao: "Rio" },
    { nome: "Lucas", idade: 24, cor: "vermelho", localizacao: "Curitiba" },
    { nome: "Fernanda", idade: 27, cor: "verde", localizacao: "Rio" },
    // Dados sintéticos para basic (padrão similar a Carlos)
    { nome: "Pedro", idade: 38, cor: "verde", localizacao: "Curitiba" },
    { nome: "Marcela", idade: 42, cor: "verde", localizacao: "São Paulo" },
    { nome: "Thiago", idade: 39, cor: "vermelho", localizacao: "Curitiba" }
];

// Normalizar todas as pessoas usando a função normalizarPessoa
const tensorPessoasNormalizado = pessoas.map(p => normalizarPessoa(p)[0]);

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1], // basic - Carlos
    // Dados sintéticos premium
    [1, 0, 0], // João
    [1, 0, 0], // Marina
    [1, 0, 0], // Roberto
    // Dados sintéticos medium
    [0, 1, 0], // Sofia
    [0, 1, 0], // Lucas
    [0, 1, 0], // Fernanda
    // Dados sintéticos basic
    [0, 0, 1], // Pedro
    [0, 0, 1], // Marcela
    [0, 0, 1]  // Thiago
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dado melhor!
// assim o algoritmo consegue entender melhor os padrões complexos
// dos dados
const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: 'zé', idade: 28, cor: 'azul', localizacao: "São Paulo" }
const pessoaTensorNormalizado = normalizarPessoa(pessoa)

const predictions = await predict(model, pessoaTensorNormalizado)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
console.log(results)