# TensorBoard
# Construindo um único neurônio

# Ao treinar uma rede neural, pode ser útil acompanhar os parâmetros da rede, as entradas e saídas dos nós, para que você possa ver se seu modelo 
# está aprendendo após cada etapa de treinamento e se o erro está sendo minimizado ou não. 

# TensorBoard é uma estrutura projetada para análise e depuração de modelos de redes neurais. O TensorBoard usa os chamados "summaries" 
# para visualizar os parâmetros do modelo. Uma vez que um código TensorFlow seja executado, podemos chamar o TensorBoard para ver resumos em uma 
# interface gráfica (GUI). Além disso, o TensorBoard pode ser usado para exibir e estudar o grafo computacional do TensorFlow, que pode ser muito 
# complexo para um modelo de Rede Neural Profunda.

# O TensorFlow usa grafos de computação para executar um aplicativo, onde cada nó representa uma operação e os arcos são os dados entre operações. 
# A ideia principal no TensorBoard é associar o chamado "summary" com os nós (operações) do grafo. Executando o código, as operações de summary 
# vão serializar os dados do nó que está associado a ele e emitirão os dados para um arquivo que pode ser lido pelo TensorBoard.

# Podemos então executar o TensorBoard e visualizar as operações de forma sumarizada. O fluxo de trabalho ao usar o TensorBoard é: 
# 1- Crie seu grafo / código computacional 
# 2- Anexe operações de resumo (summary) aos nós que você está interessado em examinar 
# 3- Inicie a execução do grafo como faria normalmente
# 4- Após executar o código, use o TensorBoard para visualizar as saídas de resumo

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Obs: Este script está compatível com as versões 1.x e 2.x do TensorFlow.
# Optamos por manter assim, pois alguns recursos avançados usados neste script ainda não foram implementados no TF 2.

# Para executar este script com TF 2, nenhum passo adicional precisa ser feito.
# Para executar com TF 1, remova o prefixo tf.compat.v1 ao longo do scriipt e substitua por tf, e comente as 3 linhas abaixo.
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.disable_eager_execution()

# Input
input_value = tf.constant(0.5, name="input_value")

# Peso
weight = tf.Variable(1.0, name="weight")

# Output esperado (usado no treinamento)
expected_output = tf.constant(0.0, name="expected_output")

# Criação do modelo
model = tf.multiply(input_value, weight, "model")

# Cost Function
loss_function = tf.pow(model - expected_output, 2, name="loss_function")

# Otimizador
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.025).minimize(loss_function)

# Definição do tf.summary.scalar() para visualizar no TensorBoard
for value in [input_value, weight, expected_output, model, loss_function]:
    tf.compat.v1.summary.scalar(value.op.name, value)

# Merge de todos os summaries em uma única saída
summaries = tf.compat.v1.summary.merge_all()
sess = tf.compat.v1.Session()

# Gravando os resultados
summary_writer = tf.compat.v1.summary.FileWriter('/tmp/testetb', sess.graph)

# Inicializando as variáveis na sessão
sess.run(tf.compat.v1.global_variables_initializer())

# Executando a sessão e gerando os summaries
for i in range(100):
    summary_writer.add_summary(sess.run(summaries), i)
    sess.run(optimizer)


# Inicializando o Tensorboard
# tensorboard --logdir = /tmp/testetb




