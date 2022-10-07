# Recurrent Neural Network com TensorFlow

# Para usar RNNs na classificação de imagens, precisamos considerar cada imagem como uma sequência de pixels. 
# Como as imagens no dataset MNIST possuem um shape 28x28 pixels, nós iremos trabalhar com 28 sequências de 28 timesteps para cada amostra.

# Pacotes
import input_data
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from tensorflow.contrib import rnn

# Obs: Este script está compatível com as versões 1.x e 2.x do TensorFlow.
# Optamos por manter assim, pois alguns recursos avançados usados neste script ainda não foram implementados no TF 2.

# Para executar este script com TF 2, basta executar via terminal.
# Para executar com TF 1, remova o prefixo tf.compat.v1 ao longo do scriipt e substitua por tf, e comente as 3 linhas abaixo.
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.disable_eager_execution()

# Dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Parâmetros para o processo de aprendizagem
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Inputs, Steps, Hidden e Output
n_input = 28 
n_steps = 28 
n_hidden = 128 # número de features na camada oculta
n_classes = 10 

# Input e classes
x = tf.compat.v1.placeholder("float", [None, n_steps, n_input])
y = tf.compat.v1.placeholder("float", [None, n_classes])

# Pesos e bias
weights = {'out': tf.Variable(tf.random.normal([n_hidden, n_classes]))}
biases = {'out': tf.Variable(tf.random.normal([n_classes]))}

# Função para o modelo RNN
def RNN(x, weights, biases):
    x = tf.transpose(a=x, perm=[1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(axis = 0, num_or_size_splits = n_steps, value = x)
    lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    outputs, states = tf.compat.v1.nn.static_rnn(lstm_cell, x, dtype = tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# Executando o modelo
pred = RNN(x, weights, biases)

# Cost Function e Otimização
cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = tf.stop_gradient( y)))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Previsões e acurácia
correct_pred = tf.equal(tf.argmax(input=pred,axis=1), tf.argmax(input=y,axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32))

# Inicializando as variáveis
init = tf.compat.v1.global_variables_initializer()

# Sessão
with tf.compat.v1.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict = {x: batch_x, y: batch_y})
            print("Iteração " + str(step*batch_size) + ", Perda = " + "{:.6f}".format(loss) + ", Acurácia no Treino = " + "{:.5f}".format(acc))
        step += 1
    print("Treinamento e Otimização Concluídos!")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Acurácia no Teste:", sess.run(accuracy, feed_dict = {x: test_data, y: test_label}))


