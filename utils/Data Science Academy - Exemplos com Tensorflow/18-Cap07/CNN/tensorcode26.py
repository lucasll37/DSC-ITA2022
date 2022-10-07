# Convolutional Neural Network com TensorFlow

# Pacotes
import input_data
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Obs: Este script está compatível com as versões 1.x e 2.x do TensorFlow.
# Optamos por manter assim, pois alguns recursos avançados usados neste script ainda não foram implementados no TF 2.

# Para executar este script com TF 2, basta executar via terminal.
# Para executar com TF 1, remova o prefixo tf.compat.v1 ao longo do scriipt e substitua por tf, e comente as 3 linhas abaixo.
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.disable_eager_execution()

# Dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Parâmetros
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Parâmetros da rede
n_input = 784 # MNIST data input (shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# Dropout
# Aplicamos o dropout para reduzir o overfitting. O dropout vai eliminar algumas unidades (nas camadas ocultas, de entrada e de saída) na rede neural.
# A decisão sobre qual neurônio será eliminado é randômica e aplicamos uma probabilidade para isso. Esse parâmetro pode ser ajustado para otimizar o desempenho da rede.
dropout = 0.75 # Dropout, probabilidade para manter unidades

# Graph input
x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.compat.v1.placeholder(tf.float32) # dropout 

# Convertendo o input (x) para um tensor
_X = tf.reshape(x, shape = [-1, 28, 28, 1])

# Funções para criar o modelo
# A função tf.nn.conv2d() computa convoluções 2D a partir do tensor de input. A esse resultado adicionamos o bias.
# A função tf.nn.relu() é usada como função de ativação nas camadas ocultas. Aplicamos a ReLu aos valores de retorno das camadas de convolução.
# O parâmetro padding indica que o tensor de output terá o mesmo tamanho do tensor de entrada.
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input=img, filters=w, strides = [1, 1, 1, 1], padding = 'VALID'), b))

# Após a operação de convolução, realizamos o passo de pooling que simplifica a informação de output previamente criada pela camada de convolução.
def max_pool(img, k):
    return tf.nn.max_pool2d(input=img, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'VALID')

# Variáveis para os pesos e bias

# Pesos
# Cada neurônio da camada oculta é conectado a um pequeno grupo de tensores de entrada (input) de dimensão 5x5. Com isso, a camada oculta terá um tamanho de 24x24.
wc1 = tf.Variable(tf.random.normal([5, 5, 1, 32])) # 5x5 conv, 1 input, 32 outputs
wc2 = tf.Variable(tf.random.normal([5, 5, 32, 64])) # 5x5 conv, 32 inputs, 64 outputs
wd1 = tf.Variable(tf.random.normal([4*4*64, 1024])) # fully connected, 7*7*64 inputs, 1024 outputs
wout = tf.Variable(tf.random.normal([1024, n_classes])) # 1024 inputs, 10 outputs (class prediction)

# Bias
bc1 = tf.Variable(tf.random.normal([32]))
bc2 = tf.Variable(tf.random.normal([64]))
bd1 = tf.Variable(tf.random.normal([1024]))
bout = tf.Variable(tf.random.normal([n_classes]))


# Camada 1 de convolução
conv1 = conv2d(_X, wc1, bc1)

# Max Pooling (down-sampling)
conv1 = max_pool(conv1, k=2)

# Aplicando o Dropout
conv1 = tf.nn.dropout(conv1, 1 - (keep_prob))

# Camada 2 de convolução
conv2 = conv2d(conv1,wc2,bc2)

# Max Pooling (down-sampling)
conv2 = max_pool(conv2, k=2)

# Aplicando o Dropout
conv2 = tf.nn.dropout(conv2, 1 - (keep_prob))

# Camada totalmente conectada
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1)) # Ativação com a Relu 
dense1 = tf.nn.dropout(dense1, 1 - (keep_prob)) # Aplicando Dropout

# Output, class prediction
pred = tf.add(tf.matmul(dense1, wout), bout)

# Cost Function e Otimização
cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = tf.stop_gradient( y)))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Avaliando o Modelo
correct_pred = tf.equal(tf.argmax(input=pred,axis=1), tf.argmax(input=y,axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32))

# Inicializando as variáveis
init = tf.compat.v1.global_variables_initializer()

# Sessão
with tf.compat.v1.Session() as sess:
    sess.run(init)
    step = 1
    # Mantém o treinamento até atingir o número máximo de iterações
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training usando batch data
        sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculando a acurácia
            acc = sess.run(accuracy, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculando a perda
            loss = sess.run(cost, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 1.})
            print ("Iteração " + str(step*batch_size) + ", Perda = " + "{:.6f}".format(loss) + ", Acurácia em Treino = " + "{:.5f}".format(acc))
        step += 1
    print ("Otimização Concluída!")
    # Calculando acurácia para 256 mnist test images
    print ("Acurácia em Teste:", sess.run(accuracy, feed_dict = {x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))


