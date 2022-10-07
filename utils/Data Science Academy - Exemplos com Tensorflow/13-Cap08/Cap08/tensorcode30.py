# Tensor Board - Visualizando o Aprendizado

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Obs: Este script está compatível com as versões 1.x e 2.x do TensorFlow.
# Optamos por manter assim, pois alguns recursos avançados usados neste script ainda não foram implementados no TF 2.

# Para executar este script com TF 2, nenhum passo adicional precisa ser feito.
# Para executar com TF 1, remova o prefixo tf.compat.v1 ao longo do scriipt e substitua por tf, e comente as 3 linhas abaixo.
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.disable_eager_execution()

# Importando função para carga de dados
#from tensorflow.examples.tutorials.mnist import input_data
import input_data

FLAGS = None


def train():
  # Importando os dados
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True, fake_data = FLAGS.fake_data)

  # Criando um modelo com múltiplas camadas
  sess = tf.compat.v1.InteractiveSession()
  
  # Create a multilayer model.

  # Input placeholders
  with tf.compat.v1.name_scope('input'):
    x = tf.compat.v1.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.compat.v1.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.compat.v1.summary.image('input', image_shaped_input, 10)

  # Criando uma variável de peso com a inicialização apropriada.
  def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

  # Criando uma variável de bias com inicialização apropriada
  def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

  # Anexando sumários ao Tensor para visualização no TensorBoard
  def variable_summaries(var):
    with tf.compat.v1.name_scope('summaries'):
      mean = tf.reduce_mean(input_tensor=var)
      tf.compat.v1.summary.scalar('mean', mean)
      with tf.compat.v1.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
      tf.compat.v1.summary.scalar('stddev', stddev)
      tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
      tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
      tf.compat.v1.summary.histogram('histogram', var)

  # Código reutilizável para construir uma simples camada de rede neural. 
  # Fazemos multiplicação de matrizes, adicionamos o bias e, em seguida, usamos a relu como função de ativação.
  # Definimos o escopo de nome para que o grafo resultante seja fácil de ler, e adicionamos alguns sumários.
  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):

    # A adição de um escopo de nome garante o agrupamento lógico das camadas no gráfico.
    with tf.compat.v1.name_scope(layer_name):
      # Esta variável manterá o estado dos pesos para a camada
      with tf.compat.v1.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.compat.v1.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.compat.v1.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.compat.v1.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.compat.v1.summary.histogram('activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.compat.v1.name_scope('dropout'):
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    tf.compat.v1.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, 1 - (keep_prob))

  # Gera a camada
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  with tf.compat.v1.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_), logits=y)
    with tf.compat.v1.name_scope('total'):
      cross_entropy = tf.reduce_mean(input_tensor=diff)
  tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)

  with tf.compat.v1.name_scope('train'):
    train_step = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.compat.v1.name_scope('accuracy'):
    with tf.compat.v1.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(input=y, axis=1), tf.argmax(input=y_, axis=1))
    with tf.compat.v1.name_scope('accuracy'):
      accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
  tf.compat.v1.summary.scalar('accuracy', accuracy)

  # Junta todos os sumários e grava em /tmp/tensorflow/mnist/logs/mnist_with_summaries (por padrão)
  merged = tf.compat.v1.summary.merge_all()
  train_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.compat.v1.global_variables_initializer().run()

  
  # Cria um TensorFlow feed_dict para mapear os dados nos placeholders
  def feed_dict(train):
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Acurácia no passo %s: %s' % (i, acc))
    else:  
      if i % 100 == 99:  
        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True), options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adicionando run metadata para', i)
      else:  # Registra um sumário
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.io.gfile.exists(FLAGS.log_dir):
    tf.io.gfile.rmtree(FLAGS.log_dir)
  tf.io.gfile.makedirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False, help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000, help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries', help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# Inicializar o TensorBoard:
# tensorboard --logdir=/tmp/tensorflow
