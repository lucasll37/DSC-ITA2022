# Selecionando o dispositivo - GPU x CPU

# No TensorFlow os dispositivos suportados são representados como sequências de caracteres. 
# Por exemplo: 
# /cpu:0: - A CPU de sua máquina 
# /gpu:0: - A GPU da sua máquina, se você tiver apenas uma 
# /gpu:1: - A segunda GPU da sua máquina, e assim por diante

# Para definir o código a ser executado na GPU, use: with tf.device('/gpu:0')

import numpy as np
import tensorflow as tf
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Obs: Este script está compatível com as versões 1.x e 2.x do TensorFlow.
# Optamos por manter assim, pois alguns recursos avançados usados neste script ainda não foram implementados no TF 2.

# Para executar este script com TF 2, nenhum passo adicional precisa ser feito.
# Para executar com TF 1, remova o prefixo tf.compat.v1 ao longo do scriipt e substitua por tf, e comente as 3 linhas abaixo.
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.disable_eager_execution()

# Podemos configurar o programa para descobrir quais dispositivos suas operações e tensores são atribuídos. 
# Para isso, criaremos uma sessão com o seguinte parâmetro log_device_placement definido como True:
log_device_placement = True

# Número de multiplicações
n = 10

# Definindo duas matrizes
A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')

# Array para armazenar os resultados
c1 = []

# Multiplicação das matrizes
def matpow(M, n):
    if n < 1: 
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

# A multiplicação das matrizes será feita na GPU
with tf.device('/gpu:0'): # Para executar na CPU, use /cpu:0
    a = tf.compat.v1.placeholder(tf.float32, [10000, 10000])
    b = tf.compat.v1.placeholder(tf.float32, [10000, 10000])
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

# A soma dos elementos da matriz será feita na CPU
with tf.device('/cpu:0'):
  sum = tf.add_n(c1)   

# Avaliar o tempo de computação - início da execução
t1_1 = datetime.datetime.now()

# Sessão
with tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(log_device_placement = log_device_placement)) as sess:
     sess.run(sum, {a:A, b:B})

# Avaliar o tempo de computação - fim da execução
t2_1 = datetime.datetime.now()

# Print do tempo de execução
print("GPU Computation Time: " + str(t2_1-t1_1))



