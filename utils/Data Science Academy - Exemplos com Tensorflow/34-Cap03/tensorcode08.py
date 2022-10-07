# Criando Tensores 


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


print("\n")

# Criando um tensor preenchido com zeros.
tensor = tf.zeros(shape = [3, 4], dtype = tf.int32)
print(('Tensor preenchido com zeros como int32, 3 linhas e 4 colunas:\n{0}').format(tensor.numpy()))

print("\n")

# Criando um tensor preenchido com valor 1 e tipo de dados float32.
tensor = tf.ones(shape = [5, 3], dtype = tf.float32)
print(('\nTensor preenchido com valor 1 e float32, 5 linhas e 3 colunas:\n{0}').format(tensor.numpy()))

print("\n")

# Criando um tensor preenchido com valor 100 e tipo de dados float64.
tensor = tf.constant(100, shape = [4, 4], dtype = tf.float64)
print(('\nTensor preenchido com valor 100 e float64, 4 linhas e 4 colunas:\n{0}').format(tensor.numpy()))

print("\n")

# Criando um tensor preenchido Rank 2 preenchido com zeros
tensor = tf.Variable(tf.zeros([1, 2]))
print(tensor)

print("\n")

# Atribuindo valores ao tensor criado no item anterior
tensor.assign_add([[100, 200]])
print(tensor)

print("\n")