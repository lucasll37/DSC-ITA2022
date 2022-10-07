# Variáveis no Tensorflow

# Python code puro
print("\n")

x = 1
y = x + 9
print('Soma em Python:')
print(y)

print("\n")

# O mesmo código anterior com Tensorflow
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Valor constante
x = tf.constant(1)


# Variáveis no TensorFlow são como variáveis em qualquer outra linguagem.
# As variáveis (como você pode adivinhar pelo nome) podem conter valores diferentes ao contrário das constantes. 
y = tf.Variable(x + 9)

print("\n")
print(y)
print(y.shape)
print("\n")

# Adicionando Tensores
tensor_a = tf.Variable([[1, 2, 2], [3, 4, 2]], dtype = tf.int32)
print('Tensor a:', tensor_a)	
print('Rank Tensor a:', tf.rank(tensor_a))
print("\n")

tensor_b = tf.Variable([[5, 6, 7]], dtype = tf.int32)
print('Tensor b:', tensor_b)
print('Rank Tensor b:', tf.rank(tensor_b))
print("\n")

tensor_add = tf.add(tensor_a, tensor_b)
print('Tensor_add:', tensor_add)	
print('Rank Tensor_add:', tf.rank(tensor_add))

print("\n")







