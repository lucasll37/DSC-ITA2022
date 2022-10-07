# Reshape de Tensores


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Criando um tensor
tensor = tf.constant([[3, 2],
                      [5, 2],
                      [9, 5],
                      [1, 3]])

print('\nShape do Tensor Original:', tensor.shape)
print('\nRank do Tensor Original:', tf.rank(tensor))

# Remodelando o tensor em uma forma de: shape = [linhas, colunas]
tensor_reshaped = tf.reshape(tensor = tensor, shape = [1, 8])


tf.print('\nTensor antes do reshape:\n', tensor)

print(('\nTensor depois do reshape:\n{0}').format(tensor_reshaped.numpy()))

print('\nShape do Novo Tensor:', tensor_reshaped.shape)

print('\nRank do Novo Tensor:', tf.rank(tensor_reshaped))

print("\n")