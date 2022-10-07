# Conversão de Dados dos Tensores


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Criando um tensor de floats
tensor = tf.constant([[3.1, 2.8],
                      [5.2, 2.3],
                      [9.7, 5.5],
                      [1.1, 3.4]], 
                      dtype = tf.float32)

# Convertendo o tensor de float32 para int32
tensor_de_inteiros = tf.cast(tensor, tf.int32)

# Arredonda os elementos do tensor
tensor_arredondado = tf.math.round(tensor)


tf.print('\nTensor com floats:\n', tensor)
tf.print('\nTipo de Dado do Tensor:\n', tensor.dtype)

tf.print('\nTensor cast de float para int (apenas remove o decimal, não faz arredondamento):\n', tensor_de_inteiros)
tf.print('\nTipo de Dado do Tensor:\n', tensor_de_inteiros.dtype)

tf.print('\nTensor com arrendodamento:\n', tensor_arredondado)
tf.print('\nTipo de Dado do Tensor:\n', tensor_arredondado.dtype)

print("\n")