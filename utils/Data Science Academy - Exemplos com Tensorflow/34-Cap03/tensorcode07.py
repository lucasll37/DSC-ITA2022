# Concatenação de Tensores


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Tensor constante (não muda)
A = tf.constant([[3, 2],
                 [5, 2]])

# Tensor constante (não muda)
B = tf.constant([[9, 5],
                 [1, 3]])

print("\n")
tf.print(A)

print("\n")
tf.print(B)

print("\n")

# Concatenando colunas
AB_concatenados = tf.concat(values = [A, B], axis = 1)
tf.print('Concatenando colunas de B em A:\n', AB_concatenados)

print("\n")

# Concatenando linhas
AB_concatenados = tf.concat(values = [A, B], axis = 0)
tf.print('\nConcatenando linhas de B em A:\n', AB_concatenados)

print("\n")

# A primeira saída concatenará coluna por eixo = 1 e a segunda concatenará linha por eixo = 0
# O que significa que incluímos os dados para a direita (colunas) ou para baixo (linhas).

