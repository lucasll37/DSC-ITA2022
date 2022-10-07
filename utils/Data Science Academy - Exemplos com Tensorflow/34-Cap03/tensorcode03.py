# Tensor

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Tensor

# O TensorFlow armazena dados em Tensores que são semelhantes a arrays multidimensionais do numPy 
# Em TensorFlow, os dados não são armazenados como inteiros, floats ou strings de caracteres. 
# Esses valores são encapsulados em um objeto chamado de tensor. 

# No caso de hello = tf.constant('Hello World!'), hello é um tensor de string 0-dimensional, 
# mas tensores podem assumir uma variedade de tamanhos como mostrado abaixo.


# hello é um tensor 0-dimensional string - Rank 0
hello = tf.constant('Hello World!')

print("\n")
print(hello)
print("Rank:", tf.rank(hello))


# A é um tensor 0-dimensional int32 (escalar) - Rank 0
A = tf.constant(1234) 

print("\n")
print(A)
print("Rank:", tf.rank(A))


# B é um tensor 1-dimensional float32 - Rank 1
B = tf.constant([123,456,789], dtype = tf.float32) 

print("\n")
print(B)
print("Rank:", tf.rank(B))


# C é um tensor 2-dimensional int16 - Rank 2
C = tf.constant([ [123, 456, 789], [222, 333, 444] ], dtype = tf.int16)    

print("\n")
print(C)
print("Rank:", tf.rank(C))


# D é um tensor 3-dimensional int32 - - Rank 3
D = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])

print("\n")
print(D)
print("Rank:", tf.rank(D))


# tf.constant () é uma das muitas operações TensorFlow disponíveis. 
# O tensor retornado por tf.constant() é chamado de tensor constante, porque o valor do tensor nunca muda.
print("\n")

# Rank	     Math entity
# 0	Scalar   (somente magnitude)
# 1	Vector   (magnitude e direção)
# 2	Matrix   (tabela)
# 3	3-Tensor (cubo)
# n	n-Tensor (n-dimensões)




