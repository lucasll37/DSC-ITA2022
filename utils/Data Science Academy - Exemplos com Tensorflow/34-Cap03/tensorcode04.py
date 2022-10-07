# Operações Matemáticas com TensorFlow

# Operações matemáticas básicas - adição, subtração, multiplicação e divisão.

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Variáveis Python
x = 12
y = 6

# Operações Matemáticas com TF
a = tf.math.add(x, y) 
b = tf.math.subtract(x, y) 
c = tf.math.multiply(x, y)  
d = tf.math.divide(x, y)  

print("\n")

print(a)
print(b)
print(c)
print(d)

print("\n")


