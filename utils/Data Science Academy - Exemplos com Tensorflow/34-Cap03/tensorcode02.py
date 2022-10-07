# Configurando o Nível de Log

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Cria o objeto TensorFlow chamado hello
# Constantes são objetos cujo valor não pode ser alterado.
hello = tf.constant('Hello World!')

print("\n")
print(hello)
print("\n")

print("\n")
tf.print(hello)
print("\n")
