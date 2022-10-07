# Slice de Tensores


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Tensor variável (muda)
A = tf.Variable([[3, 2, 6],
                 [6, 2, 7]])


print("\n")
print('Tensor A:', A)

print("\n")

# Slice
Novo_A1 = A[1]
print('Novo Tensor A1:', Novo_A1)
print("\n")

Novo_A2 = A[1, 2]
print('Novo Tensor A2:', Novo_A2)
print("\n")

Novo_A3 = A[:,2]
print('Novo Tensor A3:', Novo_A3)
print("\n")

Novo_A4 = A[0:1, 1:2]
print('Novo Tensor A4:', Novo_A4)
print("\n")
