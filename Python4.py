#=========================================================
# RED NEURONAL ARTIFICIAL
# ALEX BRAULIO VON STERNENFELS HERNÁNDEZ
# FUNDAMENTOS DE IA ESFM IPN
#========================================================
import numpy as np
import pandas as pd
from matplotlib impor pyplot as plt

#INICIALIZACIÓN
def init_params():
    w1 = np.random.rand(10,784) -0.5
    B1 = np.random.rand(10,1) -0.5
    w1 = np.random.rand(10,710) -0.5
    w1 = np.random.rand(10,1) -0.5
    return w1, b1, w2, b2
#FUNCIÓN DE ACTIVACIÓN ReLU
def ReLU(Z):
    return np.maximun(Z, 0)
#FUNCIÓN DE ACTIVACIÓN softmax
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
#EVALUAR RED 
def forward_prop(w1, b1, w2, b2, X):
    Z1 = W1.dot(X)
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    retun Z1, A1, Z2, A2
#DERIVADA DE LA ReLU
def ReLU:deriv(Z):
    return Z > 0
#CLASIFICACIÓN DE SALIDAS
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
#CÁLCULO DE GRADIENTE
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot_(Y)
    dz2 = a2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * npsum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dw1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2
#MEJORA DE PARÁMETROS
def update_params(W1, b1,W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, w2, b2
#PREDICCIONES
def get_predictions(A2):
    return np.argmax(A2,0)
#PRECISIÓN
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
#DESCENSO DE GRADIENTE
def gradient_desent(X,Y,alpha,interactions):
    W1,b1,W2,b2 0 init_params()
    for i in range(interactions):
        Z1,A1,Z2,A2 = forward_prop()



