#=============================================
#          DESCENSO DE GRADIENTE
# ALEX BRAULIO VON STERNENFELS HERNANDEZ
# FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL
# ESFM IPN MARZO 2025
#=============================================

# MÓDULOS NECESARIOS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ¡¡IMPORTANTANTE!! AGREGAR LA FUNCIÓN DE MINIMOS CUADRADOS

# FUNCIÓN DESCENSO DE GRADIENTE (ADAM)
def DG_ADAM(epocs,dim,X,Y,Ybar,alpha,grad):
    error = np.zeros(epocs,dtype=np.float32)
    mn = np.zeros(dim,dtype=np.float32)
    vn = np.zeros(dim,dtype=np.float32)
    g = np.zeros(dim,dtype=np.float32)
    mg2 = np.zeros(dim,dtype=np.float32)
    w = np.zeros(dim,dtype=np.float32)

    beta1 = 0.80
    beta2 = 0.99
    b1 = beta1
    b2 = beta2
    eps = 1.0e-8

    mn[0],mn[1] = grad(X,Y,w[0],w[w1])
    vn = mn*mn
    
    for j in range(epocs):
        g[0],g[1] = grad(X,Y,w[0],w[1])
        g2 =g*g

    for j in range(dim):

