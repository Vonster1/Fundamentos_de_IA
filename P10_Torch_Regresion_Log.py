#===============================================
# REGRESION LOGISTICA CON PYTORCH
# ALEX BRAULIO VON STERNENFELS HERNANDEZ
# FUNDAMENTOS DE IA ESFM IPN
#===============================================
import torch
import torch.nn as nn 
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# PREPARAR DATOS
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
print("Las variables medidas son: ",  n_features)

# SEPARAR DATOS PARA APRENDIZAJE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, ramdom_state=1234)

# ESCALAR DATOS
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PASAR A TENSORES DE PYTORCH
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(Y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# MODELO
class Model(nn.Module):
    def __init__(self, n_input_features


