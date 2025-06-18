#=======================================
# REGRESION LINEAL SIMPLE EN PYTORCH
# ALEX BRAULIO VON STERNENFELS HERNANDEZ 
# FUNDAMENTOS DE IA ESFM IPN
#=======================================
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# PREPARAR DATOS
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# ENVIAR A TENSOR DE PRECISION SENCILLA
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_samples, n_features = X.shape

# MODELO, MODELO LINEAL F = wx + b
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# ERROR Y OPTIMIZADOR
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# CICLO DE APRENDIZAJE
num_epoch = 200
for epoch in range(num_epoch):
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1)%10 == 0 :
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# GRAFICA 
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'h')
plt.show()
