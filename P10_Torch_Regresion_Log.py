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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# ESCALAR DATOS
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PASAR A TENSORES DE PYTORCH
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# MODELO
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model,self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = Model(n_features)

# ERROR Y OPTIMIZACION
num_epochs = 100
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# CICLO DE APRENDIZAJE
for epoch in range(num_epochs):
    y_pred = model(X_train)
    loss=criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if(epoch+1)%10==0:
        print(f'epoch: {epoch+1}, loss={loss.item():.4f}')

# EXAMINAR RESULTADO
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print('precision: [acc.item():.4f]')





