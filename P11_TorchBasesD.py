#============================================================
# MANEJO DE DATOS EN PYTORCH
# ALEX BRAULIO VON STERNENFELS HERNANDEZ
# ESFM IPN 
#============================================================

# MÓDULOS NECESARIOS
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# BIGDATA DEBE DIVIDIRSE EN PEQUEÑOS GRUPOS DE DATOS

# CICLO DE ENTRENAMIENTO 
# for epoch in range(num_epochs):
#   # CICLO SOBRE TODOS LOS GRUPOS DE DATOS
#   for i in range(total_batches)


# epoch = una evaluación y retropropagación para todos los datos de entranamiento
# total_batches = número total de subconjuntos de datos
# batch_size = número de datos de entrenamiento en cada subconjunto
# number of iterations = número de iteraciones sobre todos los datos de entrenamiento

# e.g : 100 samples, batch_size = 20  ->  100/20 = 5 iterations for 1 epoch

# DATA LOADER PUEDE DIVIDIR LOS DATOS EN GRUPOS

# IMPLEMENTACIÓN DE BASE DE DATOS TÍPICA
# Implement __init__ , __getitem__ , and __len__

# HIJO DE DATASET
class WindeDataset(Dataset):
    def __init__(self):
        # INICIALIZAR, BAJAR DATOS, ETC.
        # LECTURA CON NUMPY O PANDAS
        
        # TÍPICOS DATOS SEPARADOS POR COMA
        # DELIMITER = SÍMBOLO DELIMITADOR
        # SKIPROWS = LÍNEAS DE ENCABEZADO
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # PRIMERA COLUMNA ES ETIQUETA DE CLASE Y EL RESTO SON CARACTERÍSTICAS
        self.x_data = torch.from_numpy(xy[:,1:]) # GRUPOS DEL 1 EN ADELANTE
        self.y_data = torch.from_numpy(xy[:,[0]]) #GRUPO 0

    # PERMITIR INDEXACIÓN PARA OBTENER EL DATO i DE DATASET[I]
    # MÉTODO GETTER
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    # LEN(DATASET) ES EL TAMAÑO DE LA BASE DE DATOS
    def __len__(self):
        return self.n_samples

# INSTANCIAR BASE DE DATOS
data = WineDataset()

# LEER CARACTERISTICAS DEL PRIMER DATO
first_data = dataset[0]
features, labels = fist_data
print(features, labels)

# CARGAR TODA LA BASE DE DATOS CON DATALOADER
train_loader = DataLoader(dataset=dataset,
                          batch_size = 4,
                          shuffle=True,
                          num_workers=2)

#CONVERTIR EN ITERADOR Y OBSERVAR UN DATO AL AZAR
dataiter = iter(train_loader)
data = next(dataiter)
features, labels = data
print(features, labels)

# CICLO DE APRENDIZAJE VACÍO
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        if (i+1)%5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step{i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shap}')

# ALGUNAS BASES DE DATOS EXISTEN EN TORCHVISION.DATASETS
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
train_loader = DataLoader(dataset=train_dataset,
                                            batch_size=3,
                                            shuffle=True)

# LOOK AT ONE RANDOM SAMPLE
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)

