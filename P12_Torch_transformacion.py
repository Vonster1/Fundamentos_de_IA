#============================================
# TRANSFORMACIONES DE TENSORES
# ALEX BRAULIO VON STENRNEFELS HERNANDEZ
# FUNDAMENTOS DE IA ESFM IPN
#============================================

# MODULOS NECESARIOS
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

# CLASE WINEDATASET HIJA DE DATASET
class WinDataset(Dataset):
    # CONSTRUCTOR
    def __init__(self,tranform=None):
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples=xy.shape[0]
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]

        self.transform = transform

    # METODO PARA OBTENER DATOS
    def __getitem__(self,index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform: 
            sample = self.transform(sample)
        return sample

    # TAMAÑO DEL CONJUNTO DE DATOS
    def __len_(self):
        return self.n_samples

# TRANSFORMACIONES COMUNES
# DE NUMPY A TENSOR PYTORCH
class ToTensor:
    def __call__(self,sample):
        inputs, targets = sample
        return torch.from_nummpy(inputs), torch.from_numpy(targets)

# ESCALAR DATOS (MULTIPLICACION CONSTANTE)
class MulTransform:
    def __init__(self,factor):
        self.factor = factor
    def __call__(self,sample):
        inputs, targets=sample
        inputs *= self.factor
        return inputs, targets

# PROGRAMA PRINCIPAL

if __name__ == "__main__":

    print('Sin transformación')
    dataset = WineDataset()
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)

    print('\nTransformado en tensor')
    dataset = WineDataset(transform=ToTensor())
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)

    print('\nCon transformación a tensor y multiplicación')
    composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
    dataset = WineDataset(transform=composed)
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)



