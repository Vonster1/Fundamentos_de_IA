#=====================================================
#  EJEMPLO DE RED NEURONAL CONVOLUCIONAL
#  TRADUCIDO DE PYTORH TUTORIAL 2023
#  ALEX BRAULIO VON STERNENFELS HERNANDEZ
#  FUNDAMENTOS DE IA ESFM IPN
#=====================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# CONFIGURACION DEL GPU
device = torch.device('cuda' if torch.cuda.is_avalable() else 'cpu')

# HIPER-PARAMETROS
num_epochs = 10  # ITERACIONES SOBRE LOS DATOS DE ENTRENAMIENTO
batch_size = 4  # SUBCONJUNTOS DE DATOS
learning_rate = 0.001 # TASA DE APRENDIZAJE

# DEFINIR PRE-PROCESAMIENTO DE DATOS(TRANSFORMACION)
# LA BASE DE DATOS TIENE IMAGENES PILIMAGE EN EL RANGO [0,1]
# LAS TRANSFORMAMOS A TENSORES DE RANGO NORMALIZADO [-1,1]
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10 : 60000 32x32 IMAGENES A COLOR EN 10 CLASES, CON 6000 IMAGENES POR CLASE
train_dataset = torchvision.datasets.CIFAR10(root='./data' ,train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data' ,train=False,
                                             download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle= False)

# OBJETOS A CLASIFICAR
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')

# GRAFICAR CONM MATPLOTLIB
def imshow(img):
    img = img / 2 + 0.5 # UNNORMALIZE
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

# OBTENER A ALGNAS IMAGENES PARA ENTRENAR
dataiter = iter(train_loader)
images, labels = next(dataiter)

# MOSTRAR CONTENIDO DE IMAGENES
imshow(torchvision.utils.make_gird(images))

# RED NEURONAL CONVOLUCIONAL
class ConvNet(nn.Module):
    def __init__(self):
        super(ConNet, self).__init__()
        self.conv1 = nn
        self.pool = nn.MaxPool2d(2,2)


