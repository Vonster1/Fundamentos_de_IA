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
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5,5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CORRER EL MODELO EN EL GPU
model = ConvNet().to(device)

# USAR CROSS-ENTROPY COMO COSTO Y GRADIENTE ESTOCÁSTICO COMO OPTIMIZADOR
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# ITERACIONES (ENTRENAMIENTO)
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backeward()
        optimizer.step()

        if(i+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epoch}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Entrenaimiento completo')

# GUARDAR RESULTADO DEL MODELO (PARAMETROS)
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

# PROBAR EL MODELO
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if(label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Precision del modelo: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Precision de {classes[i]}: {acc}%')





