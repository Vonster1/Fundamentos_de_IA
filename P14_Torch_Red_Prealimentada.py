#==========================================
#  RED PREALIMENTADA (FEED FORWARD)
#  ALEX BRAULIIO VON STERNENFELS HERNANDEZ
#  FUNDAMENTOS DE IA ESFM IPN
#==========================================
# MODULOS NECESARIOS
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# CONFIGURACION DEL GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HIPER-PARAMETROS
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST BASE DE DATOS 
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          transform=transform.ToTensor())
# CARGA DE DATOS
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
examples = iter(test_loader)
example_data, example_targets = next(examples)

# MOSTRAR DATOS EN UNA IMAGEN
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

# RED NEURONAL COMPLETAMENTE CONECTADA CON UNA CAPA OCULTA
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# CORRER MODELO EN EL GPU
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# OPTIMIZACION Y CALCULO DE ERROR
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ENTRENAR EL MODEL
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        # EVALUCION
        outputs  = model(images)
        loss = criterion(outputs,labels) 
        # CALCULO DEL GRADIENTE Y OPTIMIZACION 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # DIAGNOSTICO
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# CHECAR EL MODELO
# EN FASE DE PRUEBA, NO REQUERIMOS CALCULAR GRADIENTES
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) 
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    # PRECISION
    acc = 100.0 *n_correct / n_samples
    print(f'Accurancy of the network on the 10000 test test images: {acc} %')
        


