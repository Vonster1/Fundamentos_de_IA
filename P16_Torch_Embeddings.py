#========================================
# EMBEDDINGS CON PYTORCH Y LIGHTNING
# ALEX BRAULIO VON STERNENFELS HERNANDEZ
# FUNDAMENTOS DE IA ESFM IPN
#========================================
import torch
import torch.nn as nn 
from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CREAR LOS DATOS DE ENTRENAMIENTO DE LA RED
inputs = torch.tensor([[1.,0.,0.,0.],
                       [0.,1.,0.,0.],
                       [0.,0.,1.,0.],
                       [0.,0.,0.,1.]])

labels = torch.tensor([[0.,1.,0.,0.],
                       [0.,0.,1.,0.],
                       [0.,0.,0.,1.],
                       [1.,0.,0.,0.]])
dataset = TensorDataset(inputs,labels)
dataloader = DataLoader(dataset)

# EMBEDDING CON LINEAR
class WordEmbeddingWithLinear(L.LightningModule):

    def __init__(self):
        super().__init__()

        self.input_to_hidden = nn.Linear(in_features=4, out_features=2, bias=False)
        self.hidden_to_output = nn.Linear(in_features=2, out_features=4, bias=False)
        self.loss = nn.CrrossEntropyLoss()

    def forward(self, input):
        hidden = self.input_to_hidden(input)
        output_values = self.hidden_to_output(hidden)
        return(output_values)

    def configure_optimizers(self):
        return Adam(self.parametes(), lr=0.1)

    def training_step(self,batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i)
        return loss

# CREAR LA RED
modelLinear = WordEmbeddingWithLinear()

# MOSTRAR PARAMETROS ANTES DE APRENDIZAJE
data = {
        "w1": modelLinear.input_to_hidden.weight.detach()[0].numpy(),
        "w1": modelLinear.input_to_hidden.weight.detach()[1].numpy(),
        "token": ["Dunas2", "es", "grandiosa", "Godzilla"],
        "input": ["input1", " input2", " input3", "input4"]
}
df = pd.DataFrame(data)
# GRAFICAR CON SCATTERPLOT
sns.scatterplot(data:df, x="w1", y="w2")



