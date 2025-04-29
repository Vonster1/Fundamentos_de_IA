#=============================================
#      PYTORCH BÁSICO
#  ALEX BRAULIO VON STERNENFELS HERNÁNDEZ
# FUNDAMENTOS DE IA ESFM IPN
#=============================================

# MÓDULO DE PYTORCH
import torch

# EN PYTORH todo ESTÁ BASADO EN OPERACIONES TENSORIALES
# UN TENSOR VIVE EN Rn x Rm x Rp ... etc

# ESCALAR VACÍO (TRAE BASURA)
x = torch.empty(3)
print(x)

# TENSOR EN R2 x R3
x = torch.empty(2,3)
print(x)

# TENSOR EN R2 x R2 x R3
x = torch.empty(2,2,3)
print(x)

# torch.rand(size): números aleatorios [0,1]
# TENSOR DE NÚMEROS ALEATORIOS DE R5 x R3
x = torch.rand(5,3)
print(x)

# TENSOR DE R5 x R3 LLENO DE CEROS
x = torch.zeros(5,3)
print(x)

# CHECAR TAMAÑO (LISTA CON DIMENSIONES)
print(x.size())

# CHECAR TIPO DE DATOS (default es float32)
print(x.dtype)

# ESPECIFICANDO TIPO DE DATOS
x = torch.zeros(5,3, dtype = torch.float16)
print(x)
print(x.dtype)

# CONSTRUÍR VECTOR CON DATOS
x = torch.tensor([5.5,3])
print(x.size())

# VECTOR OPTIMIZABLE (VARIABLES DEL GRADIENTE)
x = torch.tensor([5.5,3], requires_grad=True)

# SUMA DE TENSORES
y = torch.rand(2,2)
x = torch.rand(2,2)
z = x + y
z = torch.add(x,y)
print(z)
y.add_(x)
print(y)

# RESTA DE TENSORES
z = x - y
z = torch.sub(x,y)
print(z)

# MULTIPLICACIÓN
z = x * y
z = torch.mul(x,y)
print(z)

# DIVISIÓN 
z = x / y
z = torch.div(x,y)
print(z)

# REBANADAS
x = torch.rand(5,2)
print(x)
print(x[:,0])  # TODOS LOS RENGLONES, COLUMNA 0
print(x[1,:])  # RENGLÓN 1, TODAS LAS COLUMNAS
print(x[1,1])  # ELEMENTO EN (1,1)
print(x[1,1].item()) # VALOR DEL ELEMENTO EN (1,1)

# CAMBIAR FORMA CON torch.view()
x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1,8) # -1: SE INFIERE DE LAS OTRAS DIMENSIONES
# SI -1 PYRTOCH DETERMINARÁ AUTOMÁTICAMENTE EL TAMAÑO NECESARIO
print(x.size(), y.size(), z.size())

# CONVERTIR UN TENSOR EN ARREGLO DE NUMPY Y VICEVERSA
a = torch.ones(5)
b = a.numpy()
print(b)
print(type(b))

# LE SUMA 1 A TODAS LAS ENTRADAS
a.add_(1)
print(a)
print(b)

# DE NUMPY A TORCH
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

# LE SUMA 1 A TODAS LA ENTRADAS DE a
a += 1
print(a)
print(b)

# DE CPU A GPU (SI HAY CUDA)
if torch.cuda.is_available():
    device = torch.device("cuda")   # LA TARJETA DE VIDEO CON CUDA
    print("Tengo GPU " + str(device))
    y_d = torch.ones_like(x,device=device)  # CREAR TENSOR EN EL GPU
    x_d = x.to(device)  
    z_d = x_d + y_d

    # z = z_d.numpy() # numpy no maneja tensores en el GPU
    # de vuelta al GPU
    z = z_d.to("cpu")
    z = z.numpy()
    print(z)



