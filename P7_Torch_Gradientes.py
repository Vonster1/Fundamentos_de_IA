#==============================================
# DIFERENCIACION AUTOMATICA AUTOGRAD
# ALEX BRAULIO VON STERNENFELS HERNANDEZ
# FUNDAMENTOS DE IA ESFM IPN
#==============================================
import torch 
# REQUIRES_GRAD = TRUE GENERA FUNCIONES GRADIENTE PARA LAS OPERACIONES QUE SE HACEN CON ESE TENSOR
x = torch.rand(3, requires_grad= True)
y = x + 2

# Y = Y(X) TIENE UN GRAD_FN ASOCIADO
print(x)
print(y)
print(y.grad_fn)

# Z = Z(Y)  =  Z(Y(X))
z = y * y * 3
print(z) 
z = z.mean()
print(z)

#  CALCULO DEL GRADIENTE CON RETROPROPAGACION
z.backward()
print(x.grad)

# TORCH.AUTOGRAD SE BASA EN REGLA DE LA CADENA
x = torch.randn(3, requires_grad=True)
y = x * 2
for _ in range(10):
    y = y * 2
    print(y)
    print(y.shape)

# EVALUAR "GRADIENTE" DY/DX EN V
v = torch.tensor([0.1,1.0,0.0001], dtype=torch.float32)
y.backward(v)
print(x.grad)

# DECIRLE A UN TENSOR QUE DEJE GENEREAR GRADIENTES
a = torch.randn(2,2)
print(a.requires_grad)
b = ((a*3)/(a-1))
print(b.grad_fn)
# CON GRADIENTE 
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)
x=torch.randn(3,requires_grad=True)
print(a.requires_grad)
a = torch.randn(2,2,requires_grad=True)
print(a.requires_grad)
# SIN GRADIENTE
b=a.detach()
print(b.requires_grad)

# CON ENVOLTURA QUE LE QUITA EL GRADIENTE
a = torch.randn(2,2,requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)

# BACKWARD() ACUMULA EL GRADIENTE EN .GRAD
# .ZERO_() LIMPIA EL GRADIENTE ANTES DE COMENZAR
weights = torch.ones(4,requires_grad=True)
print(weights)

# EPOCH: PASO DE OPTIMIZACION
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    with torch.no_grad():
        weights -= 0.1 * weights.grad
    weights.grad.zero_()
model_output = (weights*3).sum()
print(weights)
print(model_output)


