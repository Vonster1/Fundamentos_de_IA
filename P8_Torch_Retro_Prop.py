#================================================
# GRADIENTES SIMPLES CON PYTORCH
# ALEX BRAULIO VON STERNENEFELS HERNANDEZ
# ESFM IPN
#================================================
import torch
x = torch.tensor(1.0)
y = torch.tensor(2.0)

# VARIABLE DE DIFERENCIACION (D/DW) -> REQUIRES_GRAD=TRUE
w = torch.tensor(1.0, requires_grad=True)

# EVALUACION CALCULO DE COSTO
y_predicted = w*x
loss = (y_predicted - y)**2
print(loss)

# RFETROPROPAGACION PARA CALCULAR GRADIENTE
loss.backward()
print(w.grad)

# NUEVOS COEFICIENTES (DESCENSO DE GRADIENTE)
# REPETIR EVALUACION Y RETROPROPAGACION
with torch.no_grad():
    w-=0.01*w.grad
w.grad.zero_()
print(w)

