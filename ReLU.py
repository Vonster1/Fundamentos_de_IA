#==================================================================================
#                         TITULO
# ALEX BRAULIO VON STERNENFELS HERNÁNDEZ
# FUNDAMENTOS DE IA. ESFM IPN
#==================================================================================
import numpy as np
import matplotlib.pyplot as plt

relu = lambda x : np.where(x>=0, x, 0)
y = np.linspace(-10,10,1000)
plt.plot(y,relu(y), 'b', label = 'linspace(-10,10,100)')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('ReLU')
plt.xticks([-4,-3,-2,-1,0,1,2,3,4])
plt.yticks([-4,-3,-2,-1,0,1,2,3,4])
plt.ylim(-4,4)
plt.xlim(-4,4)
plt.show()
