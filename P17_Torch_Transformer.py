#=========================================
# TRANSFORMER CON PYTHON
# ALEX BRAULI VON STERNENFELS HERNANDEZ
# FUNDAMENTOS DE IA ESFM IPN
#=========================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

# CELULA DE ATENCION (MULTIPLES)
class MultiHeadAttention(nn.Module):
    # CONSTRUCTOR 
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_ heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    # PRODUCTO ESCALAR ESCALADO
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        attn_

       # MINUTO 0:15 

if mask is not None:
    attn_scores = attn_scores.masked_fill(mask==0, -1e9)
attn_
# MINUTO 0:15 






