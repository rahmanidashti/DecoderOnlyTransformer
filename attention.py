import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    
    def __init__(self, d_model=2):

        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        # nn.Linear - creating Weight matrix and math computation
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = 0
        self.col_dim = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        q = self.W_q(encodings_for_q) 
        # Q: n, d_k - (WE+PE): n, d * W_q: d, d_k
        k = self.W_k(encodings_for_k) # n, d_k
        v = self.W_v(encodings_for_v) # n, d_v

        # attention(Q, K, V) = softmax(QK.T / Sqrt(d_k)) V

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        # transpose() - transpose the selected dims (dim0, dim1) of the tensor/matrix
        scaled_sims = sims / torch.tensor(k.size(self.col_dim) ** 0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
            # masking is used to prevent early tokens from cheating and looking at later tokens
        attention_precents = F.softmax(scaled_sims, dim=self.col_dim)
        # determining the precentages of influence that each token should have on the others
        attention_scores = torch.matmul(attention_precents, v)
        return attention_scores