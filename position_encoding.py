import torch
import torch.nn as nn

class PositionEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        '''
        d_model: dimesion of the model, number of word embedding values per token
        max_len: max number of tokens our SimpleGPT can process -- input and output combined
        '''

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        # torch.arange - create a sequence of numbers, values from the interval [start, end)
        # unsqueeze(1) - turn the sequence to a column matrix
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        # step=2 - based on PE formula it is multiple by 2
        # For d_model=2, the embedding_index is just tensor[.0]

        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        # for row ':' will consider all rows, and for columns '0::2' means start from 0 and 2 means every other column after that 
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        # to move pe to GPU

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :]