import torch
import torch.nn as nn

from torch.optim import Adam

import lightning as L

from position_encoding import PositionEncoding
from attention import Attention


class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        # num_tokens: max number of tokens in the vocab
        super().__init__()
        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)
        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0)), device=self.device))
        mask = mask == 0

        self_attention_values = self.self_attention(position_encoded,
                                                    position_encoded,
                                                    position_encoded,
                                                    mask=mask)

        residual_connection_layers = position_encoded + self_attention_values
        fc_layer_output = self.fc_layer(residual_connection_layers)

        return fc_layer_output
    
    def configure_optimizers(self):
        return Adam(self. parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self. forward (input_tokens[0])
        loss = self. loss(output, labels[0])
        return loss