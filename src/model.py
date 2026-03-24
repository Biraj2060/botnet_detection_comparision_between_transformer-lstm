# src/model.py
import torch
import torch.nn as nn
import math


class BotnetTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        d_model=256,
        n_heads=8,
        n_layers=3,
        dropout=0.3,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        self.pos_enc     = PositionalEncoding(d_model, dropout)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout  = nn.Dropout(dropout)
        pe            = torch.zeros(max_len, d_model)
        position      = torch.arange(0, max_len).unsqueeze(1).float()
        div_term      = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2]  = torch.sin(position * div_term)
        pe[:, 1::2]  = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
