# src/cnn_lstm.py
import torch
import torch.nn as nn


class BotnetCNNLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout)
        self.relu  = nn.ReLU()
        self.lstm  = nn.LSTM(
            input_size    = 128,
            hidden_size   = 128,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)                  # (batch, input_dim, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)                         # (batch, 128, seq_len//2)
        x = self.drop1(x)
        x = x.permute(0, 2, 1)                  # (batch, seq_len//2, 128)
        x, _ = self.lstm(x)
        x = x[:, -1, :]                          # last time step
        return self.classifier(x)
