# -*- coding: utf-8 -*-
import os
import sys

import torch.nn as nn
from source.models.base import LambdaLayer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class CNNLSTM_Model(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            LambdaLayer(lambda x: x.permute(0, 2, 1)),
            nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=4, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            LambdaLayer(lambda x: x.permute(0, 2, 1)),
        )
        self.lstm_layer = nn.Sequential(
            nn.LSTM(
                input_size=32,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            ),
            LambdaLayer(lambda x: x[0][:, -1, :]),
            nn.Dropout(0.2),
        )
        self.dense_layer = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, inputs):
        x = self.cnn_layer(inputs)
        x = self.lstm_layer(x)
        x = self.dense_layer(x)
        x = inputs[:, [-1], 0] + x
        return x


def main():
    import torch
    from torchinfo import summary
    model = CNNLSTM_Model(3)
    t = torch.rand((128, 24, 3))
    summary(model, input_data=t, device='cpu')
    # model = torch.jit.trace(model, t)
    # model.save('model.pt')


if __name__ == '__main__':
    main()
