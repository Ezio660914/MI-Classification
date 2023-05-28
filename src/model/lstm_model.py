# -*- coding: utf-8 -*-
import os
import sys

import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class BiLSTM_Model(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=input_channels,
            batch_first=True,
            bidirectional=False
        )
        self.bi_lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=input_channels,
            batch_first=True,
            bidirectional=True
        )
        self.fc_stack = nn.Sequential(
            nn.Linear(input_channels * 2, 4),
            nn.ReLU(),
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.bi_lstm(x)
        x = x[:, -1, :]  # 只取最后一个
        x = self.fc_stack(x)
        return x


def main():
    import torch
    from torchinfo import summary
    model = BiLSTM_Model(1)
    t = torch.rand((64, 24, 1))
    summary(model, input_data=t, device='cpu')
    # model = torch.jit.trace(model, t)
    # model.save('model.pt')


if __name__ == '__main__':
    main()
