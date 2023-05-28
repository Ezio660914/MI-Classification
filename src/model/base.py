# -*- coding: utf-8 -*-
import os
import sys

from torch import nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    pass


if __name__ == "__main__":
    main()


class LambdaLayer(nn.Module):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    def forward(self, x):
        return self.expr(x)
