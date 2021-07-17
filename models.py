import numpy as np
import torch
from torch.nn import functional as F

# linear regression model
class LinearRegression(torch.nn.Module):

    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

# polynomial regression model
class PolynomialRegression(torch.nn.Module):

    def __init__(self, degree):
        super(PolynomialRegression, self).__init__()
        self.poly = torch.nn.Linear(degree, 1, bias=True)

    def forward(self, x):
        out = self.poly(x)
        return out

# logistic regression model
class LogisticRegression(torch.nn.Module):

    def __init__(self, inputSize):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        outputs = self.linear(x)
        outputs = F.sigmoid(output)
        return outputs

