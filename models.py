import numpy as np
import torch
from torch.nn import functional as F

# linear regression model
class LinearRegression(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

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

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        outputs = self.linear(x)
        outputs = F.sigmoid(outputs)
        return outputs





# DNN model
class DNN(torch.nn.Module):

    def __init__(self, input_size=512, num_classes=10, num_layers=2, activation=nn.ReLU):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.layers = _add_layers()

        def _add_layers(self):
            layer = []
            layer += [nn.Layer(self.input_size,self.hidden_size)]
            layer += [activation]
            hidden_size = int(self.input_size / 4)
            for i in  range(self.num_layers - 2):
                layer+= [nn.Linear(hidden_size, hidden_size)]
                layer += [activation]
                if hidden_size > 32:
                    hidden_size /= 4
            layer += [nn.Linear(hidden_size, num_classes)]
            return nn.Sequential(*layer)

        def forward(self, x):
            x = x.reshape(x.size(0), self.input_size)
            return self.layers(x)

# helper class for passing values
class Same():

    def __init__(self):
        super(Same, self).__init__()

    def forward(self, x):
        return x
    
# helper class for CNN
class ConvUnit(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU(), batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class CNN(nn.Module):

    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm), 
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output