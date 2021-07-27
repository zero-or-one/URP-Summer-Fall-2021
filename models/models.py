import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Variable


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
class DNN(nn.Module):

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
class Same(nn.Module):

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
        model += [nn.ReLU()]
        super(ConvUnit, self).__init__(*model)

class CNN(nn.Module):

    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(CNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            ConvUnit(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            ConvUnit(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            ConvUnit(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Same(),
            ConvUnit(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            ConvUnit(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            ConvUnit(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Same(),
            ConvUnit(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            ConvUnit(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output
'''
class ResNet18(nn.Module):
    def __init__(self, filters_percentage=1.0, n_channels = 3, num_classes=10, block=_ResBlock, num_blocks=[2,2,2,2], n_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(n_channels,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, int(64*filters_percentage), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*filters_percentage), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*filters_percentage), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*filters_percentage), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512*filters_percentage)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
'''