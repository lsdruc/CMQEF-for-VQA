from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims,dropout=0.17):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.PReLU())
        layers.append(nn.Dropout(p=dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class FCNet1(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims,dropout=0.17):
        super(FCNet1, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(p=dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.Sigmoid())
        layers.append(nn.Dropout(p=dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class GTH(nn.Module):
    """Simple class for Gated Tanh
    """
    def __init__(self, in_dim, out_dim):
        super(GTH, self).__init__()

        self.nonlinear = FCNet([in_dim, out_dim])
        self.gate = FCNet1([in_dim, out_dim])

    def forward(self, x):
        x_proj = self.nonlinear(x)
        gate = self.gate(x)
        x_proj = x_proj*gate
        return x_proj
if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print('============')
    fc2 = FCNet([10, 20])
    print(fc2)