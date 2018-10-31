import torch.nn as nn


class LinearNetwork(nn.Module):
    def __init__(self, input_dims, output_dims):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        return self.l1(x)

class CartPoleNetwork(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(4, dim1),
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2, 2)
        )

    def forward(self, x):
        return self.layers(x)

class LunarLanderNetwork(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(8, dim),
            nn.ReLU(),
            nn.Linear(dim, 4)
        )

    def forward(self, x):
        return self.layers(x)

class PongNetwork(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(128, dim1),
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2, 4)
        )

    def forward(self, x):
        return self.layers(x)
