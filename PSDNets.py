import torch
from torch import nn
import torch.nn.functional as F


class ICNN(nn.Module):
    def __init__(self, layer_sizes, activation=F.softplus, eps=0.01):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1,len(layer_sizes)-1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l)) for l in layer_sizes[1:]])

        self.act = activation
        self.reset_parameters()
        self.eps = eps

    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)
        for i,b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, t, x):
        shape = x.shape
        nsim, dim = shape[0], shape[-1]
        x = x.reshape(nsim, dim)
        if len(t.shape) == 0:
            t = torch.ones((nsim, 1)).to(x.device) * t
        aug_x = torch.cat((x, t), dim=1)
        if nsim > 1:
            aug_x.squeeze()

        z = F.linear(aug_x, self.W[0], self.bias[0])
        z = self.act(z)

        for W,b,U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(aug_x, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
            z = self.act(z)

        return ((F.linear(aug_x, self.W[-1], self.bias[-1]) + F.linear(z, F.softplus(self.U[-1])) / self.U[-1].shape[0]) + self.eps*(aug_x**2).sum(1)[:,None]).reshape(nsim, 1, 1)
