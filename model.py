##Fully connected PINN with Fourier Feature Mapping that eliminates spectral bias
import torch
import torch.nn as nn
import numpy as np
class PhaseFieldNet(nn.Module):

    def __init__(self, layers: list[int], use_fourier: bool = True, s: float = 1.0):
        super().__init__()
        self.use_fourier = use_ffm
        in_dim = layers[0]

        if use_ffm:
            n_freq = 32
            B = torch.randn(in_dim, n_freq) * s
            self.register_buffer("B", B)
            first_in = 2 * n_freq
        else:
            first_in = in_dim

        dims = [first_in] + layers[1:]
        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_fourier:
            proj = x @ self.B
            x = torch.cat([torch.sin(2 * np.pi * proj),
                            torch.cos(2 * np.pi * proj)], dim=-1)
        return self.net(x)
