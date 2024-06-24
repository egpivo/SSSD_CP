import torch
import torch.nn as nn


class CustomLayerNorm(nn.Module):
    def __init__(self, length, eps=1e-5):
        super().__init__()
        self.length = length
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, 1, length))
        self.bias = nn.Parameter(torch.zeros(1, 1, length))

        # Register buffers for mean and variance
        self.register_buffer("mean", None)
        self.register_buffer("var", None)

    def forward(self, x):
        # Compute mean and variance across the last two dimensions (channel and length)
        mean = x.mean(dim=[1, 2], keepdim=True)
        var = x.var(dim=[1, 2], keepdim=True, unbiased=False)

        # Store the computed mean and variance in buffers
        self.mean = mean
        self.var = var

        # Normalize using mean and variance
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        x = self.weight * x + self.bias
        return x

    def denormalize(self, x):
        # Denormalize using stored mean and variance
        mean = self.mean
        var = self.var

        x = (x - self.bias) / self.weight
        x = x * torch.sqrt(var + self.eps) + mean
        return x
