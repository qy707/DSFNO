# This file contains network modules to build downscaling FNO model.
# Author: Qidong Yang
# Date: 2022-08-26

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


##### Constraint Layer #####

class SoftmaxConstraint(nn.Module):
    def __init__(self, exp_factor=1):
        super(SoftmaxConstraint, self).__init__()

        self.exp_factor = exp_factor

    def forward(self, x, y, upsample_factor):
        
        # x: (n_batch, in_channel, size_x, size_y)
        # y: (n_batch, in_channel, upsample_factor * size_x, upsample_factor * size_y)

        y = torch.exp(y * self.exp_factor)
        # (n_batch, in_channel, upsample_factor * size_x, upsample_factor * size_y)
        avg_y = F.avg_pool2d(y, kernel_size=upsample_factor)
        # (n_batch, in_channel, size_x, size_y)
        out = y * torch.kron(x/avg_y, torch.ones((upsample_factor, upsample_factor)).cuda())
        # (n_batch, in_channel, upsample_factor * size_x, upsample_factor * size_y)

        return out


##### Model Modules #####

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        # 2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        # in_channels: the number of input channels
        # out_channels: the number of output channels
        # modes1: the number of modes used for dimension 1, at most floor(N/2) + 1
        # modes2: the number of modes used for dimension 2, at most floor(N/2) + 1

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (2 * in_channels))**(1.0 / 2.0)  # initialization scale
        self.weights1 = nn.Parameter(self.scale * (torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (n_batch, in_channels, x, y), (in_channels, out_channels, x, y) -> (n_batch, out_channels, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, upsample_factor):

        # x: input function (n_batch, in_channels, n_dim1, n_dim2)
        # upsample_factor: upsample factor for n_dim1 and n_dim2 
        # dim1: scaled n_dim1 i.e. n_dim1 * upsample_factor
        # dim2: scaled n_dim2 i.e. n_dim2 * upsample_factor

        # assert self.modes1 < dim1 // 2 and self.modes2 < dim2 // 2

        batch_size = x.shape[0]
        dim1 = int(upsample_factor * x.shape[2])
        dim2 = int(upsample_factor * x.shape[3])

        # Compute Fourier coeffcients
        x_ft = torch.fft.rfft2(x)
        # (n_batch, in_channels, n_dim1, n_dim2//2 + 1)

        modes1_use = int(min(self.modes1, dim1 // 2, x.shape[2] // 2))
        modes2_use = int(min(self.modes2, dim2 // 2 + 1, x.shape[3] // 2 + 1))

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels, dim1, dim2 // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :modes1_use, :modes2_use] = \
                self.compl_mul2d(x_ft[:, :, :modes1_use, :modes2_use], self.weights1[:, :, :modes1_use, :modes2_use])
        out_ft[:, :, -modes1_use:, :modes2_use] = \
                self.compl_mul2d(x_ft[:, :, -modes1_use:, :modes2_use], self.weights2[:, :, -modes1_use:, :modes2_use])

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(dim1, dim2)) * upsample_factor * upsample_factor
        # (n_batch, out_channels, dim1, dim2)

        return x


class OperatorBlock(nn.Module):
    def __init__(self, n_channels, modes1, modes2, activation):
        super(OperatorBlock, self).__init__()

        self.n_channels = n_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.activation = activation

        self.conv = SpectralConv2d(self.n_channels, self.n_channels, self.modes1, self.modes2)

        self.w = nn.Conv2d(self.n_channels, self.n_channels, 1)

    def forward(self, x):

        x1 = self.conv(x, 1)
        x2 = self.w(x)
        x = x1 + x2

        if self.activation == True:
            x = F.gelu(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()

        self.n_channels = n_channels

        self.conv1 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + residual

        return out


##### Downscaling Model #####

class DSFNO(nn.Module):
    def __init__(self, in_channel=1, n_channels=64, n_residual_blocks=4, n_operator_blocks=2, modes=18, apply_constraint=True):
        super(DSFNO, self).__init__()

        self.in_channel = in_channel
        self.n_channels = n_channels
        self.n_residual_blocks = n_residual_blocks
        self.n_operator_blocks = n_operator_blocks
        self.modes = modes
        self.apply_constraint = apply_constraint

        # First Conv Layer
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channel, self.n_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        # Residual Blocks
        self.res_blocks = nn.ModuleList()
        for i in range(self.n_residual_blocks):
            self.res_blocks.append(ResidualBlock(self.n_channels))

        # Second Conv Layer
        self.conv2 = nn.Sequential(nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        # FNO Blocks
        self.fno_blocks = nn.ModuleList()
        for i in range(self.n_operator_blocks - 1):
            self.fno_blocks.append(OperatorBlock(self.n_channels, self.modes, self.modes, True))
        self.fno_blocks.append(OperatorBlock(self.n_channels, self.modes, self.modes, False))

        # Channel Linear Layers
        self.fc1 = nn.Linear(self.n_channels, 128)
        self.fc2 = nn.Linear(128, self.in_channel)

        # Constraint Layer
        if self.apply_constraint:
            self.constraint = SoftmaxConstraint()

    def forward(self, x, upsample_factor):

        # x: (n_batch, size_x, size_y, 1)

        size_x = x.size(1)
        size_y = x.size(2)

        x = x.permute(0, 3, 1, 2)
        # (n_batch, 1, size_x, size_y)

        out = self.conv1(x)
        # (n_batch, n_channels, size_x, size_y)

        for layer in self.res_blocks:
            out = layer(out)
        # (n_batch, n_channels, size_x, size_y)

        out = self.conv2(out)
        # (n_batch, n_channels, size_x, size_y)

        out = torch.nn.functional.interpolate(out, scale_factor=upsample_factor, mode='bicubic', align_corners=False)
        # (n_batch, n_channels, upsample_factor * size_x, upsample_factor * size_y)

        for layer in self.fno_blocks:
            out = layer(out)
        # (n_batch, n_channels, upsample_factor * size_x, upsample_factor * size_y)

        out = out.permute(0, 2, 3, 1)
        # (n_batch, upsample_factor * size_x, upsample_factor * size_y, n_channels)

        out = self.fc1(out)
        out = F.gelu(out)
        # (n_batch, upsample_factor * size_x, upsample_factor * size_y, 128)

        out = self.fc2(out)
        # (n_batch, upsample_factor * size_x, upsample_factor * size_y, 1)
        
        if self.apply_constraint:
            out = out.permute(0, 3, 1, 2)
            # (n_batch, 1, upsample_factor * size_x, upsample_factor * size_y)

            out = self.constraint(x, out, upsample_factor)
            # (n_batch, 1, upsample_factor * size_x, upsample_factor * size_y)
        
            out = out.permute(0, 2, 3, 1)
            # (n_batch, upsample_factor * size_x, upsample_factor * size_y, 1)

        return out



