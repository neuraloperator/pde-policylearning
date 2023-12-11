import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import operator
from neuralop.models import FNO2d
from timeit import default_timer


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2dObserverOld(nn.Module):
    def __init__(self, modes1, modes2, width, use_v_plane=False):
        super(FNO2dObserverOld, self).__init__()
        """
        This is the old FNO2d used in PDE bench, etc.
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.use_v_plane = use_v_plane
        self.padding = 9 # Pad the domain if input is non-periodic
        # input channel is 4: (a(x, y), x1, x2, y)
        self.input_channel_num = 4 if use_v_plane else 3
        self.fc0 = nn.Linear(self.input_channel_num, self.width) 

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, p_plane, v_plane):
        grid = self.get_grid(p_plane.shape, p_plane.device)
        if self.use_v_plane:
            p_plane = torch.cat((p_plane, v_plane, grid), dim=-1)
        else:
            p_plane = torch.cat((p_plane, grid), dim=-1)
        p_plane = self.fc0(p_plane)
        p_plane = p_plane.permute(0, 3, 1, 2)
        p_plane = F.pad(p_plane, [0,self.padding, 0,self.padding])

        x1 = self.conv0(p_plane)
        x2 = self.w0(p_plane)
        p_plane = x1 + x2
        p_plane = F.gelu(p_plane)

        x1 = self.conv1(p_plane)
        x2 = self.w1(p_plane)
        p_plane = x1 + x2
        p_plane = F.gelu(p_plane)

        x1 = self.conv2(p_plane)
        x2 = self.w2(p_plane)
        p_plane = x1 + x2
        p_plane = F.gelu(p_plane)

        x1 = self.conv3(p_plane)
        x2 = self.w3(p_plane)
        p_plane = x1 + x2

        p_plane = p_plane[..., :-self.padding, :-self.padding]
        p_plane = p_plane.permute(0, 2, 3, 1)
        p_plane = self.fc1(p_plane)
        p_plane = F.gelu(p_plane)
        p_plane = self.fc2(p_plane)
        return p_plane
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class FNO2dObserver(nn.Module):
    def __init__(self, modes1, modes2, width, use_v_plane=False):
        super(FNO2dObserver, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.use_v_plane = use_v_plane
        self.padding = 9 # Pad the domain if input is non-periodic
        # input channel is 4: (a(x, y), x1, x2, y)
        self.input_channel_num = 4 if use_v_plane else 3
        self.fno2d = FNO2d(self.modes1, self.modes2, self.width, in_channels=self.input_channel_num, out_channels=1)

    def forward(self, p_plane, v_plane):
        grid = self.get_grid(p_plane.shape, p_plane.device)
        if self.use_v_plane:
            p_plane = torch.cat((p_plane, v_plane, grid), dim=-1)
        else:
            p_plane = torch.cat((p_plane, grid), dim=-1)
        p_plane = p_plane.permute(0, 3, 1, 2)
        p_plane = self.fno2d(p_plane)
        return p_plane
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
