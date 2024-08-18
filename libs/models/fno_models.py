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
