import torch
import torch.nn as nn
from .basics import SpectralConv3d
from .utils import add_padding, remove_padding, _get_act
import numpy as np
from libs.DINo.network import MultiplicativeNet
from torch.nn.parameter import Parameter
import math
from torch.nn import init
import torch.nn.functional as F
from torch import nn, Tensor


class MultiplicativeNet(nn.Module):
    __constants__ = ['in1_features', 'in2_features', 'out_features']
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in1_features: int, in2_features: int, out_features: int, device=None, dtype=None) -> None:
        """
        x2T A + B x1
        x2: code, x1: spatial coordinates
        """
        super(MultiplicativeNet, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.A = Parameter(torch.empty(out_features, in2_features))
        self.B = Parameter(torch.empty(out_features, in1_features))
        self.bias = Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in1_features)
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        # input1: [N, X, Y, T, O]
        # input2: [N, O]
        # W: o, i, j
        # B: o, i
        # A: o, j
        # bias: o
        res = 0
        if len(input2.shape) < 2:
            input2 = input2.unsqueeze(-1)
        bias_code = torch.einsum('bj,oj->bo', input2, self.A)  # [N, O]
        bias_code = bias_code.unsqueeze(1).unsqueeze(1).unsqueeze(1) # [N, T, X, Y, O]

        linear_trans_2 = torch.einsum('bthwi,oi->bthwo', input1, self.B)  # [N, T, X, Y, O]
       
        res += linear_trans_2 
        res += bias_code
        res += self.bias
        return res
        
    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None)
        
        
class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.
    Adapted from https://github.com/boschresearch/multiplicative-filter-networks
    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """
    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers):
        super().__init__()
        self.first = 3
        self.bilinear = nn.ModuleList(
            [MultiplicativeNet(in_size, code_size, hidden_size)] +
            [MultiplicativeNet(hidden_size, code_size, hidden_size) for _ in range(int(n_layers))]
        )
        self.output_bilinear = nn.Linear(hidden_size, out_size)
        return

    def forward(self, x, code):
        # shape: [bs, t, h, w, 1, feat]
        out = self.filters[0](x) * self.bilinear[0](x*0., code)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.bilinear[i](out, code)
        out = self.output_bilinear(out)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out, x


class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    Adapted from https://github.com/boschresearch/multiplicative-filter-networks
    """
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.weight = Parameter(torch.empty((out_features // 2, in_features)))
        self.weight_scale = weight_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        linear_features = F.linear(x, self.weight * self.weight_scale)
        return torch.cat([torch.sin(linear_features), torch.cos(linear_features)], dim=-1)


class FourierNet(MFNBase):
    """
    Taken from https://github.com/boschresearch/multiplicative-filter-networks
    """
    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers=3, input_scale=256.0, **kwargs):
        super().__init__(in_size, hidden_size, code_size, out_size, n_layers)
        self.filters = nn.ModuleList(
                [FourierLayer(in_size, hidden_size // 2, input_scale / np.sqrt(n_layers + 1)) for _ in range(n_layers + 1)])
    
    def get_filters_weight(self):
        weights = list()
        for ftr in self.filters:
            weights.append(ftr.weight)
        return torch.cat(weights)


class PINObserver2d(nn.Module):
    def __init__(self, 
                 modes1, 
                 modes2, 
                 modes3,
                 width=16, 
                 fc_dim=128,
                 layers=None,
                 in_dim=4, out_dim=1,
                 act='gelu', 
                 pad_ratio=[0., 0.],
                 use_fourier_layer=False):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        '''
        super(PINObserver2d, self).__init__()

        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions.'

        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio
        self.in_dim = in_dim
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(self.in_dim, layers[0])
        self.use_fourier_layer = use_fourier_layer
        if use_fourier_layer:
            self.fourier_layer1 = FourierLayer(in_features=1, out_features=8, weight_scale=1.0)
        else:
            self.fourier_layer1 = None
        self.multiplicative_net1 = MultiplicativeNet(in1_features=layers[0], in2_features=1, out_features=layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.multiplicative_net2 = MultiplicativeNet(in1_features=layers[-1], in2_features=1, out_features=layers[-1])
        
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)

    def forward(self, x, re):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)
            re: (batchsize, x_grid, y_grid, t_grid, 1)

        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)
        '''
        re = re.float()
        if self.use_fourier_layer:
            fourier_re = self.fourier_layer1(re.unsqueeze(-1))
        size_z = x.shape[-2]
        if max(self.pad_ratio) > 0:
            num_pad = [round(size_z * i) for i in self.pad_ratio]
        else:
            num_pad = [0., 0.]
        length = len(self.ws)
        batchsize = x.shape[0]
        
        x = self.fc0(x)
        if self.use_fourier_layer:
            x = self.multiplicative_net1(x, fourier_re)
        else:
            x = self.multiplicative_net1(x, re)
        x = x.permute(0, 4, 1, 2, 3)
        x = add_padding(x, num_pad=num_pad)
        size_x, size_y, size_z = x.shape[-3], x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y, size_z)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = remove_padding(x, num_pad=num_pad)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.multiplicative_net2(x, re)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    

class PlanePredHead(nn.Module):
    def __init__(self, layers, modes1, modes2, modes3, fc_dim, out_dim, act):
        super(PlanePredHead, self).__init__()
        
        self.layers = layers
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])
        
        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])
        
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)
    
    def forward(self, x, num_pad, re, multiplicative_net2):
        length = len(self.ws)
        size_x, size_y, size_z = x.shape[-3], x.shape[-2], x.shape[-1]
        batchsize = x.shape[0]
        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y, size_z)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = remove_padding(x, num_pad=num_pad)
        x = x.permute(0, 2, 3, 4, 1)
        x = multiplicative_net2(x, re)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PINObserverFullField(nn.Module):
    def __init__(self, 
                 modes1, 
                 modes2, 
                 modes3,
                 width=16, 
                 fc_dim=128,
                 layers=None,
                 in_dim=4, out_dim=1,
                 act='gelu', 
                 pad_ratio=[0., 0.],
                 use_fourier_layer=False,):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        '''
        super(PINObserverFullField, self).__init__()

        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions.'

        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.max_re = 1000
        self.pad_ratio = pad_ratio
        self.in_dim = in_dim
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(self.in_dim, layers[0])
        self.use_fourier_layer = use_fourier_layer
        if use_fourier_layer:
            self.fourier_layer1 = FourierLayer(in_features=1, out_features=8, weight_scale=1.0)
        else:
            self.fourier_layer1 = None
        self.multiplicative_net1 = MultiplicativeNet(in1_features=layers[0], in2_features=1, out_features=layers[0])
        self.multiplicative_net2 = MultiplicativeNet(in1_features=layers[-1], in2_features=1, out_features=layers[-1])

        self.pred_net = PlanePredHead(layers=layers, modes1=self.modes1, modes2=self.modes2, modes3=self.modes3, 
                                      fc_dim=fc_dim, out_dim=out_dim, act=act)

    def forward(self, x, re):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 1)
            re: (batchsize, x_grid, y_grid, t_grid, 1)

        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)
        '''
        re = re.float() / self.max_re
        if self.use_fourier_layer:
            fourier_re = self.fourier_layer1(re.unsqueeze(-1))
        size_z = x.shape[-2]
        if max(self.pad_ratio) > 0:
            num_pad = [round(size_z * i) for i in self.pad_ratio]
        else:
            num_pad = [0., 0.]
        # length = len(self.ws)
        batchsize = x.shape[0]
        x = self.fc0(x)
        if self.use_fourier_layer:
            x = self.multiplicative_net1(x, fourier_re)
        else:
            x = self.multiplicative_net1(x, re)
        x = x.permute(0, 4, 1, 2, 3)
        x = add_padding(x, num_pad=num_pad)
        
        res_plane = self.pred_net(x, num_pad, re, self.multiplicative_net2)
        return res_plane
        
        
class PolicyModel2D(nn.Module):
    def __init__(self, 
                 modes1, 
                 modes2, 
                 modes3,
                 width=16, 
                 fc_dim=128,
                 layers=None,
                 in_dim=4, out_dim=1,
                 act='gelu', 
                 pad_ratio=[0., 0.],
                 use_fourier_layer=False,):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        '''
        super(PolicyModel2D, self).__init__()

        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions.'

        self.pad_ratio = pad_ratio
        self.max_re = 1000
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio
        self.in_dim = in_dim
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(self.in_dim, layers[0])
        self.use_fourier_layer = use_fourier_layer
        if use_fourier_layer:
            self.fourier_layer1 = FourierLayer(in_features=1, out_features=8, weight_scale=1.0)
        else:
            self.fourier_layer1 = None
        self.multiplicative_net1 = MultiplicativeNet(in1_features=layers[0], in2_features=1, out_features=layers[0])
        self.multiplicative_net2 = MultiplicativeNet(in1_features=layers[-1], in2_features=1, out_features=layers[-1])

        self.pred_net = PlanePredHead(layers=layers, modes1=self.modes1, modes2=self.modes2, modes3=self.modes3, 
                                      fc_dim=fc_dim, out_dim=out_dim, act=act)

    def forward(self, x, re):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)
            re: (batchsize, x_grid, y_grid, t_grid, 1)

        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)
        '''
        re = re.float() / self.max_re
        if self.use_fourier_layer:
            fourier_re = self.fourier_layer1(re.unsqueeze(-1))
        size_z = x.shape[-2]
        if max(self.pad_ratio) > 0:
            num_pad = [round(size_z * i) for i in self.pad_ratio]
        else:
            num_pad = [0., 0.]
        batchsize = x.shape[0]
        
        x = self.fc0(x)
        if self.use_fourier_layer:
            x = self.multiplicative_net1(x, fourier_re)
        else:
            x = self.multiplicative_net1(x, re)
        x = x.permute(0, 4, 1, 2, 3)
        x = add_padding(x, num_pad=num_pad)
        
        res_plane = self.pred_net(x, num_pad, re, self.multiplicative_net2)
        return res_plane
