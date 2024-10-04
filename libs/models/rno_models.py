import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from libs.models.pino_models import SpectralConv2d
from libs.models.transformer_models import SpectralRegressor
from libs.utilities3 import *
from neuralop.models import RNO2d



class RNO2dObserver(RNO2d):
    def __init__(self, modes1, modes2, width, recurrent_index, layer_num=3, pad_amount=None, pad_dim='1'):
        super().__init__(modes1, modes2, width, recurrent_index, layer_num=layer_num, pad_amount=pad_amount, pad_dim=pad_dim)
        return

