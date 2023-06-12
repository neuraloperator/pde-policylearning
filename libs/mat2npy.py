"""
@author: Zongyi Li
This file is the Fourier Neural Operator for the 2D Navier—Stokes problem discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

################################################################
# Imports
################################################################
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
from torch.optim import Adam

torch.manual_seed(0)
np.random.seed(0)

################################################################
# configs
################################################################
fill_width = 6
root_dir = './data'
# mat_name = 'planes_channel180_minchan'
mat_name = 'planes-001'
minchan = 'minchan' in mat_name
mat_path = os.path.join(root_dir, mat_name + '.mat')
save_dir = os.path.join(root_dir, mat_name)
os.makedirs(save_dir, exist_ok=True)
reader = MatReader(mat_path)
print("Loading mat data ...")
meta_data = {}
# np.save(os.path.join(save_dir, f'metadata.npy'), meta_data)

field_name = 'P_plane' if not minchan else 'P_planes'
meta_data[field_name] = {}
field_data = reader.read_field(field_name).permute(2,0,1)
filed_data_mean = torch.mean(field_data, 0)
field_data_std = torch.std(field_data, 0)
field_data_max = torch.max(field_data)
field_data_min = torch.min(field_data)
meta_data[field_name]['mean'] = np.array(filed_data_mean)
meta_data[field_name]['std'] = np.array(field_data_std)
meta_data[field_name]['max'] = np.array(field_data_max)
meta_data[field_name]['min'] = np.array(field_data_min)

for idx, one_data in enumerate(field_data):
    print(f"Handling {field_name} data idx {idx} ...")
    one_data = np.array(one_data)
    idx_str = str(idx).zfill(fill_width)
    np.save(os.path.join(save_dir, f'{field_name}_{idx_str}.npy'), one_data)

field_name = 'V_plane' if not minchan else 'V_planes'
meta_data[field_name] = {}
field_data = reader.read_field(field_name).permute(2,0,1)
filed_data_mean = torch.mean(field_data, 0)
field_data_std = torch.std(field_data, 0)
field_data_max = torch.max(field_data)
field_data_min = torch.min(field_data)
meta_data[field_name]['mean'] = np.array(filed_data_mean)
meta_data[field_name]['std'] = np.array(field_data_std)
meta_data[field_name]['max'] = np.array(field_data_max)
meta_data[field_name]['min'] = np.array(field_data_min)

for idx, one_data in enumerate(field_data):
    print(f"Handling {field_name} data idx {idx} ...")
    one_data = np.array(one_data)
    idx_str = str(idx).zfill(fill_width)
    np.save(os.path.join(save_dir, f'{field_name}_{idx_str}.npy'), one_data)

np.save(os.path.join(save_dir, f'metadata.npy'), meta_data)
