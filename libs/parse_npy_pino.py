################################################################
# Imports
################################################################
import math
import os
import numpy as np
import torch
import time
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

def spectrum2(u):
    T = u.shape[0]
    # u = torch.rfft(u, 2, normalized=False, onesided=False)
    u = torch.fft.fft2(u)
    # ur = u[..., 0]
    # uc = u[..., 1]
    s = u.shape[-1]
    # 2d wavenumbers following Pytorch fft convention
    k_max = s // 2
    wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                            torch.arange(start=-k_max, end=0, step=1)), 0).repeat(s, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers
    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k.numpy()
    # Remove symmetric components from wavenumbers
    index = -1.0 * np.ones((s, s))
    index[0:k_max + 1, 0:k_max + 1] = sum_k[0:k_max + 1, 0:k_max + 1]
    spectrum = np.zeros((T, s))
    for j in range(1, s + 1):
        ind = np.where(index == j)
        # spectrum[:, j - 1] = np.sqrt((ur[:, ind[0], ind[1]].sum(axis=1)) ** 2
                                     # + (uc[:, ind[0], ind[1]].sum(axis=1)) ** 2)
        spectrum[:, j - 1] =  (u[:, ind[0], ind[1]].sum(axis=1)).abs() ** 2
    spectrum = spectrum.mean(axis=0)
    return spectrum



def vis_spec(data_list, labels, output_file='output_plot', figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)  # Adjust figsize as needed for your plot
    linewidth = 3  # You can adjust the line width as needed
    ax.set_yscale('log')

    for data, label in zip(data_list, labels):
        data_spec = spectrum2(torch.tensor(data.reshape(-1, data.shape[-2], data.shape[-1])))
        ax.plot(data_spec, label=label, linewidth=linewidth)

    # ax.axvline(x=32, color='grey', linestyle='--', linewidth=linewidth)
    ax.set_xlim(1, 80)
    ax.set_ylim(10000, 1000000000)
    ax.legend(fontsize='large')
    plt.title('Spectrum Visualization')
    plt.xlabel('Wavenumber')
    plt.ylabel('Energy')
    plt.savefig(output_file + '.jpg', dpi=300)  # Adjust dpi (dots per inch) for higher resolution if needed

root_dir = './data/pino'
rey_names = ['100', '200', '250', '300', '350', '400', '500']
data_list = []
labels = []

for rey_name in rey_names:
    npy_name = f'NS_fine_Re{rey_name}_T128_part0'
    npy_path = os.path.join(root_dir, npy_name + '.npy')
    print("Loading ...", npy_path)
    start_t = time.time()
    data = np.load(npy_path)
    data_list.append(data)
    labels.append(f"Re={rey_name}")
    print("Consuming time:", time.time() - start_t)

vis_spec(data_list, labels, 'output_plot')

save_dir = os.path.join(root_dir, npy_name)
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
