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


def spectrum3(u):
    T, S1, S2, S3 = u.shape
    u = torch.fft.fftn(u,dim=3)
    s1, s2, s3 = S1, S2, S3  # Renaming for clarity
    k_max1, k_max2, k_max3 = s1 // 2, s2 // 2, s3 // 2
    wavenumers1 = torch.cat((torch.arange(start=0, end=k_max1, step=1), \
                             torch.arange(start=-k_max1, end=0, step=1)), 0)
    wavenumers2 = torch.cat((torch.arange(start=0, end=k_max2, step=1), \
                             torch.arange(start=-k_max2, end=0, step=1)), 0)
    wavenumers3 = torch.cat((torch.arange(start=0, end=k_max3, step=1), \
                             torch.arange(start=-k_max3, end=0, step=1)), 0)
    k_x, k_y, k_z = torch.meshgrid(wavenumers1, wavenumers2, wavenumers3)
    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y) + torch.abs(k_z)
    sum_k = sum_k.numpy()
    
    # Remove symmetric components from wavenumbers
    index = -1.0 * np.ones((s1, s2, s3))
    index[0:k_max1 + 1, 0:k_max2 + 1, 0:k_max3 + 1] = sum_k[0:k_max1 + 1, 0:k_max2 + 1, 0:k_max3 + 1]

    spectrum = np.zeros((T, s1))
    for j in range(1, s1 + 1):
        ind = np.where(index == j)
        spectrum[:, j - 1] = (u[:, ind[0], ind[1], ind[2]].sum(axis=1)).abs() ** 2
    spectrum = spectrum.mean(axis=0)
    return spectrum[::-1]


def vis_spec(data_list, labels, output_file='output_plot', figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)  # Adjust figsize as needed for your plot
    linewidth = 0.5  # You can adjust the line width as needed
    # ax.set_yscale('log')

    for data, label in zip(data_list, labels):
        proj_data = data[:, 2, :, :, :]  # pressure
        # proj_data = proj_data[3] # select v velocity
        # proj_data = proj_data[0] # select pressure
        data_spec = spectrum3(torch.tensor(proj_data))
        ax.plot(data_spec, label=label, linewidth=linewidth)

    # ax.axvline(x=32, color='grey', linestyle='--', linewidth=linewidth)
    ax.set_xlim(1, 20)
    # ax.set_xlim(1, 80)
    ax.set_ylim(0, 40)
    # ax.set_ylim(10000, 1000000000)
    ax.legend(fontsize='large')
    plt.title('Spectrum Visualization')
    plt.xlabel('Wavenumber')
    plt.ylabel('Energy')
    plt.savefig(output_file + '.jpg', dpi=300)  # Adjust dpi (dots per inch) for higher resolution if needed

root_dir = './data/pino'
rey_names = ['100', '200', '250', '300', '350', '400']
data_list = []
labels = []
meta_re = []
data_num = 20
for rey_name in rey_names:
    npy_name = f'NS_fine_Re{rey_name}_T128_part0'
    npy_path = os.path.join(root_dir, npy_name + '.npy')
    print("Loading ...", npy_path)
    start_t = time.time()
    data = np.load(npy_path)
    meta_re.append(np.array([int(rey_name) for _ in range(data_num)]))
    data_list.append(data[:data_num])
    print("Consuming time:", time.time() - start_t)

meta_re = np.concatenate(meta_re)
all_data = np.concatenate(data_list, axis=0)
np.savez('data/pino/multi_reynolds_NS_fine_T128', data1=all_data, data2=meta_re)

loaded_data = np.load('data/pino/multi_reynolds_NS_fine_T128.npz')

vis_spec(data_list, labels, 'output_plot')

import pdb; pdb.set_trace()
