"""
@author: Zongyi Li
This file is the Fourier Neural Operator for the 2D Navier—Stokes problem discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
################################################################
# Connecting to WandB 
################################################################
import wandb
wandb.login()
# WANDB_API_KEY = '78544c6ed5f52873b1588acd09ead571942d7dfd'

################################################################
# Imports
################################################################
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

from functools import reduce
from functools import partial

from timeit import default_timer
from libs.utilities3 import *
from libs.unet_models import *
from libs.fno_models import *
from libs.pde_data_loader import *
from torch.optim import Adam

torch.manual_seed(0)
np.random.seed(0)

################################################################
# configs
################################################################
# DATA_FOLDER = './data/planes-001'
DATA_FOLDER = './data/planes_channel180_minchan'
project_name = 'fno_vs_unet'
exp_name = '1-system-FNO-original-loss'

if 'minchan' in DATA_FOLDER:
    path_name = 'planes_channel180_minchan'
else:
    path_name = 'planes'
    
debug = False
batch_size = 20
learning_rate = 1e-3

epochs = 500
step_size = 100
gamma = 0.5

modes = 12
width = 32

if path_name == 'planes':
    downsample_rate = 1  # 8 previously
    x_range = 768//downsample_rate
    y_range = 288//downsample_rate
    ntrain = 3000
    ntest = 1000
elif 'planes_channel180_minchan' in path_name: 
    downsample_rate = 1
    x_range = 32//downsample_rate
    y_range = 32//downsample_rate
    ntrain = 7500
    ntest = 2501
else:
    raise RuntimeError("Type not supported!")
use_v_plane = False
use_patch = False

################################################################
# load data and data normalization
################################################################
idx = torch.randperm(ntrain + ntest)
training_idx = idx[:ntrain]
testing_idx = idx[-ntest:]
train_dataset = PDEDataset(DATA_FOLDER, training_idx, downsample_rate, x_range, y_range, use_patch=use_patch)
test_dataset = PDEDataset(DATA_FOLDER, testing_idx, downsample_rate, x_range, y_range, use_patch=use_patch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=not debug, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
n_steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)


################################################################
# create model
################################################################
model_name = 'FNO2dObserver'
use_spectral_conv = False
if model_name == 'FNO2dObserverOld':
    model = FNO2dObserverOld(modes, modes, width, use_v_plane=use_v_plane).cuda()
elif model_name == 'FNO2dObserver':
    model = FNO2dObserver(modes, modes, width, use_v_plane=use_v_plane).cuda()
else:
    model = UNet(use_spectral_conv=use_spectral_conv).cuda()

################################################################
# training and evaluation
################################################################
print("param number:", count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

output_path = './outputs/'
output_path += path_name
output_path += '_observer.mat'

myloss = LpLoss(size_average=False)

if not debug:
    wandb.init(
        project=project_name + "_" + path_name,
        name=exp_name,
        config={
            "task info": "p-plane-to-v",
            "model_name": model_name,
            "file_name": path_name,
            "has_prev_press": True,
            "patches": False,
            "permute": True,
            "use_spectral_conv": use_spectral_conv,
            "DATA_FOLDER": DATA_FOLDER,
            "ntrain": ntrain,
            "ntest": ntest,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "step_size": step_size,
            "gamma": gamma,
            "modes": modes,
            "width": width,
            "r": downsample_rate,
            "use_v_plane": use_v_plane,
            "use_patch": use_patch
            })

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for step, (p_plane, v_plane) in enumerate(train_loader):
        p_plane, v_plane = p_plane.cuda(), v_plane.cuda()
        p_plane = p_plane.reshape(-1, x_range, y_range, 1)
        v_plane = v_plane.reshape(-1, x_range, y_range, 1)
        optimizer.zero_grad()
        out = model(p_plane, v_plane).reshape(-1, x_range, y_range)
        out_decoded = train_dataset.v_norm.cuda_decode(out)
        v_plane = v_plane.squeeze()
        v_plane_decoded = train_dataset.v_norm.cuda_decode(v_plane)
        # loss = myloss(out.view(batch_size, -1), v_plane.view(batch_size, -1))
        loss = myloss(out_decoded.view(batch_size, -1), v_plane_decoded.view(batch_size, -1))
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()
        metrics = {"train/train_loss": loss.item(), 
                   "train/epoch": (step + 1 + (n_steps_per_epoch * ep)) / n_steps_per_epoch}
        if step + 1 < n_steps_per_epoch and not debug:
            # Log train metrics to wandb 
            wandb.log(metrics)

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for p_plane, v_plane in test_loader:
            p_plane, v_plane = p_plane.cuda(), v_plane.cuda()
            p_plane = p_plane.reshape(-1, x_range, y_range, 1)
            v_plane = v_plane.reshape(-1, x_range, y_range, 1)
            out = model(p_plane, v_plane).reshape(-1, x_range, y_range)
            out_decoded = train_dataset.v_norm.cuda_decode(out)
            v_plane = v_plane.squeeze()
            p_plane_decoded = train_dataset.p_norm.cuda_decode(p_plane)
            v_plane_decoded = train_dataset.v_norm.cuda_decode(v_plane)
            # test_loss = myloss(out.view(batch_size, -1), v_plane.view(batch_size, -1)).item()
            test_loss = myloss(out_decoded.view(batch_size, -1), v_plane_decoded.view(batch_size, -1)).item()
            test_l2 += test_loss
            test_metrics = {"test/test_loss": test_loss}
            if not debug:
                wandb.log(test_metrics)

    train_l2/= ntrain
    test_l2 /= ntest
    avg_metrics = {"train/avg_train_loss": train_l2,
                   "test/avg_test_loss": test_l2}
    if not debug:
        wandb.log(avg_metrics)

    t2 = default_timer()
    print(f"epoch: {ep}, time passed: {t2-t1}, train loss: {train_l2}, test loss: {test_l2}")

    if ep == epochs - 1 or ep % 50 == 0:
        dat = {'x': p_plane_decoded.cpu().numpy(), 'pred': out_decoded.cpu().numpy(), 'y': v_plane_decoded.cpu().numpy(),}
        # scipy.io.savemat(output_path, mdict=dat)
        # Plots
        for index in [0, 5, 10, 19]:
            vmin = dat['y'][index, :, :].min()
            vmax = dat['y'][index, :, :].max()
            fig, axes = plt.subplots(nrows=1, ncols=4)
            plt.subplot(1, 3, 1)
            im1 = plt.imshow(dat['x'][index, :, :, 0], cmap='jet', aspect='auto')
            plt.title('Input')
            plt.subplot(1, 3, 2)
            im2 = plt.imshow(dat['y'][index, :, :], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
            plt.title('True Output')
            plt.subplot(1, 3, 3)
            im3 = plt.imshow(dat['pred'][index, :, :], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
            plt.title('Prediction')
            cbar_ax = fig.add_axes([.92, 0.15, 0.04, 0.7])
            fig.colorbar(im3, cax=cbar_ax)
            if not debug:
                wandb.log({f"data_id_{index}": plt})

# torch.save(model, "/central/groups/tensorlab/khassibi/fourier_neural_operator/outputs/planes")

################################################################
# making the plots
################################################################
# dat = scipy.io.loadmat(output_path)

# Plots
for index in [0, 5, 10, 19]:
    vmin = dat['y'][index, :, :].min()
    vmax = dat['y'][index, :, :].max()
    fig, axes = plt.subplots(nrows=1, ncols=4)
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(dat['x'][index, :, :, 0], cmap='jet', aspect='auto')
    plt.title('Input')
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(dat['y'][index, :, :], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    plt.title('True Output')
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(dat['pred'][index, :, :], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    plt.title('Prediction')
    cbar_ax = fig.add_axes([.92, 0.15, 0.04, 0.7])
    fig.colorbar(im3, cax=cbar_ax)
    if not debug:
        wandb.log({f"chart_{index}": plt})
if not debug:
    wandb.finish()