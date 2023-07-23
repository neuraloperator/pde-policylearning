

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy.random as random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wasserstein_distance

font = {'size'   : 28}
matplotlib.rc('font', **font)

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities4 import *

torch.manual_seed(0)
np.random.seed(0)

T = 500
s = 256
S = s

Re = 5000
index = 1
T = 100

HOME_PATH = '../'

############################################################################
# RE500
# dataloader = MatReader(HOME_PATH+'pred/mno5000.mat')

# pretrain = torch.load(HOME_PATH+'pred/re500-1_8s-800-pino-140k-prediction.pt')
# pretrain_truth = pretrain['truth'].squeeze().permute(2,0,1)
# pretrain_pred = pretrain['pred'].squeeze().permute(2,0,1)
#
# finetune = torch.load(HOME_PATH+'pred/re500-1_8s-800-pino-1k-finetune-prediction.pt')
# finetune_truth = finetune['truth'].squeeze().permute(2,0,1)
# finetune_pred = finetune['pred'].squeeze().permute(2,0,1)
#
# fno = torch.load(HOME_PATH+'pred/re500-1_8s-800-fno-50k-prediction.pt')
# fno_truth = fno['truth'].squeeze().permute(2,0,1)
# fno_pred = fno['pred'].squeeze().permute(2,0,1)
#
# fno = torch.load(HOME_PATH+'pred/prediction_unet.pt')
# unet_truth = fno['truth'].squeeze().permute(2,0,1)
# unet_pred = fno['pred'].squeeze().permute(2,0,1)

finetune = torch.load(HOME_PATH+'pred/pino-Re500-1_8s-eval.pt')
finetune_truth = finetune['truth'].squeeze().permute(0,3,1,2)
finetune_pred = finetune['pred'].squeeze().permute(0,3,1,2)

fno = torch.load(HOME_PATH+'pred/fno-Re500-1_8-eval.pt')
fno_truth = fno['truth'].squeeze().permute(0,3,1,2)
fno_pred = fno['pred'].squeeze().permute(0,3,1,2)

fno = torch.load(HOME_PATH+'pred/prediction_50unet.pt')
unet_truth = fno['truth'].squeeze().permute(0,3,1,2)
unet_pred = fno['pred'].squeeze().permute(0,3,1,2)


shape = fno_pred.shape
print(fno_pred.shape, unet_pred.shape)
unet_pred_low = unet_pred.unsqueeze(1)

batchsize, size_x, size_y, size_z = 1, shape[0], shape[1], shape[2]
gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float)
gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float)
gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
gridz = torch.tensor(np.linspace(-1, 1, size_z), dtype=torch.float)
gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
grid = torch.cat((gridx, gridy, gridz), dim=-1)

# fno_pred_interp = F.grid_sample(fno_pred_low, grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze()
unet_pred_interp = F.interpolate(unet_pred_low, shape[1:], mode='trilinear').squeeze()
# fno_pred_interp = fno_pred_low.squeeze()


# plt.imshow(fno_truth[-1].numpy())
# plt.savefig('truth.png')
# plt.imshow(fno_pred[-1].numpy())
# plt.savefig('fno.png')
fno_error = np.abs(fno_pred[-1].numpy() - fno_truth[-1].numpy())
# plt.imshow(fno_error, vmax=5)
# plt.savefig('fno_error.png')
# plt.imshow(unet_pred_low[0,0,-1].numpy())
# plt.savefig('unet_low.png')
# plt.imshow(unet_pred_interp[-1].numpy())
# plt.savefig('unet_interp.png')
interp_error = np.abs(unet_pred_interp[-1].numpy() - fno_truth[-1].numpy())
# plt.imshow(interp_error, vmax=5)
# plt.savefig('unet_interp_error.png')

print(np.max(fno_error), np.max(interp_error))
# ##############################################################
### FFT plot
##############################################################

#
def spectrum2(u):
    T = u.shape[0]
    u = u.reshape(T, s, s)
    # u = torch.rfft(u, 2, normalized=False, onesided=False)
    u = torch.fft.fft2(u)
    # ur = u[..., 0]
    # uc = u[..., 1]


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



frame = 64
# pred_sp = spectrum2(pretrain_pred[0:frame+1])
# truth_sp = spectrum2(pretrain_truth[0:frame+1])
# finetune_sp = spectrum2(finetune_pred[0:frame+1])
# fno_sp = spectrum2(fno_pred[0:frame+1])
# unet_interp_sp = spectrum2(unet_pred_interp[0:frame+1])

# pred_sp = spectrum2(pretrain_pred.reshape(50*65, 256,256))
truth_sp = spectrum2(fno_truth.reshape(50*65, 256,256))
finetune_sp = spectrum2(finetune_pred.reshape(50*65, 256,256))
fno_sp = spectrum2(fno_pred.reshape(50*65, 256,256))
unet_interp_sp = spectrum2(unet_pred_interp.reshape(50*65, 256,256))

np.save('truth_sp.npy', truth_sp)
np.save('pino_finetune_sp.npy', finetune_sp)
np.save('fno_sp.npy', fno_sp)
np.save('unet_interp_sp.npy', unet_interp_sp)

# print(pred_sp.shape)
fig, ax = plt.subplots(figsize=(10,10))

linewidth = 3
ax.set_yscale('log')
# ax.set_xscale('log')

length = 128
k = np.arange(length) * 1.0
k3 = k**-3 * 100000000000
k5 = k**-(5/3) * 5000000000
# ax.plot(pred_sp, 'r',  label="pino", linewidth=linewidth)
ax.plot(unet_interp_sp, 'r',  label="NN+Interpolation", linewidth=linewidth)
ax.plot(fno_sp, 'b',  label="FNO", linewidth=linewidth)
ax.plot(finetune_sp, 'g',  label="PINO", linewidth=linewidth)
ax.plot(truth_sp, 'k', linestyle=":", label="Ground Truth", linewidth=4)
ax.axvline(x=32, color='grey', linestyle='--', linewidth=linewidth)
# ax.plot(k, k5, 'k--',  label="k^-5/3 scaling", linewidth=linewidth)

# ax.set_xlim(1,length)
ax.set_xlim(1,80)
# ax.set_ylim(1,10000000000)
ax.set_ylim(10000,10000000000)
# ax.set_yticks([0.05,0.10,0.15])
#
plt.legend(prop={'size': 20})
# plt.title('averaged over t=[0,'+str(frame)+']' )
plt.title('spectrum of Kolmogorov Flows' )

plt.xlabel('wavenumber')
plt.ylabel('energy')

leg = plt.legend(loc='best')
leg.get_frame().set_alpha(0.5)
# plt.show()
# plt.savefig('re5000-sp-truth-t'+str(frame)+'.png')
plt.savefig('ifno_spectrum.png')
