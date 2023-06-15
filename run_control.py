import numpy as np
from libs.utilities3 import *
from libs.control_env import *
from libs.utilities3 import *
from libs.unet_models import *
from libs.fno_models import *
from libs.pde_data_loader import *

# from libs.rk_algorithm import *
import os

print("Initialization env...")
timestep = 100
control_env = NSControl(timestep=timestep)
print("Load model ...")

################################################################
# create model
################################################################
model_name = 'UNet'
use_v_plane = False
modes = 12
width = 32

assert model_name in ['UNet', 'FNO2dObserverOld', 'FNO2dObserver'], "Model not supported!"
use_spectral_conv = False
if model_name == 'FNO2dObserverOld':
    model = FNO2dObserverOld(modes, modes, width, use_v_plane=use_v_plane).cuda()
elif model_name == 'FNO2dObserver':
    model = FNO2dObserver(modes, modes, width, use_v_plane=use_v_plane).cuda()
else:
    model = UNet(use_spectral_conv=use_spectral_conv).cuda()

exp_name = '2-system-UNet-ori-loss'
DATA_FOLDER = './data/planes_channel180_minchan'
if 'minchan' in DATA_FOLDER:
    path_name = 'planes_channel180_minchan'
else:
    path_name = 'planes'

'''
Dataset settings.
'''
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


'''
Policy settings.
'''
policy_name = 'bc'
# policy_name = 'rand'
assert policy_name in ['rand', 'bc'], 'Not supported policy.'
rand_scale = 100  # match the random scale to bc
train_dataset = PDEDataset(DATA_FOLDER, [1, 2, 3, 4, 5], downsample_rate, x_range, y_range, use_patch=False)
print("Loading model.")
model = torch.load(f"./outputs/{path_name}_{exp_name}.pth").cuda()
print("Model loaded!")


# Main time loop
for i in range(timestep):
    pressure = control_env.compute_pressure()
    # pressure: [32, 32], opV2: [32, 32]
    # opV2 = control_env.rand_control(pressure)
    pressure = torch.tensor(pressure).cuda()
    pressure = train_dataset.p_norm.encode(pressure)
    pressure = pressure.reshape(-1, x_range, y_range, 1).float()
    if policy_name == 'rand':
        opV2 = control_env.rand_control(pressure)
        opV2 *= rand_scale
    else:
        opV2 = model(pressure, None).reshape(-1, x_range, y_range)
        opV2 = train_dataset.p_norm.decode(opV2.cpu())
        opV2 = opV2.detach().numpy()
    pressure, reward, done, info = control_env.step(opV2)
    print(reward)
