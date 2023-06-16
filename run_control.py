import wandb
wandb.login()
import numpy as np
from libs.utilities3 import *
from libs.control_env import *
from libs.unet_models import *
from libs.fno_models import *
from libs.pde_data_loader import *
from libs.visualization import *
# from libs.rk_algorithm import *
from tqdm import tqdm
import os
import imageio


project_name = 'control_v1'
load_model_name = 'planes_channel180_minchan_2-system-UNet-ori-loss.pth'
display_variables = ['policy_name', 'reward_type', 'noise_scale', 'timestep']
DATA_FOLDER = './data/planes_channel180_minchan'
if 'minchan' in DATA_FOLDER:
    path_name = 'planes_channel180_minchan'
else:
    path_name = 'planes'

model_name = 'UNet'
use_v_plane = False
modes = 12
width = 32
debug = True
timestep = 1000
noise_scale = 1.0
assert model_name in ['UNet', 'FNO2dObserverOld', 'FNO2dObserver'], "Model not supported!"
use_spectral_conv = False

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
# policy_name = 'bc'
policy_name = 'rand'
assert policy_name in ['rand', 'bc'], 'Not supported policy.'
reward_type = 'mse'
assert reward_type in ['mse', 'div'], "Not supported reward type."
vis_frame = 60
vis_interval = max(timestep // vis_frame, 1)
output_dir = './outputs'


'''
Policy settings.
'''
rand_scale = 500  # match the random scale to bc
if policy_name == 'bc':
    print("Loading model.")
    model = torch.load(os.path.join(output_dir, load_model_name)).cuda()
    print("Model loaded!")
elif policy_name == 'rand':
    display_variables.append('rand_scale')


config_dict = {
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
    "modes": modes,
    "width": width,
    "r": downsample_rate,
    "use_v_plane": use_v_plane,
    "policy_name": policy_name,
    "rand_scale": rand_scale,
    "reward_type": reward_type,
    'noise_scale': noise_scale,
    "timestep": timestep}

exp_name = ""
for one_v in display_variables:
    exp_name += one_v + "_"
    exp_name += str(config_dict[one_v])
    exp_name += "; "

if not debug:
    wandb.init(
        project=project_name + "_" + path_name,
        name=exp_name,
        config=config_dict)


################################################################
# create env
################################################################
print("Initialization env...")
control_env = NSControl(timestep=timestep, noise_scale=noise_scale)
print("Load model ...")

################################################################
# create model
################################################################
if model_name == 'FNO2dObserverOld':
    model = FNO2dObserverOld(modes, modes, width, use_v_plane=use_v_plane).cuda()
elif model_name == 'FNO2dObserver':
    model = FNO2dObserver(modes, modes, width, use_v_plane=use_v_plane).cuda()
else:
    model = UNet(use_spectral_conv=use_spectral_conv).cuda()

################################################################
# create dataset
################################################################
demo_dataset = PDEDataset(DATA_FOLDER, [1, 2, 3, 4, 5], downsample_rate, x_range, y_range, use_patch=False)

################################################################
# main control loop
################################################################
pressure_v, opV2_v = [], []
for i in tqdm(range(timestep)):
    pressure = control_env.compute_pressure()
    # pressure: [32, 32], opV2: [32, 32]
    # opV2 = control_env.rand_control(pressure)
    pressure = torch.tensor(pressure).cuda()
    pressure = demo_dataset.p_norm.encode(pressure)
    pressure = pressure.reshape(-1, x_range, y_range, 1).float()
    if policy_name == 'rand':
        opV2 = control_env.rand_control(pressure)
        opV2 *= rand_scale
    else:
        opV2 = model(pressure, None).reshape(-1, x_range, y_range)
        opV2 = demo_dataset.p_norm.decode(opV2.cpu())
        opV2 = opV2.detach().numpy()
    pressure, reward, done, info = control_env.step(opV2)
    if not debug:
        wandb.log(info)
    if i % vis_interval == 0:
        cur_opV2_image = matrix2image(opV2, extend_value=0.2)
        cur_pressure_image = matrix2image(pressure, extend_value=0.2)
        opV2_v.append(cur_opV2_image)
        pressure_v.append(cur_pressure_image)
    print(f"timestep: {i}, scores: {info}")

print("Saving results to video ...")
save_images_to_video(opV2_v, os.path.join(output_dir, exp_name + '; v_plane.mp4'), fps=15)
save_images_to_video(pressure_v, os.path.join(output_dir, exp_name + '; pressure.mp4'), fps=15)
print("Program finished!")
