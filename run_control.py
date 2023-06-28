import wandb
import numpy as np
from libs.utilities3 import *
from libs.control_env import *
from libs.unet_models import *
from libs.models.fno_models import *
from libs.pde_data_loader import *
from libs.visualization import *
from libs.arguments import *
# from libs.rk_algorithm import *
from tqdm import tqdm
import os
import imageio


def main(args):
    '''
    Policy settings.
    '''
    if args.policy_name == 'fno':
        print("Loading model.")
        model = torch.load(os.path.join(args.output_dir, args.load_model_name)).cuda()
        print("Model loaded!")
    elif args.policy_name == 'rand':
        args.display_variables.append('rand_scale')

    config_dict = {
        "task info": "p-plane-to-v",
        "model_name": args.model_name,
        "file_name": args.path_name,
        "has_prev_press": True,
        "patches": False,
        "permute": True,
        "use_spectral_conv": args.use_spectral_conv,
        "DATA_FOLDER": args.DATA_FOLDER,
        "modes": args.modes,
        "width": args.width,
        "r": args.downsample_rate,
        "use_v_plane": args.use_v_plane,
        "policy_name": args.policy_name,
        "rand_scale": args.rand_scale,
        "reward_type": args.reward_type,
        'noise_scale': args.noise_scale,
        "timestep": args.timestep}

    exp_name = ""
    for one_v in args.display_variables:
        exp_name += one_v + "_"
        exp_name += str(config_dict[one_v])
        exp_name += "; "

    if not args.close_wandb:
        wandb.init(
            project=args.project_name + "_" + args.path_name,
            name=exp_name,
            config=config_dict)

    ################################################################
    # create env
    ################################################################
    print("Initialization env...")
    control_env = NSControl(timestep=args.timestep, noise_scale=args.noise_scale)
    print("Load model ...")

    ################################################################
    # create dataset
    ################################################################
    demo_dataset = PDEDataset(args, args.DATA_FOLDER, [1, 2, 3, 4, 5], args.downsample_rate, args.x_range, args.y_range, use_patch=False)

    ################################################################
    # main control loop
    ################################################################
    pressure_v, opV2_v, top_view_v, front_view_v, side_view_v = [], [], [], [], []
    for i in tqdm(range(args.timestep)):
        # pressure: [32, 32], opV2: [32, 32]
        side_pressure = control_env.get_state()
        side_pressure = torch.tensor(side_pressure)
        side_pressure = demo_dataset.p_norm.encode(side_pressure).cuda()
        side_pressure = side_pressure.reshape(-1, args.x_range, args.y_range, 1).float()
        if args.policy_name == 'rand':
            opV2 = control_env.rand_control(side_pressure)
            opV2 *= args.rand_scale
        else:
            opV2 = model(side_pressure, None).reshape(-1, args.x_range, args.y_range)
            opV2 = demo_dataset.p_norm.decode(opV2.cpu())
            opV2 = opV2.detach().numpy().squeeze()
        if control_env.reward_div() < -10:
            import pdb; pdb.set_trace()
        side_pressure, reward, done, info = control_env.step(opV2)
        if not args.close_wandb:
            wandb.log(info)
        if i % args.vis_interval == 0:
            top_view, front_view, side_view = control_env.vis_state(vis_img=args.vis_sample_img)
            top_view_v.append(top_view)
            front_view_v.append(front_view)
            side_view_v.append(side_view)
            cur_opV2_image = matrix2image(opV2, extend_value=0.2)
            cur_pressure_image = matrix2image(side_pressure, extend_value=0.2)
            opV2_v.append(cur_opV2_image)
            pressure_v.append(cur_pressure_image)
        print(f"timestep: {i}, scores: {info}")

    # save visualization results
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Saving results to folder {exp_dir}.")
    save_images_to_video(top_view_v, os.path.join(exp_dir, exp_name + 'top_view.mp4'), fps=15)
    save_images_to_video(front_view_v, os.path.join(exp_dir, exp_name + 'front_view.mp4'), fps=15)
    save_images_to_video(side_view_v, os.path.join(exp_dir, exp_name + 'side_view.mp4'), fps=15)
    save_images_to_video(opV2_v, os.path.join(exp_dir, exp_name + 'v_plane.mp4'), fps=15)
    save_images_to_video(pressure_v, os.path.join(exp_dir, exp_name + 'pressure.mp4'), fps=15)
    print("Program finished!")


if __name__ == '__main__':
    # Setup args
    args = parse_arguments()
    loaded_args = load_arguments_from_yaml(args.control_yaml)
    args = merge_args_with_yaml(args, loaded_args)
    assert args.model_name in ['UNet', 'FNO2dObserverOld', 'FNO2dObserver'], "Model not supported!"
    args.vis_interval = max(args.timestep // args.vis_frame, 1)
    if not args.close_wandb:
        wandb.login()
    # save_arguments_to_yaml(args, )
    main(args)
