import wandb
import numpy as np
import torch
import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from libs.utilities3 import *
from libs.envs.control_env import NSControlEnvMatlab
from libs.envs.ns_control_2d import NSControlEnv2D
from libs.unet_models import *
from libs.models.fno_models import *
from libs.pde_data_loader import *
from libs.visualization import *
from libs.arguments import *
# from libs.rk_algorithm import *
from tqdm import tqdm
import os
import torch.optim as optim
from einops import rearrange
from pympler import muppy, summary
from pympler import tracker


def run_control(args, observer_model=None, policy_model=None, train_dataset=None, wandb_exist=False):
    '''
    Policy settings.
    '''
    if observer_model is not None:
        device = next(observer_model.parameters()).device
    else:
        device = None
    args.vis_interval = max(args.control_timestep // args.vis_frame, 1) if args.vis_frame > 0 else -1
    if args.policy_name == 'fno' or args.policy_name == 'rno':
        if observer_model is None:
            print("Loading model.")
            observer_model = torch.load(os.path.join(args.output_dir, args.load_model_name)).cuda()
            print("Model loaded!")
    elif args.policy_name == 'rand':
        args.display_variables.append('rand_scale')
    
    if args.policy_name != 'gt' and args.policy_name != 'unmanipulated':
        args.collect_data = False
    
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
        "env_name": args.env_name,
        "rand_scale": args.rand_scale,
        "reward_type": args.reward_type,
        'noise_scale': args.noise_scale,
        "control_timestep": args.control_timestep,
        "model_timestep": args.model_timestep,
        "exp_name": args.exp_name,
        "init_cond_path": args.init_cond_path, 
        "detect_plane": args.detect_plane,
        "test_plane": args.test_plane,
        "bc_type": args.bc_type, 
        "Re": args.Re}

    exp_name = ""
    for one_v in args.display_variables:
        exp_name += one_v + "_"
        exp_name += str(config_dict[one_v])
        exp_name += "; "

    if not args.close_wandb and not wandb_exist:
        print("Init wandb!")
        wandb.init(
            project=args.project_name + "_" + args.path_name,
            name=exp_name,
            config=config_dict,
            )
        wandb.config.update({"wandb.Table": False})  # not show the tables
        # define metrics
        wandb.define_metric("control_timestep")
        wandb.define_metric("drag_reduction/*", step_metric="control_timestep")
        wandb.define_metric("drag_reduction_relative/*", step_metric="control_timestep")
        
    '''
    Create env.
    '''
    
    print("Initialization env...")
    if args.env_name == 'NSControlEnv2D':
        env_class = NSControlEnv2D
        control_env = env_class(args, detect_plane=args.detect_plane, bc_type=args.bc_type)
    elif args.env_name == 'NSControlEnvMatlab':
        env_class = NSControlEnvMatlab
        control_env = env_class(args)
    else:
        raise RuntimeError("Not supported environment!")
    print("Environment is initialized!")
    
    '''
    Setup data.
    '''
    
    if args.collect_data:
        collect_data_folder = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(collect_data_folder, exist_ok=True)
    else:
        collect_data_folder = None
    
    if train_dataset is not None:
        demo_dataset = train_dataset
    elif args.policy_name not in ["gt", "rand", "unmanipulated"]:
        demo_dataset = PDEDataset(args, args.DATA_FOLDER, [1, 2, 3, 4, 5], args.downsample_rate, args.x_range, 
                                args.y_range, use_patch=False)
    else:
        demo_dataset = None

    '''
    Main control loop.
    '''
    
    pressure_v, opV2_v, top_view_v, front_view_v, side_view_v, all_p_boundary, all_v_boundary = [], [], [], [], [], [], []
    metadata = {}
    all_u_field, all_v_field, all_w_field = [], [], []
    for i in (pbar := tqdm(range(args.control_timestep + 1))):
        # pressure: [32, 32], opV2: [32, 32]
        if args.policy_name in ['fno', 'rno']:  # neural policies
            p1, p2 = control_env.get_boundary_pressures()
            side_pressure = torch.tensor(p2)
            side_pressure = demo_dataset.p_norm.encode(side_pressure).cuda()
            side_pressure = side_pressure.reshape(-1, args.x_range, args.y_range, 1).float()
        if args.policy_name == 'rand':
            opV2 = control_env.rand_control()
            opV2 *= args.rand_scale
        elif args.policy_name == 'fno':
            opV2 = observer_model(side_pressure, None).reshape(-1, args.x_range, args.y_range)
            opV2 = demo_dataset.p_norm.decode(opV2.cpu())
            opV2 = opV2.detach().numpy().squeeze()
        elif args.policy_name == 'rno':
            side_pressure = side_pressure.reshape(-1, 1, args.x_range, args.y_range, 1)
            opV2 = observer_model(side_pressure, None).squeeze()
            opV2 = demo_dataset.p_norm.decode(opV2.cpu())
            opV2 = opV2.detach().numpy().squeeze()
            opV1 = opV2 * 0
        elif args.policy_name == 'gt':
            p1, p2 = control_env.get_boundary_pressures()
            opV1, opV2 = control_env.gt_control()
        elif args.policy_name == 'unmanipulated':
            opV1, opV2 = control_env.gt_control()
            opV1 *= 0
            opV2 *= 0
        elif args.policy_name == 'optimal-policy-observer':
            p1, p2 = control_env.get_boundary_pressures()
            opV1, opV2 = control_env.gt_control()   # one-side control
            opV2 = torch.tensor(opV2).float().to(device).unsqueeze(0).unsqueeze(0)
            opV2 = torch.einsum('btxy -> bxyt', opV2).unsqueeze(-1)  # expand feature dim
            re = torch.tensor(control_env.Re).to(device).unsqueeze(0).float()
            p2 = torch.tensor(p2).to(device).float()
            p2 = p2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            optimizer = optim.Adam(policy_model.parameters(), lr=1e-4)
            res_opV2 = policy_model(p2, re)
            pred_v_field = observer_model(opV2 + res_opV2, re)
            pred_v_field = torch.einsum('bxztk -> btxz', pred_v_field)
            reg_weight = 0.1
            initial_loss = torch.norm(pred_v_field) + reg_weight * torch.norm(opV2 + res_opV2)  # minimize this.
            print("Initial Loss:", initial_loss.item())
            num_epochs = 3
            for epoch in range(num_epochs):
                optimizer.zero_grad()  # Zero the gradients
                res_opV2 = policy_model(p2, re)
                pred_v_field = observer_model(opV2 + res_opV2, re)  # Forward pass
                loss = torch.norm(pred_v_field) + reg_weight * torch.norm(opV2 + res_opV2) # minimize this.
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the parameters
            opV2 += res_opV2
            opV2 = opV2.detach().cpu().numpy().squeeze()            
        elif args.policy_name == 'optimal-observer':
            opV1, opV2 = control_env.gt_control()   # one-side control
            opV2 = torch.tensor(opV2).float().to(device).unsqueeze(0).unsqueeze(0)
            opV2 = torch.einsum('btxy -> bxyt', opV2).unsqueeze(-1)  # expand feature dim
            re = torch.tensor(control_env.Re).to(device).unsqueeze(0)
            
            # Instantiate the optimizer
            optimizer = optim.Adam([opV2], lr=0.001)  # You can adjust the learning rate
            opV2.requires_grad = True
            # Normalize and forward model
            norm_opV2 = train_dataset.bound_v_norm.cuda_encode(opV2.squeeze()).float()
            norm_opV2 = norm_opV2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            norm_pred_v_field = observer_model(norm_opV2, re)
            pred_field = []
            for plane_index in range(len(train_dataset.plane_indexs)):
                pred_one_plane = norm_pred_v_field[:, plane_index, :, :]
                pred_one_plane = train_dataset.v_field_norm.cuda_decode(pred_one_plane)
                pred_field.append(pred_one_plane)
            pred_field = torch.stack(pred_field, dim=2)
            reg_weight = 0.1
            initial_loss = torch.norm(pred_field) + reg_weight * torch.norm(opV2) # minimize this.
            # print("Initial Loss:", initial_loss.item())
            num_epochs = 10
            for epoch in range(num_epochs):
                optimizer.zero_grad()  # Zero the gradients
                norm_opV2 = train_dataset.bound_v_norm.cuda_encode(opV2.squeeze()).float()
                norm_opV2 = norm_opV2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                norm_pred_v_field = observer_model(norm_opV2, re)  # Forward pass
                pred_field = []
                for plane_index in range(len(train_dataset.plane_indexs)):
                    pred_one_plane = norm_pred_v_field[:, plane_index, :, :]
                    pred_one_plane = train_dataset.v_field_norm.cuda_decode(pred_one_plane)
                    pred_field.append(pred_one_plane)
                pred_field = torch.stack(pred_field, dim=2)
                loss = torch.norm(pred_field) + reg_weight * torch.norm(opV2) # minimize this.
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the parameters
            opV2 = opV2.detach().cpu().numpy().squeeze()
        else:
            raise RuntimeError("Not supported policy name.")
        if i == 0 and args.policy_name == 'unmanipulated':   # remove jitter at beginning
            # print("Initializing unmanipulated ... ")
            # for _ in range(100):
            #     control_env.step(opV1, opV2, print_info=False)
            # print("Initialization done ... ")
            control_env.reset_init()

        '''
        Collect data when needed
        '''
        
        mean_num = 100
        if args.collect_data and i > args.collect_start:
            idx_str = str(i).zfill(6)
            # (0) save Reynold numbers
            metadata['re'] = args.Re
            # (1) save boundary pressure
            p1, p2 = p1.astype(np.float64), p2.astype(np.float64)
            opV1, opV2 = opV1.astype(np.float64), opV2.astype(np.float64)
            field_name = 'P_planes'
            np.save(os.path.join(collect_data_folder, f'{field_name}_{idx_str}.npy'), np.array(p2))
            if i < mean_num:
                all_p_boundary.append(p2)
                metadata[field_name] = {}
                metadata[field_name]['mean'] = np.array(all_p_boundary).mean(0)
                metadata[field_name]['std'] = np.array(all_p_boundary).std(0)
            # (2) save boundary velocity
            field_name = 'V_planes'
            np.save(os.path.join(collect_data_folder, f'{field_name}_{idx_str}.npy'), np.array(opV2))
            if i < mean_num:
                all_v_boundary.append(opV2)
                metadata[field_name] = {}
                metadata[field_name]['mean'] = np.array(all_v_boundary).mean(0)
                metadata[field_name]['std'] = np.array(all_v_boundary).std(0)
            # (3) save u field info
            field_name = 'U_field'
            np.save(os.path.join(collect_data_folder, f'{field_name}_{idx_str}.npy'), np.array(control_env.U))
            if i < mean_num:
                all_u_field.append(np.array(control_env.U))
                metadata[field_name] = {}
                metadata[field_name]['mean'] = np.array(all_u_field).mean(0)
                metadata[field_name]['std'] = np.array(all_u_field).std(0)
            # (4) save v field info
            field_name = 'V_field'
            np.save(os.path.join(collect_data_folder, f'{field_name}_{idx_str}.npy'), np.array(control_env.V))
            if i < mean_num:
                all_v_field.append(np.array(control_env.V))
                metadata[field_name] = {}
                metadata[field_name]['mean'] = np.array(all_v_field).mean(0)
                metadata[field_name]['std'] = np.array(all_v_field).std(0)
            # (5) save w field info
            field_name = 'W_field'
            np.save(os.path.join(collect_data_folder, f'{field_name}_{idx_str}.npy'), np.array(control_env.W))
            if i < mean_num:
                all_w_field.append(np.array(control_env.W))
                metadata[field_name] = {}
                metadata[field_name]['mean'] = np.array(all_w_field).mean(0)
                metadata[field_name]['std'] = np.array(all_w_field).std(0)
            np.save(os.path.join(collect_data_folder, f'metadata.npy'), metadata)
        if control_env.reward_div() < -10:
            raise RuntimeError("Control is bloded!")
        side_pressure, reward, done, info = control_env.step(opV1, opV2)
        
        if not args.close_wandb and i > 0:  # ignore the first iteration
            info['control_timestep'] = i
            wandb.log(info)
            if i % args.show_spatial_dist_interval == 1 and args.vis_interval != -1:
                control_env.plot_spatial_distribution(i)
        if args.vis_interval != -1 and i % args.vis_interval == 0:
            top_view, front_view, side_view = control_env.vis_state(vis_img=args.vis_sample_img)
            top_view_v.append(top_view)
            front_view_v.append(front_view)
            side_view_v.append(side_view)
            cur_opV2_image = matrix2image(control_env.V[:, -10, :], extend_value=1e-2)
            cur_pressure_image = matrix2image(side_pressure, extend_value=1e-2)
            opV2_v.append(cur_opV2_image)
            pressure_v.append(cur_pressure_image)
        if i % 100 == 0 and args.dump_state:
            control_env.dump_state(save_path=os.path.join('outputs', f'flow_{i}.npy'))
        if i > 0:  # omit the first iter
            print_info = f"dPdx: {info['drag_reduction/3_3_dPdx_reverse_cal']:.7f}; DR: {1 - info['drag_reduction_relative/3_3_dPdx_reverse_cal']:.4f}"
            pbar.set_description(print_info)

    '''
    Save visualization results.
    '''
    
    if args.vis_interval != -1:
        exp_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"Saving results to folder {exp_dir}.")
        save_images_to_video(top_view_v, os.path.join(exp_dir, exp_name + 'top_view.mp4'), fps=15)
        save_images_to_video(front_view_v, os.path.join(exp_dir, exp_name + 'front_view.mp4'), fps=15)
        save_images_to_video(side_view_v, os.path.join(exp_dir, exp_name + 'side_view.mp4'), fps=15)
        save_images_to_video(opV2_v, os.path.join(exp_dir, exp_name + 'v_plane.mp4'), fps=15)
        save_images_to_video(pressure_v, os.path.join(exp_dir, exp_name + 'pressure.mp4'), fps=15)
    print("Program finished!")
    if not args.close_wandb and not wandb_exist:
        wandb.finish()
    
    # Analyzing memory
    print("memory consumption info:")
    summary.print_(summary.summarize(muppy.get_objects()))


if __name__ == '__main__':
    # Setup args
    args = parse_arguments()
    loaded_args = load_arguments_from_yaml(args.control_yaml)
    args = merge_args_with_yaml(args, loaded_args)
    if not args.close_wandb:
        wandb.login()
    run_control(args)
