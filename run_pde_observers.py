"""
@author: Zongyi Li
This file is the Fourier Neural Operator for the 2D Navier—Stokes problem discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import wandb
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from libs.utilities3 import *
from libs.unet_models import *
from libs.models.fno_models import *
from libs.models.rno_models import RNO2dObserver
from libs.models.transformer_models import *
from libs.visualization import *
from libs.pde_data_loader import *
from libs.arguments import *
from libs.metrics import *
from tqdm import tqdm
from torch.optim import Adam


torch.manual_seed(0)
np.random.seed(0)


def main(args, sample_data=False, train_shuffle=True):
    args.using_transformer = 'Transformer' in args.model_name
    assert args.model_name in ['UNet', 'RNO2dObserverOld', 'FNO2dObserverOld', 'FNO2dObserver', 'Transformer2D'],  "Model not supported!"
    if not args.close_wandb:
        wandb.login()
    if args.random_split:
        idx = torch.randperm(args.ntrain + args.ntest)
    else:
        idx = torch.arange(args.ntrain + args.ntest)
    training_idx = idx[:args.ntrain]
    testing_idx = idx[-args.ntest:]
    if args.recurrent_model:
        dataset_fn = SequentialPDEDataset
    else:
        dataset_fn = PDEDataset
    train_dataset = dataset_fn(args, args.DATA_FOLDER, training_idx, args.downsample_rate, args.x_range, args.y_range, use_patch=args.use_patch)
    test_dataset = dataset_fn(args, args.DATA_FOLDER, testing_idx, args.downsample_rate, args.x_range, args.y_range, use_patch=args.use_patch)
    if sample_data:
        p_plane, v_plane = train_dataset[0]
        p_plane, v_plane = p_plane.cuda(), v_plane.cuda()
        v_plane = v_plane.squeeze()
        v_plane_decoded = train_dataset.v_norm.cuda_decode(v_plane)
        np.savetxt('outputs/v_plane_decoded.txt', v_plane_decoded.cpu().numpy())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)

    ################################################################
    # create model
    ################################################################
    if args.model_name == 'FNO2dObserverOld':
        model = FNO2dObserverOld(args.modes, args.modes, args.width, use_v_plane=args.use_v_plane).cuda()
    elif args.model_name == 'FNO2dObserver':
        model = FNO2dObserver(args.modes, args.modes, args.width, use_v_plane=args.use_v_plane).cuda()
    elif args.model_name == 'RNO2dObserverOld':
        model = RNO2dObserver(args.modes, args.modes, args.width, recurrent_index=args.recurrent_index, layer_num=args.layer_num).cuda()
    elif args.model_name == 'UNet':
        model = UNet(use_spectral_conv=args.use_spectral_conv).cuda()
    elif args.model_name == 'Transformer2D':
        model = SimpleTransformer(**args.model).cuda()
    else:
        raise NotImplementedError("Model not supported!")

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_path = './outputs/'
    output_path += args.path_name
    output_path += '_observer.mat'
    myloss = LpLoss(size_average=False)

    if not args.close_wandb:
        wandb.init(
            project=args.project_name + "_" + args.path_name,
            name=args.exp_name,
            config={
                "task info": "p-plane-to-v",
                "model_name": args.model_name,
                "file_name": args.path_name,
                "has_prev_press": True,
                "patches": False,
                "permute": True,
                "DATA_FOLDER": args.DATA_FOLDER,
                "ntrain": args.ntrain,
                "ntest": args.ntest,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "step_size": args.step_size,
                "gamma": args.gamma,
                "modes": args.modes,
                "width": args.width,
                "r": args.downsample_rate,
                "use_v_plane": args.use_v_plane,
                "use_patch": args.use_patch
                })

    best_loss = 10000000000000
    for ep in tqdm(range(args.epochs)):
        model.train()
        t1 = default_timer()
        train_l2, train_num = 0, 0
        for step, (p_plane, v_plane) in enumerate(tqdm(train_loader)):
            p_plane, v_plane = p_plane.cuda().float(), v_plane.cuda().float()
            if args.recurrent_model:
                p_plane = p_plane.reshape(-1, args.timestep, args.x_range, args.y_range, 1)
                v_plane = v_plane.reshape(-1, args.timestep, args.x_range, args.y_range, 1)
                v_plane = v_plane[:, args.recurrent_index, :, :, :]  # select the predict element
                args.batch_size = v_plane.shape[0]
            elif args.using_transformer:
                p_plane = p_plane.reshape(-1, args.timestep, args.x_range, args.y_range, 1)
            else:
                p_plane = p_plane.reshape(-1, args.x_range, args.y_range, 1)
                v_plane = v_plane.reshape(-1, args.x_range, args.y_range, 1)
            train_num += len(v_plane)
            optimizer.zero_grad()
            import pdb; pdb.set_trace()
            out = model(p_plane, v_plane)
            out = out.reshape(-1, args.x_range, args.y_range)
            out_decoded = train_dataset.v_norm.cuda_decode(out)
            v_plane = v_plane.squeeze()
            v_plane_decoded = train_dataset.v_norm.cuda_decode(v_plane)
            # loss = myloss(out.view(args.batch_size, -1), v_plane.view(args.batch_size, -1))
            loss = myloss(out_decoded.view(args.batch_size, -1), v_plane_decoded.view(args.batch_size, -1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
            metrics = {"train/train_loss": loss.item(), 
                    "train/epoch": (step + 1 + (n_steps_per_epoch * ep)) / n_steps_per_epoch}
            if step + 1 < n_steps_per_epoch and not args.close_wandb:
                # Log train metrics to wandb 
                wandb.log(metrics)

        model.eval()
        test_l2, test_num = 0.0, 0
        with torch.no_grad():
            for p_plane, v_plane in test_loader:
                p_plane, v_plane = p_plane.cuda().float(), v_plane.cuda().float()
                if args.recurrent_model:
                    p_plane = p_plane.reshape(-1, args.timestep, args.x_range, args.y_range, 1)
                    v_plane = v_plane.reshape(-1, args.timestep, args.x_range, args.y_range, 1)
                    v_plane = v_plane[:, args.recurrent_index, :, :, :]
                    args.batch_size = v_plane.shape[0]
                elif args.using_transformer:
                    p_plane = p_plane.reshape(-1, args.timestep, args.x_range, args.y_range, 1)
                else:
                    p_plane = p_plane.reshape(-1, args.x_range, args.y_range, 1)
                    v_plane = v_plane.reshape(-1, args.x_range, args.y_range, 1)
                test_num += len(v_plane)
                out = model(p_plane, v_plane)
                out = out.reshape(-1, args.x_range, args.y_range)
                if args.using_transformer:
                    p_plane = p_plane.reshape(-1, args.x_range, args.y_range, 1)
                elif args.recurrent_model:
                    p_plane = p_plane[:, args.recurrent_index, :, :, :]
                out_decoded = train_dataset.v_norm.cuda_decode(out)
                v_plane = v_plane.squeeze()
                p_plane_decoded = train_dataset.p_norm.cuda_decode(p_plane)
                v_plane_decoded = train_dataset.v_norm.cuda_decode(v_plane)
                # test_loss = myloss(out.view(args.batch_size, -1), v_plane.view(args.batch_size, -1)).item()
                test_loss = myloss(out_decoded.view(args.batch_size, -1), v_plane_decoded.view(args.batch_size, -1)).item()
                test_l2 += test_loss
                test_metrics = {"test/test_loss": test_loss / args.batch_size}
                if not args.close_wandb:
                    wandb.log(test_metrics)

        train_l2/= train_num
        test_l2 /= test_num
        t2 = default_timer()
        if test_l2 < best_loss:
            best_loss = test_l2
            dat = {'x': p_plane_decoded.cpu().numpy(), 'pred': out_decoded.cpu().numpy(), 'y': v_plane_decoded.cpu().numpy(),}
            if not args.close_wandb:
                vis_diagram(dat)
            model_save_p = f"./outputs/{args.path_name}_{args.exp_name}.pth"
            torch.save(model, model_save_p)
            print(f"Best model saved at {model_save_p}!")
        print(f"epoch: {ep}, time passed: {t2-t1}, train loss: {train_l2}, test loss: {test_l2}, best loss: {best_loss}.")
        avg_metrics = {"train/avg_train_loss": train_l2,
                    "test/avg_test_loss": test_l2,
                    "test/best_loss": best_loss}
        
        if not args.close_wandb:
            wandb.log(avg_metrics)
            
    if not args.close_wandb:
        vis_diagram(dat)
        wandb.finish()


if __name__ == '__main__':
    args = parse_arguments()
    loaded_args = load_arguments_from_yaml(args.train_yaml)
    args = merge_args_with_yaml(args, loaded_args)
    main(args)
