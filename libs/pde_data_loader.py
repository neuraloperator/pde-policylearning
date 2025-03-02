import os
import torch
import numpy as np
from torch.utils.data import Dataset
from libs.utilities3 import *


class PDEDataset(Dataset):
    def __init__(self, args, data_folder, data_index, downsample_rate, x_range, y_range, use_patch=False, full_field=False):
        super().__init__()
        self.data_folder = data_folder
        self.downsample_rate, self.x_range, self.y_range = downsample_rate, x_range, y_range
        self.metadata = np.load(os.path.join(data_folder, 'metadata.npy'), allow_pickle=True).tolist()
        self.file_list = os.listdir(data_folder)
        if 'P_planes' in self.metadata.keys():
            p_plane_name = 'P_planes'
            v_plane_name = 'V_planes'
        elif 'P_plane' in self.metadata.keys():
            p_plane_name = 'P_plane'
            v_plane_name = 'V_plane'
        else:
            raise RuntimeError("Not recognized key name!")
        self.p_plane_files = sorted([onef for onef in self.file_list if p_plane_name in onef])
        self.v_plane_files = sorted([onef for onef in self.file_list if v_plane_name in onef])
        self.p_plane_mean, self.p_plane_std = self.metadata[p_plane_name]['mean'], self.metadata[p_plane_name]['std']
        # self.p_plane_max, self.p_plane_min = self.metadata[p_plane_name]['max'], self.metadata[p_plane_name]['min']
        self.v_plane_mean, self.v_plane_std = self.metadata[v_plane_name]['mean'], self.metadata[v_plane_name]['std']
        # self.v_plane_max, self.v_plane_min = self.metadata[v_plane_name]['max'], self.metadata[v_plane_name]['min']
        self.data_index = data_index
        self.data_length = len(self.data_index)
        self.use_patch = use_patch
        if self.use_patch:
            p_mean = self.p_plane_mean.reshape(-1, self.x_range, self.y_range).mean(0)
            p_std = self.p_plane_std.reshape(-1, self.x_range, self.y_range).mean(0)
            v_mean = self.v_plane_mean.reshape(-1, self.x_range, self.y_range).mean(0)
            v_std = self.v_plane_std.reshape(-1, self.x_range, self.y_range).mean(0)
        else:
            p_mean = self.p_plane_mean[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
            p_std = self.p_plane_std[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
            v_mean = self.v_plane_mean[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
            v_std = self.v_plane_std[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
        
        self.p_norm = NormalizerGivenMeanStd(p_mean, p_std)
        self.v_norm = NormalizerGivenMeanStd(v_mean, v_std)
        # self.p_norm = RangeNormalizerGivenMinMax(self.p_plane_min, self.p_plane_max)
        # self.v_norm = RangeNormalizerGivenMinMax(self.v_plane_min, self.v_plane_max)
        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        cur_index = self.data_index[index]
        p_plane = np.load(os.path.join(self.data_folder, self.p_plane_files[cur_index]))
        p_plane = torch.tensor(p_plane)
        if self.use_patch:
            p_plane = p_plane.reshape(-1, self.x_range, self.y_range)
        else:
            p_plane = p_plane[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
        p_plane = self.p_norm.encode(p_plane)
        p_plane = p_plane.unsqueeze(-1)
        v_plane = np.load(os.path.join(self.data_folder, self.v_plane_files[cur_index]))
        v_plane = torch.tensor(v_plane)
        if self.use_patch:
            v_plane = v_plane.reshape(-1, self.x_range, self.y_range)
        else:
            v_plane = v_plane[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
        v_plane = self.v_norm.encode(v_plane)
        v_plane = v_plane.unsqueeze(-1)
        return p_plane, v_plane


class SequentialPDEDataset(Dataset):
    """
    returns [timestep, height, width, dim] with dim = 1.
    """
    def __init__(self, args, data_folder, data_index, downsample_rate, x_range, y_range, use_patch=False, full_field=True):
        super().__init__()
        self.timestep = args.model_timestep
        self.data_folder = data_folder
        self.full_field = full_field
        self.downsample_rate, self.x_range, self.y_range = downsample_rate, x_range, y_range
        self.metadata = np.load(os.path.join(data_folder, 'metadata.npy'), allow_pickle=True).tolist()
        self.file_list = os.listdir(data_folder)
        u_field_name, v_field_name, w_field_name = 'U_field', 'V_field', 'W_field'
        self.u_field_files = sorted([onef for onef in self.file_list if u_field_name in onef])
        self.v_field_files = sorted([onef for onef in self.file_list if v_field_name in onef])
        self.w_field_files = sorted([onef for onef in self.file_list if w_field_name in onef])
        print("In sequential dataset, the options downsample_rate, x_range and y_range are not supported!")
        self.v_field_mean, self.v_field_std = self.metadata[v_field_name]['mean'], self.metadata[v_field_name]['std']
        self.data_index = data_index
        self.data_length = len(self.data_index)
        self.use_patch = use_patch
        if self.use_patch:
            p_mean = self.p_plane_mean.reshape(-1, self.x_range, self.y_range).mean(0)
            p_std = self.p_plane_std.reshape(-1, self.x_range, self.y_range).mean(0)
            v_mean = self.v_plane_mean.reshape(-1, self.x_range, self.y_range).mean(0)
            v_std = self.v_plane_std.reshape(-1, self.x_range, self.y_range).mean(0)
        else:
            p_mean = self.p_plane_mean[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
            p_std = self.p_plane_std[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
            v_mean = self.v_plane_mean[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
            v_std = self.v_plane_std[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
        
        self.p_norm = NormalizerGivenMeanStd(p_mean, p_std)
        self.v_norm = NormalizerGivenMeanStd(v_mean, v_std)
        
    def __len__(self):
        return self.data_length // self.timestep

    def __getitem__(self, index):
        sequential_p, sequential_v = [], []
        for cur_t in range(self.timestep):
            cur_index = self.data_index[index * self.timestep + cur_t]
            p_plane = np.load(os.path.join(self.data_folder, self.p_plane_files[cur_index]))
            p_plane = torch.tensor(p_plane)
            if self.use_patch:
                p_plane = p_plane.reshape(-1, self.x_range, self.y_range)
            else:
                p_plane = p_plane[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
            p_plane = self.p_norm.encode(p_plane)
            v_plane = np.load(os.path.join(self.data_folder, self.v_plane_files[cur_index]))
            v_plane = torch.tensor(v_plane)
            if self.use_patch:
                v_plane = v_plane.reshape(-1, self.x_range, self.y_range)
            else:
                v_plane = v_plane[::self.downsample_rate, ::self.downsample_rate][:self.x_range, :self.y_range]
            v_plane = self.v_norm.encode(v_plane)
            sequential_p.append(p_plane)
            sequential_v.append(v_plane)
        sequential_p = torch.stack(sequential_p)
        sequential_v = torch.stack(sequential_v)
        return sequential_p, sequential_v


class FullFieldNSDataset(Dataset):
    """
    returns [timestep, height, width, dim] with dim = 1.
    """
    def __init__(self, args, data_folder, data_index, plane_indexs, downsample_rate, x_range, y_range, use_patch=False, full_field=True):
        super().__init__()
        self.timestep = args.model_timestep
        self.data_folder = data_folder
        self.full_field = full_field
        self.downsample_rate, self.x_range, self.y_range = downsample_rate, x_range, y_range
        self.metadata = np.load(os.path.join(data_folder, 'metadata.npy'), allow_pickle=True).tolist()
        self.re = torch.tensor(self.metadata['re'])
        self.dpdx_all = self.metadata['U_field']['dpdx']
        self.file_list = os.listdir(data_folder)
        u_field_name, v_field_name, w_field_name = 'U_field', 'V_field', 'W_field'
        self.u_field_files = sorted([onef for onef in self.file_list if u_field_name in onef])
        self.v_field_files = sorted([onef for onef in self.file_list if v_field_name in onef])
        self.w_field_files = sorted([onef for onef in self.file_list if w_field_name in onef])
        #  In sequential dataset, the options downsample_rate, x_range and y_range are not supported.
        self.scale_factor = 1
        self.bound_v_mean, self.bound_v_std = self.metadata[v_field_name]['mean'][:, -1, :], self.metadata[v_field_name]['std'][:, -1, :] / self.scale_factor
        self.v_field_mean, self.v_field_std = self.metadata[v_field_name]['mean'][:, 1:-1, :], self.metadata[v_field_name]['std'][:, 1:-1, :]
        self.data_index = data_index
        self.data_length = len(self.data_index)
        self.plane_indexs = plane_indexs # predict values at these planes
        self.bound_v_norm = NormalizerGivenMeanStd(self.bound_v_mean, self.bound_v_std)
        self.v_field_norm = self.bound_v_norm
        p_plane_name = 'P_planes'
        self.p_plane_mean, self.p_plane_std = self.metadata[p_plane_name]['mean'], self.metadata[p_plane_name]['std']
        self.p_plane_norm = NormalizerGivenMeanStd(self.p_plane_mean, self.p_plane_std)
        
    def __len__(self):
        return self.data_length // self.timestep

    def __getitem__(self, index):
        seq_v_plane, seq_v_field = [], []
        seq_u, seq_v, seq_w, seq_dpdx, seq_re = [], [], [], [], []
        for cur_t in range(self.timestep):
            cur_index = self.data_index[index * self.timestep + cur_t]
            one_dpdx = self.dpdx_all[cur_index]
            seq_dpdx.append(one_dpdx)
            all_v_field = torch.tensor(np.load(os.path.join(self.data_folder, self.v_field_files[cur_index])))
            seq_v.append(all_v_field)
            all_u_field = torch.tensor(np.load(os.path.join(self.data_folder, self.u_field_files[cur_index])))
            seq_u.append(all_u_field)
            all_w_field = torch.tensor(np.load(os.path.join(self.data_folder, self.w_field_files[cur_index])))
            seq_w.append(all_w_field)
            v_plane = self.bound_v_norm.encode(all_v_field[:, -1, :])
            target_v_field = []
            for plane_id in self.plane_indexs:
                cur_v_field = self.v_field_norm.encode(all_v_field[:, plane_id, :])  # [x, y]
                target_v_field.append(cur_v_field)
            target_v_field = torch.stack(target_v_field)
            seq_v_plane.append(v_plane)
            seq_v_field.append(target_v_field)
            seq_re.append(self.re)
        seq_v_plane = torch.stack(seq_v_plane)  # [T, X, Y]
        seq_v_field = torch.stack(seq_v_field)  # [T, P (plane num), X, Y]
        seq_u = torch.stack(seq_u)
        seq_v = torch.stack(seq_v)
        seq_w = torch.stack(seq_w)
        seq_re = torch.tensor(seq_re)
        seq_dpdx = torch.tensor(seq_dpdx)
        return seq_v_plane, seq_v_field, seq_u, seq_v, seq_w, seq_re, seq_dpdx