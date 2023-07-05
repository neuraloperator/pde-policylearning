import os
import torch
import numpy as np
from torch.utils.data import Dataset
from libs.utilities3 import *


class PDEDataset(Dataset):
    def __init__(self, args, data_folder, data_index, downsample_rate, x_range, y_range, use_patch=False):
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
    def __init__(self, args, data_folder, data_index, downsample_rate, x_range, y_range, use_patch=False):
        self.timestep = args.timestep
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
