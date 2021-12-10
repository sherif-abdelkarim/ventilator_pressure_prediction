import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class VentilatorDatasetMLP(Dataset):
    def __init__(self, annotations_file, idx=None, device='cuda'):
        self.data = pd.read_csv(annotations_file)
        if idx is not None:
            self.data = self.data.iloc[idx].reset_index(drop=True)
        self.data['R'] = self.data['R']/50
        self.data['C'] = self.data['C']/50
        self.data['u_in'] = self.data['u_in']/100
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        R = row['R']
        C = row['C']
        u_in = row['u_in']
        u_out = row['u_out']
        input_ = np.stack([R, C, u_in, u_out])
        target = row['pressure']

        input_ = torch.tensor(input_).float().to(self.device)
        target = torch.tensor(target).float().to(self.device)
        return input_, target


class VentilatorDatasetLSTM(Dataset):
    def __init__(self, annotations_file, idx=None, device='cuda'):
        self.data = pd.read_csv(annotations_file)
        if idx is not None:
            self.data = self.data.loc[self.data['breath_id'].isin(idx)].reset_index(drop=True)
        self.data['R'] = self.data['R']/50
        self.data['C'] = self.data['C']/50
        self.data['u_in'] = self.data['u_in']/100
        self.inputs = self.data[['R', 'C', 'u_in', 'u_out']].to_numpy().reshape(-1, 80, 4)
        self.targets = self.data[['pressure']].to_numpy().reshape(-1, 80)
        self.device = device

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        target = self.targets[idx]

        input_ = torch.tensor(input_).float().to(self.device)
        target = torch.tensor(target).float().to(self.device)
        return input_, target
