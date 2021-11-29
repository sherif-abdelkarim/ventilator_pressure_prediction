import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class VentilatorDataset(Dataset):
    def __init__(self, annotations_file, idx, device):
        self.data = pd.read_csv(annotations_file)
        self.data = self.data.iloc[idx].reset_index(drop=True)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        R = row['R']/50.
        C = row['C']/50.
        u_in = row['u_in']/100.
        u_out = row['u_out']
        input_ = np.stack([R, C, u_in, u_out])
        target = row['pressure']

        input_ = torch.tensor(input_).float().to(self.device)
        target = torch.tensor(target).to(self.device)
        return input_, target
