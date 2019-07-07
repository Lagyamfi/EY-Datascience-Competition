### TORCH ###

import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

import csv
from torch.utils.data import Dataset

class NextWaveDataset(Dataset):
    def __init__(self, datafile):
        with open(datafile) as f:
            reader = csv.reader(f)
            header = next(reader)

            feature_dim = len(header) - 2 if 'label.inc' in header else len(header) - 1

            self.traj_id, self.data, self.label = [], [], []
            for row in reader:
                self.traj_id += [row[header.index('trajectory_id')]]

                if 'label.inc' in header:
                    self.data += [[
                        float(column) for i, column in enumerate(row)
                        if i != header.index('trajectory_id') and i != header.index('label.inc')
                    ]]
                    self.label += [float(row[header.index('label.inc')])]

                else:
                    self.data += [[
                        float(column) if len(column) > 0 else 0 for i, column in enumerate(row)
                        if i != header.index('trajectory_id')
                    ]]
                    self.label += [0]

    def __len__(self):
        return len(self.traj_id)

    def __getitem__(self, idx):
        return self.traj_id[idx], Tensor(self.data[idx]), self.label[idx]