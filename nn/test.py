### TORCH ###

import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

### PARAMETERS ###

batch_size = 128
lr = 0.001
validation_split = 0.2
n_epochs = 20

### DATALOADER ###

from torch.utils.data import DataLoader, random_split
from dataset import NextWaveDataset

dataset = {
    'train': NextWaveDataset('../results/train_clean.csv'),
    'test': NextWaveDataset('../results/test_clean.csv')
}

loader = {k: DataLoader(v, batch_size, shuffle=True, num_workers=1) for k, v in dataset.items()}

### MODEL ###

from torch import nn, optim
from model import FCNet

net = FCNet(len(dataset['test'].data[0]))

net.load_state_dict(
    torch.load('../checkpoint/epp16AUC0.981.pt')
)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

### PREDICT ###

net.eval()
predictions = []
for i, (traj_id, data, label) in enumerate(loader['test']):
    out = net(data)
    for t, p in zip(traj_id, out):
        predictions += [(t, round(p.item()))]

with open('../results/nn_submission.csv','w') as f:
    f.write("id,target\n")
    for t, p in predictions:
        f.write(str(t) + "," + str(p) + "\n")