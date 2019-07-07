from sklearn.metrics import roc_auc_score

### TORCH ###

import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

### PARAMETERS ###

batch_size = 248
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

split_len = [round((1-validation_split) * len(dataset['train'])), round(validation_split * len(dataset['train']))]
dataset['train'], dataset['val'] = random_split(dataset['train'], lengths=split_len)

loader = {k: DataLoader(v, batch_size, shuffle=True, num_workers=1) for k, v in dataset.items()}

### MODEL ###

from torch import nn, optim
from model import FCNet

net = FCNet(len(dataset['test'].data[0]))
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

### TRAINING ###

best_auc = 0
for epoch in range(n_epochs):
    net.train()
    running_loss, S = 0, 0
    for i, (traj_id, data, label) in enumerate(loader['train']):
        optimizer.zero_grad()

        out = net(data)
        loss = criterion(out.view(-1), label.float())

        running_loss += loss
        S += len(traj_id)

        loss.backward()
        optimizer.step()
    train_loss = running_loss / S

    net.eval()
    running_loss, running_auc, S, j= 0, 0, 0, 0
    for i, (traj_id, data, label) in enumerate(loader['val']):
        out = net(data)
        loss = criterion(out.view(-1), label.float())
        running_auc += roc_auc_score(label.detach(), out.detach())
        running_loss += loss
        S += len(traj_id)
        j += 1
    val_loss, val_auc = running_loss / S, running_auc / j

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(net.state_dict(), '../checkpoint/' + 'epp' + str(epoch) + 'AUC' + str(val_auc)[:5] + '.pt')

    print(epoch, train_loss.item(), val_loss.item(), val_auc)
