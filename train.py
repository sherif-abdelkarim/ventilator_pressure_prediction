import torch
import argparse
import datetime
import os
import json

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import VentilatorDatasetMLP, VentilatorDatasetLSTM
from models.lstm import LSTM
from models.transformer import Transformer
from models.mlp import MLP

# Read arguments and setup device
parser = argparse.ArgumentParser(description='Ventilator Pressure Prediciton Project')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--model', type=str, default='mlp', choices=['lstm', 'transformer', 'mlp'])
parser.add_argument('--data_path', type=str, default='./data/train.csv')
parser.add_argument('--split_path', type=str, default='./data/split_breath_id.json')


args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: {}".format(device))

# Creating results/logs saving folder
saving_dir = os.path.join('./experiments', str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
if not os.path.isdir(saving_dir):
    os.makedirs(saving_dir)

with open(os.path.join(saving_dir, 'config.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


criterion = nn.MSELoss()
l1_loss = nn.L1Loss()

split = json.load(open(args.split_path))
if args.model == 'lstm':
    VentilatorDataset = VentilatorDatasetLSTM
elif args.model == 'transformer':
    VentilatorDataset = VentilatorDatasetLSTM
elif args.model == 'mlp':
    VentilatorDataset = VentilatorDatasetMLP
else:
    raise NotImplementedError

train_dataset = VentilatorDataset(args.data_path, split['train_idx'], device)
valid_dataset = VentilatorDataset(args.data_path, split['valid_idx'], device)
print('train dataset', len(train_dataset))
print('valid dataset', len(valid_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

print('train iterations', len(train_loader))
print('valid iterations', len(valid_loader))

if args.model == 'lstm':
    # net = LSTM(in_features=4, out_features=1)
    # net = nn.LSTM(input_size=train_dataset.inputs.shape[-1], hidden_size=128, num_layers=1, batch_first=True)
    net = LSTM(in_features=train_dataset.inputs.shape[-1], out_features=1).to(device)
elif args.model == 'transformer':
    net = Transformer(in_features=4, out_features=1).to(device)
elif args.model == 'mlp':
    net = MLP(in_features=4, out_features=1).to(device)
else:
    raise NotImplementedError

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=2)

for epoch in range(args.epochs):
    total_loss = 0
    total_error = 0
    net.train()
    for i, (inputs, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(inputs).squeeze(dim=-1)
        loss = criterion(output, target)
        total_loss = loss.item()
        loss.backward()
        optimizer.step()

        error = l1_loss(output, target)
        total_error += error
        # if i % 1000 == 0:
        #     print('Training Epoch {}, Batch {}/{}: MSE: {}, MAE: {}'.format(epoch + 1, i, len(train_loader), loss, error))
    print('Epoch {},  MSE: {}, MAE: {}'.format(epoch + 1, total_loss/len(train_loader), total_error/len(train_loader)))

    scheduler.step(total_error/len(train_loader))
    print('Validating...')
    total_loss = 0
    total_error = 0
    net.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(valid_loader):
            optimizer.zero_grad()
            output = net(inputs).squeeze(dim=-1)
            loss = criterion(output, target)
            error = l1_loss(output, target)
            total_loss += loss
            total_error += error

        print('Validation after Epoch {}: MSE: {}, MAE: {}'.format(epoch + 1,
                                                                   total_loss/len(valid_loader),
                                                                   total_error/len(valid_loader)))
