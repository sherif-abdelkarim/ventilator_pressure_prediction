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
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--model', type=str, default='mlp', choices=['lstm', 'bi_lstm', 'transformer', 'mlp'])
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--data_path', type=str, default='./data/train.csv')
parser.add_argument('--split_path', type=str, default='./data/split_breath_id.json')


args = parser.parse_args()
print(args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: {}".format(device))

if device == 'cuda':
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

checkpoint = args.path

criterion = nn.L1Loss()

if args.model == 'lstm':
    VentilatorDataset = VentilatorDatasetLSTM
elif args.model == 'transformer':
    VentilatorDataset = VentilatorDatasetLSTM
elif args.model == 'mlp':
    VentilatorDataset = VentilatorDatasetMLP
else:
    raise NotImplementedError

test_dataset = VentilatorDataset(args.data_path, device=device)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

print('test iterations', len(test_loader))

if args.model == 'lstm':
    net = LSTM(in_features=test_dataset.inputs.shape[-1], bidirectional=False, out_features=1).to(device)
elif args.model == 'bi_lstm':
    net = LSTM(in_features=test_dataset.inputs.shape[-1], bidirectional=True, out_features=1).to(device)
elif args.model == 'transformer':
    net = Transformer(in_features=4, out_features=1).to(device)
elif args.model == 'mlp':
    net = MLP(in_features=4, out_features=1).to(device)
else:
    raise NotImplementedError

total_test_loss = 0
net.eval()
with torch.no_grad():
    for i, (inputs, target) in enumerate(test_loader):
        output = net(inputs).squeeze(dim=-1)
        loss = criterion(output, target)
        total_test_loss += loss.item()
    avg_test_loss = total_test_loss/len(test_loader)
print('Testing done. Loss: {:.4f}'.format(avg_test_loss))
