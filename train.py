import torch
import argparse
import datetime
import os
import json

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from dataset import VentilatorDataset
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
parser.add_argument('--split_path', type=str, default='./data/split.json')


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

if args.model == 'lstm':
    # net = LSTM(in_features=4, out_features=1)
    net = nn.LSTM(input_size=4, hidden_size=128, num_layers=1)
elif args.model == 'tranformer':
    net = Transformer(in_features=4, out_features=1)
elif args.model == 'mlp':
    net = MLP(in_features=4, out_features=1)
else:
    raise NotImplementedError

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

criterion = nn.MSELoss()
l1_loss = nn.L1Loss()

split = json.load(open(args.split_path))
train_dataset = VentilatorDataset(args.data_path, split['train_idx'], device)
valid_dataset = VentilatorDataset(args.data_path, split['valid_idx'], device)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

print('train iterations', len(train_loader))
print('valid iterations', len(valid_loader))

for epoch in range(args.epochs):
    net.train()
    for i, (inputs, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Training Epoch {}, Batch {}/{}: Loss: {}'.format(epoch, i, len(train_loader), loss))

    print('Validating')
    total_loss = 0
    total_error = 0
    net.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(valid_loader):
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output.squeeze(), target)
            error = l1_loss(output.squeeze(), target)
            total_loss += loss
            total_error += error
        print('Validation after Epoch {}: MSE: {}, MAE: {}'.format(epoch,
                                                                   total_loss/len(valid_loader),
                                                                   total_error/len(valid_loader)))
