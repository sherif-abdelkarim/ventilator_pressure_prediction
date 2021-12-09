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
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--reduce_on_plateau', type=bool, action='store_true', default=False)
parser.add_argument('--model', type=str, default='mlp', choices=['lstm', 'bi_lstm', 'transformer', 'mlp'])
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
# Creating results/logs saving folder
saving_dir = os.path.join('./experiments', str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
if not os.path.isdir(saving_dir):
    os.makedirs(saving_dir)
print('Saving in:', saving_dir)

with open(os.path.join(saving_dir, 'log.txt'), 'a') as f:
    print(args, file=f)

checkpoint_dir = os.path.join(saving_dir, 'checkpoints')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(os.path.join(saving_dir, 'config.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


# criterion = nn.MSELoss()
criterion = nn.L1Loss()
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
    net = LSTM(in_features=train_dataset.inputs.shape[-1], bidirectional=False, out_features=1).to(device)
elif args.model == 'bi_lstm':
    net = LSTM(in_features=train_dataset.inputs.shape[-1], bidirectional=True, out_features=1).to(device)
elif args.model == 'transformer':
    net = Transformer(in_features=4, out_features=1).to(device)
elif args.model == 'mlp':
    net = MLP(in_features=4, out_features=1).to(device)
else:
    raise NotImplementedError

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
if args.reduce_on_plateau:
    print('Using ReduceLROnPlateau LR scheduler.')
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=10, factor=0.5)
best_loss = float('inf')
best_epoch = -1
for epoch in range(args.epochs):
    total_train_loss = 0
    # total_error = 0
    net.train()
    for i, (inputs, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(inputs).squeeze(dim=-1)
        loss = criterion(output, target)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # error = l1_loss(output, target)
        # total_error += error
        # if i % 1000 == 0:
        #     print('Training Epoch {}, Batch {}/{}: MSE: {}, MAE: {}'.format(epoch + 1, i, len(train_loader), loss, error))
    avg_train_loss = total_train_loss/len(train_loader)
    print('Epoch {}:  Loss: {:.4f}'.format(epoch + 1, avg_train_loss))
    with open(os.path.join(saving_dir, 'log.txt'), 'a') as f:
        print('Epoch {}:  Loss: {:.4f}'.format(epoch + 1, avg_train_loss), file=f)

    if args.reduce_on_plateau:
        scheduler.step(avg_train_loss)
    total_valid_loss = 0
    # total_error = 0
    net.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(valid_loader):
            optimizer.zero_grad()
            output = net(inputs).squeeze(dim=-1)
            loss = criterion(output, target)
            # error = l1_loss(output, target)
            total_valid_loss += loss.item()
            # total_error += error
        avg_valid_loss = total_valid_loss/len(valid_loader)

        print('Validation after Epoch {}: Loss: {:.4f}'.format(epoch + 1, avg_valid_loss))
        with open(os.path.join(saving_dir, 'log.txt'), 'a') as f:
            print('Validation after Epoch {}: Loss: {:.4f}'.format(epoch + 1, avg_valid_loss), file=f)

        if avg_valid_loss < best_loss:
            print('New best model found, current best loss is: {:.4f}'.format(avg_valid_loss))
            with open(os.path.join(saving_dir, 'log.txt'), 'a') as f:
                print('New best model found, current best loss is: {:.4f}'.format(avg_valid_loss), file=f)

            best_loss = avg_valid_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(checkpoint_dir, 'best.ptDict'))
print('Training done. Best loss: {:.4f}, at epoch: {:.4f}'.format(best_loss, best_epoch))
