import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, in_features, out_features, bidirectional=False):
        super(LSTM, self).__init__()
        hidden = [128]
        mutliplier = 2 if bidirectional else 1
        # self.lstm1 = nn.LSTM(in_features, hidden[0],
        #                      batch_first=True, bidirectional=bidirectional)
        # self.lstm2 = nn.LSTM(mutliplier * hidden[0], hidden[1],
        #                      batch_first=True, bidirectional=bidirectional)
        # self.lstm3 = nn.LSTM(mutliplier * hidden[1], hidden[2],
        #                      batch_first=True, bidirectional=bidirectional)
        # self.lstm4 = nn.LSTM(mutliplier * hidden[2], hidden[3],
        #                      batch_first=True, bidirectional=bidirectional)
        # self.fc1 = nn.Linear(mutliplier * hidden[3], 50)
        # self.selu = nn.SELU()
        # self.fc2 = nn.Linear(50, out_features)

        self.lstm = nn.LSTM(in_features, hidden[0],
                             batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(mutliplier * hidden[0], out_features)

    def forward(self, x):
        # x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)
        # x, _ = self.lstm3(x)
        # x, _ = self.lstm4(x)
        # x = self.fc1(x)
        # x = self.selu(x)
        # x = self.fc2(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
