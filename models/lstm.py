import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, in_features, out_features, num_hidden, bidirectional=False):
        super(LSTM, self).__init__()
        # hidden = [512, 256, 128]
        self.num_hidden = num_hidden
        hidden = [128 * (2**i) for i in range(num_hidden)][::-1]
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

        # self.lstm1 = nn.LSTM(in_features, hidden[0],
        #                      batch_first=True, bidirectional=bidirectional)
        self.lstms = nn.ModuleList([nn.LSTM(in_features, hidden[0], batch_first=True, bidirectional=bidirectional)])
        for i in range(1, len(hidden)):
            self.lstms.append(nn.LSTM(mutliplier * hidden[i-1], hidden[i], batch_first=True, bidirectional=bidirectional))

        # self.lstm2 = nn.LSTM(mutliplier * hidden[0], hidden[1],
        #                      batch_first=True, bidirectional=bidirectional)
        # self.lstm2 = nn.LSTM(mutliplier * hidden[1], hidden[2],
        #                      batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(mutliplier * hidden[-1], out_features)

    def forward(self, x):
        # x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)
        # x, _ = self.lstm3(x)
        # x, _ = self.lstm4(x)
        # x = self.fc1(x)
        # x = self.selu(x)
        # x = self.fc2(x)
        for i in range(self.num_hidden):
            x, _ = self.lstms[i](x)
        # x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)
        x = self.fc(x)
        return x
