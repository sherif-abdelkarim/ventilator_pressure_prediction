import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(in_features, out_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Linear(hidden_features, out_features),
        )
    def forward(self, x):
        out = self.mlp(x)
        # out = nn.functional.sigmoid(out)
        return out
