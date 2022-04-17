import torch.nn as nn


class NodeFeatureEncoder(nn.Module):
    def __init__(self, device, ckpt=None):
        super(NodeFeatureEncoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.to(device)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x):
        return self.layer(x)


class EdgeFeatureEncoder(nn.Module):
    def __init__(self, device, ckpt=None):
        super(EdgeFeatureEncoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU()
        )
        self.to(device)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x):
        return self.layer(x)


class EdgePredictor(nn.Module):
    def __init__(self, device, ckpt=None):
        super(EdgePredictor, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.to(device)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x_edge):
        return self.pred(x_edge)
