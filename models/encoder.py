import torch.nn as nn


class NodeFeatureEncoder(nn.Module):
    def __init__(self, device, ckpt=None):
        super(NodeFeatureEncoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
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
            nn.Linear(8, 6)
        )
        self.to(device)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x):
        return self.layer(x)
